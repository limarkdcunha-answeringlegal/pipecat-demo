import asyncio
import json
import os
import signal
from functools import partial

import httpx
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.frames.frames import LLMUpdateSettingsFrame

from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.settings import LLMSettings
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService


from pipecat.workers.runner import WorkerRunner
from pipecat_flows import FlowManager, FlowsFunctionSchema, NodeConfig

from api_client import fetch_case_type_questions, fetch_company_case_types
from bot_config import (
    DEFAULT_VOICE_ID,
)

load_dotenv(override=True)


class TransferRegistry:
    """
    Coordinates state between the caller bot and the attorney bot WebSocket, which run as completely separate async contexts.
    The registry is the shared lookup bridge between them, keyed by caller call_sid.
    """

    def __init__(self):
        self._transfers: dict[str, dict] = {}

    def register(self, call_sid: str, case_summary: str) -> dict:
        """Called by _warm_transfer to create and store transfer state."""
        state = {
            "accepted": asyncio.Event(),
            "declined": asyncio.Event(),
            "attorney_sid": [],
            "case_summary": case_summary,
        }
        self._transfers[call_sid] = state
        return state

    def get(self, call_sid: str) -> dict | None:
        """Called by the attorney bot to find its corresponding transfer state."""
        return self._transfers.get(call_sid)

    def remove(self, call_sid: str) -> None:
        """Called in finally block of _warm_transfer to clean up."""
        self._transfers.pop(call_sid, None)


transfer_registry = TransferRegistry()


# In-memory transfer context store
# Bot 1 (caller leg) stores full collected data here before transferring.
# Bot 2 (attorney leg) reads it to brief the attorney with everything collected.
# TODO: Dict for now, should be replaced with Redis in future
_transfer_contexts: dict[str, dict] = {}


def store_transfer_context(call_sid: str, context: dict) -> None:
    """Bot 1 stores full collected intake data before initiating transfer."""
    _transfer_contexts[call_sid] = context
    logger.info(f"Stored transfer context for {call_sid}: {list(context.keys())}")


def fetch_transfer_context(call_sid: str) -> dict:
    """Bot 2 reads full context to brief the attorney. Pops to auto-clean."""
    return _transfer_contexts.pop(call_sid, {})


def _pick_transfer_rule(transfer_rules: list[dict]) -> dict | None:
    """Pick highest-priority patch-enabled transfer rule."""
    enabled = [r for r in transfer_rules if r.get("patch_enabled")]
    if not enabled:
        return None
    return sorted(enabled, key=lambda r: r.get("priority", 99))[0]


# This needs to be worked upon
async def extract_answers_from_description(
    description: str, questions: list[dict]
) -> dict:
    """Run a single LLM pass over the caller's free-form intro and pre-fill any
    question answers it can confidently extract. Returns {field_id: answer}.

    Only fills fields the caller clearly answered — leaves the rest for the bot
    to ask normally. Conservative on purpose: a wrong guess is worse than a
    repeated question.
    """
    if not description.strip() or not questions:
        return {}

    # Only consider enabled questions; give the model the field_id + question text
    askable = [
        {"field_id": q["field_id"], "question": q["question"]}
        for q in questions
        if q.get("is_enabled", True)
    ]
    if not askable:
        return {}

    field_lines = "\n".join(f'- {q["field_id"]}: "{q["question"]}"' for q in askable)
    system = (
        "You extract answers to intake questions from what a caller said. "
        "Return ONLY a JSON object mapping field_id to the answer, and ONLY for "
        "questions the caller has CLEARLY and EXPLICITLY answered in their statement. "
        "If a question is not clearly answered, OMIT that field_id entirely — never guess. "
        "Answers must be the caller's actual information, not paraphrased advice. "
        'Example output: {"first_name": "John", "state": "Texas"}'
    )
    user = (
        f'Caller said:\n"{description}"\n\n'
        f"Intake questions (field_id: question):\n{field_lines}\n\n"
        "Return the JSON object of confidently-extracted answers."
    )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            extracted = json.loads(content)
    except Exception as e:
        logger.error(f"Answer extraction failed: {e}")
        return {}

    # Keep only valid field_ids with non-empty string values
    valid_ids = {q["field_id"] for q in askable}
    cleaned = {
        k: str(v).strip()
        for k, v in extracted.items()
        if k in valid_ids and str(v).strip()
    }
    if cleaned:
        logger.info(
            f"Pre-filled {len(cleaned)} answer(s) from intro: {list(cleaned.keys())}"
        )
    return cleaned


def make_save_node(collected: dict) -> NodeConfig:
    """Terminal node — thanks caller and ends the conversation."""
    return NodeConfig(
        name="save_and_end",
        # Reset tool_choice to 'auto' — this node has no tools, so a lingering
        # 'required' from the case-type node would 400 the OpenAI call.
        pre_actions=[{"type": "set_tool_choice", "choice": "auto"}],
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Thank the caller warmly, let them know their information has been received "
                    "and someone will be in touch shortly. Then end the call."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
        functions=[],
    )


async def handle_answer(
    args: dict,
    fm: FlowManager,
    *,
    questions: list,
    collected: dict,
    index: int,
    transfer_rule: dict | None,
    call_sid: str,
    firm: dict,
    case_types: list,
    q: dict,
    transfer_in_progress: list,
) -> tuple:
    answer = args.get("answer", "")

    # Save current field immediately
    collected[q["field_id"]] = answer
    logger.info(f"Q[{q['field_id']}] = {answer!r}")

    return answer, make_question_node(
        questions,
        collected,
        index + 1,
        transfer_rule,
        call_sid,
        firm,
        case_types,
        transfer_in_progress=transfer_in_progress,
    )


async def handle_question_transfer(
    args: dict,
    fm: FlowManager,
    *,
    transfer_rule: dict | None,
    call_sid: str,
    firm: dict,
    collected: dict,
    transfer_in_progress: list,
) -> tuple:
    if not transfer_rule:
        number = firm.get("transfer_number", "")
        name = firm.get("attorney_name", "the firm")
    else:
        number = transfer_rule.get("contact_phone", "")
        name = transfer_rule.get("contact_name", "the team")

    case_summary = collected.get("case_type_slug", "legal matter")
    from bot_twilio import _warm_transfer  # late import avoids circular dependency

    success = await _warm_transfer(
        call_sid, number, name, case_summary, collected, transfer_in_progress
    )

    if success:
        return "transferred", make_save_node(collected)

    return "transfer_failed", NodeConfig(
        name="attorney_unavailable",
        pre_actions=[{"type": "set_tool_choice", "choice": "auto"}],
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Tell the caller the attorney is currently unavailable and you were unable "
                    "to connect them. Apologize briefly and end the call."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
        functions=[],
    )


async def handle_pivot(
    args: dict,
    fm: FlowManager,
    *,
    firm: dict,
    collected: dict,
    questions: list,
    index: int,
    transfer_rule: dict | None,
    call_sid: str,
    case_types: list,
    transfer_in_progress: list,
) -> tuple:
    ctype_id = args.get("case_type_id", "")
    try:
        ct_data = await fetch_case_type_questions(firm["id"], ctype_id)
    except Exception as e:
        logger.error(f"Pivot fetch failed for {ctype_id}: {e}")
        return "pivot_failed", make_question_node(
            questions,
            collected,
            index,
            transfer_rule,
            call_sid,
            firm,
            case_types,
            transfer_in_progress=transfer_in_progress,
        )

    new_questions = ct_data.get("questions", [])
    new_transfer_rule = _pick_transfer_rule(ct_data.get("transfer_rules", []))
    new_collected = {
        "case_type_id": ctype_id,
        "case_type_slug": ct_data.get("case_type_slug", ""),
    }

    if not new_questions:
        return ctype_id, make_save_node(new_collected)

    return ctype_id, make_question_node(
        new_questions,
        new_collected,
        0,
        new_transfer_rule,
        call_sid,
        firm,
        case_types,
        transfer_in_progress=transfer_in_progress,
    )


def make_question_node(
    questions: list[dict],
    collected: dict,
    index: int,
    transfer_rule: dict | None,
    call_sid: str,
    firm: dict,
    case_types: list[dict] | None = None,
    *,
    transfer_in_progress: list,
) -> NodeConfig:
    """Build a node for the current question. Skips disabled/conditional-blocked questions.
    Returns make_save_node when all questions are exhausted.
    """
    case_types = case_types or []

    while index < len(questions):
        q = questions[index]
        if not q.get("is_enabled", True):
            index += 1
            continue
        cond_field = q.get("conditional_on_field_id", "")
        cond_value = str(q.get("conditional_on_value", ""))
        if cond_field and str(collected.get(cond_field, "")) != cond_value:
            index += 1
            continue
        # Skip questions already answered by background extraction
        if q["field_id"] in collected:
            logger.info(f"Skipping Q[{q['field_id']}] — already answered")
            index += 1
            continue
        break
    else:
        return make_save_node(collected)

    q = questions[index]
    options = q.get("options", [])
    options_hint = f" Options: {', '.join(str(o) for o in options)}." if options else ""

    functions = [
        FlowsFunctionSchema(
            name="record_answer",
            description=(
                "ALWAYS call this the instant the caller says anything — "
                "one word, partial answer, refusal, or off-topic response. "
                "You MUST call this before saying anything else. No exceptions. "
                "Do NOT respond conversationally, offer empathy, or ask follow-up questions. "
                "Just call this function with whatever the caller said."
            ),
            properties={
                "answer": {
                    "type": "string",
                    "description": "Exactly what the caller said",
                },
            },
            required=["answer"],
            handler=partial(
                handle_answer,
                questions=questions,
                collected=collected,
                index=index,
                transfer_rule=transfer_rule,
                call_sid=call_sid,
                firm=firm,
                case_types=case_types,
                q=q,
                transfer_in_progress=transfer_in_progress,
            ),
        ),
        FlowsFunctionSchema(
            name="transfer_to_human",
            description=(
                "Call this if the caller asks to speak to a person, sounds frustrated or upset, "
                "or explicitly requests a transfer at any point."
            ),
            properties={},
            required=[],
            handler=partial(
                handle_question_transfer,
                transfer_rule=transfer_rule,
                call_sid=call_sid,
                firm=firm,
                collected=collected,
                transfer_in_progress=transfer_in_progress,
            ),
        ),
    ]

    if case_types:
        case_list = "\n".join(f"- {ct['label']} (id: {ct['id']})" for ct in case_types)
        functions.append(
            FlowsFunctionSchema(
                name="change_case_type",
                description=(
                    "ONLY call this if the caller EXPLICITLY and CLEARLY says they want to switch to a "
                    "completely different legal matter than the one being discussed — for example "
                    "'actually I also need help with a traffic ticket' or 'forget the divorce, I have a "
                    "bankruptcy question'. "
                    "Do NOT call this while collecting normal answers (name, email, phone, yes/no). "
                    "Do NOT call this if you are unsure. When in doubt, call record_answer instead. "
                    f"Available case types:\n{case_list}"
                ),
                properties={
                    "case_type_id": {
                        "type": "string",
                        "description": "The id of the new case type",
                    },
                },
                required=["case_type_id"],
                handler=partial(
                    handle_pivot,
                    firm=firm,
                    collected=collected,
                    questions=questions,
                    index=index,
                    transfer_rule=transfer_rule,
                    call_sid=call_sid,
                    case_types=case_types,
                    transfer_in_progress=transfer_in_progress,
                ),
            )
        )

    return NodeConfig(
        name=f"q_{q['field_id']}",
        pre_actions=[{"type": "set_tool_choice", "choice": "auto"}],
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f'Ask the caller this exact question, word for word: "{q["question"]}"{options_hint} '
                    "After the caller responds — no matter what they say — call record_answer immediately. "
                    "Do NOT respond conversationally first. Do NOT say 'great' or 'I see' or ask follow-ups. "
                    "Do NOT offer advice or empathy. Call record_answer, then the next question will play."
                ),
            }
        ],
        functions=functions,
    )


async def handle_case_type(
    args: dict,
    fm: FlowManager,
    *,
    case_types: list,
    company_id: str,
    call_sid: str,
    firm: dict,
    transfer_in_progress: list,
) -> tuple:
    ctype_id = args.get("case_type_id", "")
    selected = next((ct for ct in case_types if ct["id"] == ctype_id), None)

    if not selected:
        logger.warning(f"Unknown case_type_id={ctype_id!r} — falling back to first")
        selected = case_types[0] if case_types else None

    if not selected:
        return ctype_id, make_save_node({"case_type_id": ctype_id})

    try:
        ct_data = await fetch_case_type_questions(company_id, ctype_id)
    except Exception as e:
        logger.error(f"Failed to fetch questions for {ctype_id}: {e}")
        return ctype_id, make_save_node({"case_type_id": ctype_id})

    questions = ct_data.get("questions", [])
    transfer_rule = _pick_transfer_rule(ct_data.get("transfer_rules", []))
    collected = {
        "case_type_id": ctype_id,
        "case_type_slug": ct_data.get("case_type_slug", ""),
    }

    if not questions:
        return ctype_id, make_save_node(collected)

    # Pre-fill answers the caller already gave in their opening description so the
    # bot doesn't re-ask them. make_question_node skips any field_id in collected.
    description = args.get("caller_description", "")
    if description:
        prefilled = await extract_answers_from_description(description, questions)
        collected.update(prefilled)

    return ctype_id, make_question_node(
        questions,
        collected,
        0,
        transfer_rule,
        call_sid,
        firm,
        case_types,
        transfer_in_progress=transfer_in_progress,
    )


async def handle_case_type_transfer(
    args: dict,
    fm: FlowManager,
    *,
    firm: dict,
    call_sid: str,
    transfer_in_progress: list,
) -> tuple:
    number = firm.get("transfer_number", "")
    name = firm.get("attorney_name", "the firm")

    from bot_twilio import _warm_transfer  # late import avoids circular dependency

    success = await _warm_transfer(
        call_sid, number, name, "legal matter", {}, transfer_in_progress
    )

    if success:
        return "transferred", make_save_node({})

    return "transfer_failed", NodeConfig(
        name="attorney_unavailable",
        pre_actions=[{"type": "set_tool_choice", "choice": "auto"}],
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Tell the caller the attorney is currently unavailable and you were unable "
                    "to connect them. Apologize briefly and end the call."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
        functions=[],
    )


def make_case_type_node(
    company_data: dict,
    call_sid: str = "",
    transfer_in_progress: list | None = None,
) -> NodeConfig:
    """Build the first node of the call flow. Greets the caller, listens for their
    situation, silently maps it to a case type, then routes to the question flow.
    """
    transfer_in_progress = transfer_in_progress or []
    firm = company_data["firm"]
    company_id = firm["id"]
    case_types = company_data["case_types"]
    case_list = "\n".join(f"- {ct['label']} (id: {ct['id']})" for ct in case_types)
    greeting = firm.get("greeting") or f"Thank you for calling {firm['name']}."

    return NodeConfig(
        name="select_case_type",
        # Don't run LLM inference on node entry — the tts_say greeting is the only
        # opening line. Otherwise the LLM generates a SECOND greeting ("Hello! How
        # can I assist you?") on top of the static one. Wait for the caller.
        respond_immediately=False,
        role_message=(
            f"You are a strict intake receptionist for {firm['name']}. "
            "Your ONLY job is to collect information by calling the available functions. "
            "You MUST call a function after every caller response — never reply in plain text instead. "
            "Do NOT offer advice, empathy, opinions, or open-ended follow-up questions. "
            "Do NOT say things like 'that sounds like a good step' or 'what specific concerns do you have'. "
            "When a caller gives you information, call the appropriate function immediately. "
            "Never engage in conversation — only collect and record."
        ),
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "The caller has already been greeted. Do NOT say hello or repeat the greeting. "
                    "Wait silently for the caller to describe their situation. "
                    "Listen to what they describe. DO NOT list or mention any case types out loud. "
                    "DO NOT respond conversationally or ask follow-up questions under any circumstances. "
                    "As soon as you can identify the case type from what the caller said — even partially — "
                    "call select_case_type immediately without saying anything first. "
                    "When you call it, pass caller_description containing the caller's FULL verbatim "
                    "statement — every name, location, date, and detail they mentioned. "
                    "Internally match their description to one of these case types:\n"
                    f"{case_list}"
                ),
            }
        ],
        pre_actions=[
            {"type": "tts_say", "text": greeting},
            {"type": "set_tool_choice", "choice": "required"},
        ],
        functions=[
            FlowsFunctionSchema(
                name="select_case_type",
                description="Call this once you've identified what type of case the caller needs help with.",
                properties={
                    "case_type_id": {
                        "type": "string",
                        "description": "The id of the matching case type",
                    },
                    "caller_description": {
                        "type": "string",
                        "description": (
                            "The caller's FULL, verbatim statement of everything they said "
                            "about their situation — every detail, name, location, fact they "
                            "volunteered. Copy it as completely as possible."
                        ),
                    },
                },
                required=["case_type_id", "caller_description"],
                handler=partial(
                    handle_case_type,
                    case_types=case_types,
                    company_id=company_id,
                    call_sid=call_sid,
                    firm=firm,
                    transfer_in_progress=transfer_in_progress,
                ),
            ),
            FlowsFunctionSchema(
                name="transfer_to_human",
                description=(
                    "Call this if the caller asks to speak to a person, sounds frustrated, "
                    "or requests a transfer before a case type is determined."
                ),
                properties={},
                required=[],
                handler=partial(
                    handle_case_type_transfer,
                    firm=firm,
                    call_sid=call_sid,
                    transfer_in_progress=transfer_in_progress,
                ),
            ),
        ],
    )


async def handle_attorney_accept(
    _args: dict,
    fm: FlowManager,
    *,
    caller_sid: str,
    accepted_event: asyncio.Event,
) -> tuple:
    logger.info(f"Attorney accepted transfer for caller {caller_sid}")
    accepted_event.set()
    return "accepted", None


async def handle_attorney_decline(
    _args: dict,
    fm: FlowManager,
    *,
    caller_sid: str,
    declined_event: asyncio.Event,
) -> tuple:
    logger.info(f"Attorney declined transfer for caller {caller_sid}")
    declined_event.set()
    return "declined", NodeConfig(
        name="attorney_decline_end",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Tell the attorney the call has been declined and the caller will be notified. "
                    "Be brief and polite."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
        functions=[],
    )


def _build_pipeline(transport, voice_id: str = DEFAULT_VOICE_ID):
    """Build the STT → LLM → TTS pipeline and return (worker, flow_manager)."""
    stt = CartesiaSTTService(api_key=os.getenv("CARTESIA_API_KEY", ""))
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model="gpt-4o-mini",
    )
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        settings=CartesiaTTSService.Settings(voice=voice_id),
    )
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    flow_manager = FlowManager(
        llm=llm,
        context_aggregator=aggregators,
        worker=worker,
        transport=transport,
    )

    async def _set_tool_choice(action: dict, _fm):
        choice = action.get("choice", "auto")

        await worker.queue_frame(
            LLMUpdateSettingsFrame(delta=LLMSettings(extra={"tool_choice": choice}))
        )

    flow_manager.register_action("set_tool_choice", _set_tool_choice)

    return worker, flow_manager


async def run_caller_bot(
    transport,
    call_sid: str = "",
    transfer_in_progress: list | None = None,
) -> None:
    transfer_in_progress = transfer_in_progress or []
    phone = os.getenv("AL_BACKEND_FALLBACK_PHONE", "")

    try:
        company_data = await fetch_company_case_types(phone)
        voice_id = company_data.get("firm", {}).get("voice_id", DEFAULT_VOICE_ID)
    except Exception as e:
        logger.error(f"Failed to fetch company details: {e}. Terminating.")
        os.kill(os.getpid(), signal.SIGTERM)
        return

    initial_node = make_case_type_node(
        company_data,
        call_sid=call_sid,
        transfer_in_progress=transfer_in_progress,
    )
    worker, flow_manager = _build_pipeline(transport, voice_id)

    @worker.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        await flow_manager.initialize(initial_node)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        if transfer_in_progress:
            logger.info("WebSocket closed during warm transfer — keeping worker alive")
            return
        logger.info("Caller disconnected — cancelling worker")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()
