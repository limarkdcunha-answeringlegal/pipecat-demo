import asyncio
import os
import signal
from functools import partial

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.frames.frames import Frame, LLMTextFrame, LLMUpdateSettingsFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.settings import LLMSettings
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService


from pipecat.workers.runner import WorkerRunner
from pipecat_flows import FlowManager, FlowsFunctionSchema, NodeConfig

from api_client import fetch_case_type_questions, fetch_company_case_types
from bot_config import (
    DEFAULT_VOICE_ID,
)
from helpers import _ack_for, extract_answers_from_description
from transfer_registry import TransferRegistry

load_dotenv(override=True)

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


# Max times the bot re-asks a question when it can't understand the answer
# before giving up and saving whatever it got (CLAUDE.md §5/§6 — configurable).
MAX_ANSWER_RETRIES = 2

# Core flow start


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

    firm = company_data.get("firm", {})
    worker, flow_manager, context = _build_pipeline(transport, voice_id)

    # "content": (
    #         f"You are a strict intake receptionist for {firm.get('name', 'this law firm')}. "
    #         "Your ONLY job is to collect information by calling the available functions. "
    #         "You MUST call a function after every caller response — never reply in plain text instead. "
    #         "Do NOT offer advice, empathy, opinions, or open-ended follow-up questions. "
    #         "Do NOT say things like 'that sounds like a good step' or 'what specific concerns do you have'. "
    #         "When a caller gives you information, call the appropriate function immediately. "
    #         "Never engage in conversation — only collect and record."
    #     ),

    context.add_message(
        {
            "role": "system",
            "content": (
                f"You are a strict intake receptionist for {firm.get('name', 'this law firm')}. "
                "Your ONLY job is to collect information."
                "You MUST never reply in plain text. "
                "Do NOT offer advice, opinions, or open-ended follow-up questions. You can only empathize in form of an acknowl"
                "Do NOT say things like 'that sounds like a good step'"
                "Never engage in open ended conversation , even if user is trying to do that steer back to asking questions, — only collect and record."
            ),
        }
    )

    # First node of the conversation
    initial_node = make_case_type_node(
        company_data,
        call_sid=call_sid,
        transfer_in_progress=transfer_in_progress,
    )

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
    greeting = (
        firm.get("greeting")
        or f"Thank you for calling {firm['name']}. How may we help you today?"
    )

    return NodeConfig(
        name="select_case_type",
        # LLM handles all caller responses naturally — small talk, clarifications, anything.
        # WARNING: Never set respond_immediately=False here. It breaks conversational turns
        # and causes the LLM to generate plain text instead of calling functions.
        respond_immediately=True,
        # Task messages reasoning line by line
        # 1. Bot also srats the with reeting in here but we already have a gretting via pre_actionsso need to prevnt bot from regrettng
        # 2. Just aiting instrcution so it doesnt start speaking imeeditaly
        # 3. It mentions the casetyoes after API call is made we dont wanna do that, we just wanna say ot user if wrong case type is expected
        # 4. Bot can somtimes go into free flow conversaiton but here we dont that
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
            # Reset tool_choice to "auto" (no forced tool) so the LLM can decide which
            # function to call based on what the caller says. Without this, a previous
            # node may have pinned tool_choice to a specific function, causing OpenAI to
            # reject subsequent requests or always call the wrong tool.
            {"type": "set_tool_choice", "choice": "auto"},
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
        ],
    )


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
    prefilled: dict | None = None,
    retry: int = 0,
) -> tuple:
    answer = args.get("answer", "")

    # transcripts. Re-ask the SAME question (same index) up to MAX_ANSWER_RETRIES
    # times before giving up and saving whatever we got.
    # TODO: work more in this, this will necessary in future
    # if _is_low_confidence(answer) and retry < MAX_ANSWER_RETRIES:
    #     logger.info(
    #         f"Low-confidence answer for Q[{q['field_id']}] = {answer!r} "
    #         f"(retry {retry + 1}/{MAX_ANSWER_RETRIES}) — re-asking"
    #     )
    #     return "reask", make_question_node(
    #         questions,
    #         collected,
    #         index,
    #         transfer_rule,
    #         call_sid,
    #         firm,
    #         case_types,
    #         transfer_in_progress=transfer_in_progress,
    #         prefilled=prefilled,
    #         retry=retry + 1,
    #     )

    # Save current field immediately
    collected[q["field_id"]] = answer
    logger.info(f"Q[{q['field_id']}] = {answer!r}")

    # Acknowledge the answer we just heard before asking the next question
    return answer, make_question_node(
        questions,
        collected,
        index + 1,
        transfer_rule,
        call_sid,
        firm,
        case_types,
        transfer_in_progress=transfer_in_progress,
        prefilled=prefilled,
        ack=_ack_for(index),
    )


async def handle_prefill_confirm(
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
    prefilled: dict,
    transfer_in_progress: list,
) -> tuple:
    """Resolve a pre-filled answer the caller was asked to confirm (CLAUDE.md §4).

    The caller either confirms the extracted value or corrects it. Either way the
    field gets committed to `collected` and the flow advances. The field is removed
    from `prefilled` so it is never confirmed twice.
    """
    field_id = q["field_id"]
    confirmed = bool(args.get("confirmed", False))
    corrected = str(args.get("corrected_answer", "")).strip()

    if confirmed:
        answer = prefilled.get(field_id, "")
        logger.info(f"Prefill confirmed Q[{field_id}] = {answer!r}")
    else:
        # Caller corrected it. Use the inline correction if they gave one;
        # otherwise fall back to the original extracted value rather than losing
        # the field entirely.
        answer = corrected or prefilled.get(field_id, "")
        logger.info(f"Prefill corrected Q[{field_id}] = {answer!r}")

    collected[field_id] = answer
    # Drop from prefilled so make_question_node doesn't re-confirm it.
    remaining_prefilled = {k: v for k, v in prefilled.items() if k != field_id}

    return answer, make_question_node(
        questions,
        collected,
        index + 1,
        transfer_rule,
        call_sid,
        firm,
        case_types,
        transfer_in_progress=transfer_in_progress,
        prefilled=remaining_prefilled,
        ack=_ack_for(index),
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


async def handle_side_note(
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
    transfer_in_progress: list,
    prefilled: dict | None = None,
    retry: int = 0,
) -> tuple:
    """Caller said something that is NOT an answer to the current question —
    small talk OR a second/different legal matter. We do NOT switch flows. We log
    the remark into collected['additional_concerns'] (so the attorney still sees it),
    then return the SAME question node so the current intake continues uninterrupted.
    """
    remark = args.get("remark", "").strip()
    if remark:
        collected.setdefault("additional_concerns", [])
        # Stored as a list; joined into the briefing later. Avoid dup-logging the
        # same remark if the model fires twice.
        if remark not in collected["additional_concerns"]:
            collected["additional_concerns"].append(remark)
        logger.info(f"Side note recorded (flow unchanged): {remark!r}")

    # Re-ask the SAME question (same index, same retry) — flow does not advance.
    return "side_note", make_question_node(
        questions,
        collected,
        index,
        transfer_rule,
        call_sid,
        firm,
        case_types,
        transfer_in_progress=transfer_in_progress,
        prefilled=prefilled,
        retry=retry,
    )


def make_confirm_node(
    questions: list[dict],
    collected: dict,
    index: int,
    transfer_rule: dict | None,
    call_sid: str,
    firm: dict,
    case_types: list,
    *,
    q: dict,
    prefilled: dict,
    transfer_in_progress: list,
) -> NodeConfig:
    """Build a confirmation node for a question the caller answered up-front in their
    opening statement (CLAUDE.md §4). Instead of asking the question cold, the bot
    plays back what it understood and asks the caller to confirm or correct it. The
    answer is only committed once confirmed via handle_prefill_confirm.
    """
    field_id = q["field_id"]
    prefilled_answer = prefilled.get(field_id, "")

    functions = [
        FlowsFunctionSchema(
            name="confirm_answer",
            description=(
                "ALWAYS call this the instant the caller responds to the confirmation. "
                "Set confirmed=true if they agree the answer is correct. "
                "Set confirmed=false if they say it is wrong, and put their corrected "
                "answer in corrected_answer (copy exactly what they now say). "
                "You MUST call this before saying anything else."
            ),
            properties={
                "confirmed": {
                    "type": "boolean",
                    "description": "true if the caller confirms the answer is correct, false if they correct it",
                },
                "corrected_answer": {
                    "type": "string",
                    "description": (
                        "If confirmed is false, the caller's corrected answer, exactly "
                        "as they said it. Empty string if they confirmed."
                    ),
                },
            },
            required=["confirmed"],
            handler=partial(
                handle_prefill_confirm,
                questions=questions,
                collected=collected,
                index=index,
                transfer_rule=transfer_rule,
                call_sid=call_sid,
                firm=firm,
                case_types=case_types,
                q=q,
                prefilled=prefilled,
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

    question_text = q.get("question", "")

    # — e.g. \"I have your first name as Lamont, "
    #                 "is that right?\" or \"You mentioned you don't have any children together — "
    #                 "did I get that right?\". Never confirm a bare value with no context "
    #                 "(do NOT say just \"You mentioned No — is that correct?\"). "

    return NodeConfig(
        name=f"confirm_{field_id}",
        respond_immediately=True,
        pre_actions=[{"type": "set_tool_choice", "choice": "auto"}],
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "The caller already volunteered an answer to the next intake question "
                    "in their opening statement. Confirm it with them in ONE natural, warm "
                    "sentence — phrase it freshly each time so it never sounds scripted.\n"
                    f'- The question this answers: "{question_text}"\n'
                    f'- What you understood the caller\'s answer to be: "{prefilled_answer}"\n'
                    "Restate the answer IN THE CONTEXT of the question so the caller knows "
                    "exactly what you're confirming."
                    "Do NOT call any function yet. Wait for them to respond. "
                    "Once they respond — whether they confirm or correct it — call confirm_answer. "
                    "Never reply in plain text after they respond."
                ),
            }
        ],
        functions=functions,
    )


def make_end_of_intake_node(
    collected: dict,
    transfer_rule: dict | None,
    call_sid: str,
    firm: dict,
    transfer_in_progress: list,
) -> NodeConfig:
    """Terminal node after all questions answered. Attempts warm transfer if a transfer
    rule exists, otherwise thanks caller and ends."""

    async def _do_transfer(_action: dict, fm: FlowManager) -> None:
        if not transfer_rule:
            number = firm.get("transfer_number", "")
            name = firm.get("attorney_name", "the firm")
        else:
            number = transfer_rule.get("contact_phone", "")
            name = transfer_rule.get("contact_name", "the team")

        if not number:
            logger.info("No transfer number configured — ending call without transfer")
            return

        case_summary = collected.get("case_type_slug", "legal matter")
        from bot_twilio import _warm_transfer

        # resume_on_fail=False: intake is done — if transfer fails, _warm_transfer
        # says goodbye and hangs up via Twilio. Resuming would restart the greeting.
        await _warm_transfer(
            call_sid,
            number,
            name,
            case_summary,
            collected,
            transfer_in_progress,
            resume_on_fail=False,
        )

    return NodeConfig(
        name="end_of_intake",
        respond_immediately=True,
        functions=[],
        pre_actions=[{"type": "set_tool_choice", "choice": "auto"}],
        post_actions=[{"type": "function", "handler": _do_transfer}],
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Thank the caller warmly — their information has been collected and you are "
                    "now connecting them with someone who can help. Keep it brief and natural."
                ),
            }
        ],
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
    prefilled: dict | None = None,
    retry: int = 0,
    ack: str = "",
) -> NodeConfig:
    """Build a node for the current question. Skips disabled/conditional-blocked questions.
    Returns make_save_node when all questions are exhausted.

    `retry` > 0 means the previous answer was unintelligible; the node prompts the
    caller to repeat instead of asking the question cold (CLAUDE.md §6).

    `prefilled` holds answers extracted from the caller's opening statement that have
    NOT yet been confirmed. When the current question was pre-filled, this builds a
    confirmation node instead of a normal question node (CLAUDE.md §4) — the answer is
    only committed to `collected` once the caller confirms or corrects it.
    """
    case_types = case_types or []
    prefilled = prefilled or {}

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
        # Already confirmed (committed to collected) — nothing to ask.
        if q["field_id"] in collected:
            index += 1
            continue
        break
    else:
        return make_end_of_intake_node(
            collected, transfer_rule, call_sid, firm, transfer_in_progress
        )

    q = questions[index]

    # Pre-filled but unconfirmed → ask a confirmation question instead of the
    # normal question. Guardrail: never silently accept an extracted answer.
    if q["field_id"] in prefilled:
        return make_confirm_node(
            questions,
            collected,
            index,
            transfer_rule,
            call_sid,
            firm,
            case_types,
            q=q,
            prefilled=prefilled,
            transfer_in_progress=transfer_in_progress,
        )

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
                prefilled=prefilled,
                retry=retry,
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
        FlowsFunctionSchema(
            name="note_side_concern",
            description=(
                "Call this when the caller says something that is NOT an answer to the current "
                "question — EITHER small talk (weather, traffic chit-chat) OR a DIFFERENT legal "
                "matter than the one being handled (e.g. mid-divorce they mention a traffic "
                "ticket or a bankruptcy question). "
                "We do NOT switch topics — we note their concern and keep collecting info for the "
                "CURRENT matter. "
                "Do NOT call this for relevant answers, hesitations, or emotional statements. "
                "When in doubt, call record_answer instead."
            ),
            properties={
                "remark": {
                    "type": "string",
                    "description": "What the caller said (the side concern or off-topic remark)",
                },
            },
            required=["remark"],
            handler=partial(
                handle_side_note,
                questions=questions,
                collected=collected,
                index=index,
                transfer_rule=transfer_rule,
                call_sid=call_sid,
                firm=firm,
                case_types=case_types,
                transfer_in_progress=transfer_in_progress,
                prefilled=prefilled,
                retry=retry,
            ),
        ),
    ]

    spoken_question = f"{q['question']}{options_hint}"
    if retry > 0:
        spoken_question = f"Sorry, I didn't quite catch that. {spoken_question}"

    ack_prefix = f"{ack} " if ack and retry == 0 else ""

    # Listen node: respond_immediately=False, has all functions.
    # LLM only gets invoked here after the caller speaks — no cascade possible.
    listen_node = NodeConfig(
        name=f"q_{q['field_id']}",
        respond_immediately=False,
        pre_actions=[{"type": "set_tool_choice", "choice": "auto"}],
        # DO NOT modify this prompt for now
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "The caller just responded. "
                    "If they answered (even partially), call record_answer with what they said. "
                    "If they brought up something that is NOT an answer — small talk OR a different "
                    "legal matter — call note_side_concern. "
                    "Never reply in plain text."
                ),
            }
        ],
        functions=functions,
    )

    async def _transition_to_listen(_action: dict, fm: FlowManager) -> None:
        await fm.set_node_from_config(listen_node)

    # Ask node: respond_immediately=True, NO functions — LLM can only speak.
    # It rephrases the raw API question naturally, then post_action transitions
    # to the listen node. With no functions available, cascade is structurally impossible.
    return NodeConfig(
        name=f"q_ask_{q['field_id']}",
        respond_immediately=True,
        functions=[],
        pre_actions=[{"type": "set_tool_choice", "choice": "auto"}],
        post_actions=[{"type": "function", "handler": _transition_to_listen}],
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"[SYSTEM INSTRUCTION — DO NOT SPEAK THIS TEXT] "
                    f"{ack_prefix}Your task: rephrase and ask the question below naturally. "
                    "Strip any garbled/nonsensical parts, understand the intent, ask it warmly. "
                    "Speak ONLY the rephrased question — nothing else. Do not read these instructions aloud.\n\n"
                    f"Question to rephrase: {spoken_question}"
                ),
            }
        ],
    )


async def _fetch_and_build_question_node(
    *,
    ctype_id: str,
    description: str,
    company_id: str,
    call_sid: str,
    firm: dict,
    case_types: list,
    transfer_in_progress: list,
) -> NodeConfig:
    """Fetch questions + extract prefills. Returns ready question node."""
    try:
        ct_data = await fetch_case_type_questions(company_id, ctype_id)
    except Exception as e:
        logger.error(f"Failed to fetch questions for {ctype_id}: {e}")
        return make_save_node({"case_type_id": ctype_id})

    questions = ct_data.get("questions", [])
    transfer_rule = _pick_transfer_rule(ct_data.get("transfer_rules", []))
    collected = {
        "case_type_id": ctype_id,
        "case_type_slug": ct_data.get("case_type_slug", ""),
    }

    if not questions:
        return make_save_node(collected)

    prefilled: dict = {}
    if description:
        prefilled = await extract_answers_from_description(description, questions)

    return make_question_node(
        questions,
        collected,
        0,
        transfer_rule,
        call_sid,
        firm,
        case_types,
        transfer_in_progress=transfer_in_progress,
        prefilled=prefilled,
    )


def make_confirm_case_type_node(
    *,
    ctype_id: str,
    ctype_label: str,
    description: str,
    case_types: list,
    company_id: str,
    call_sid: str,
    firm: dict,
    transfer_in_progress: list,
) -> NodeConfig:
    """Ask caller to confirm the detected case type before fetching questions."""

    async def handle_confirmed(_args: dict, fm: FlowManager) -> tuple:
        node = await _fetch_and_build_question_node(
            ctype_id=ctype_id,
            description=description,
            company_id=company_id,
            call_sid=call_sid,
            firm=firm,
            case_types=case_types,
            transfer_in_progress=transfer_in_progress,
        )
        return "confirmed", node

    async def handle_corrected(_args: dict, fm: FlowManager) -> tuple:
        node = make_case_type_node(
            {"firm": firm, "case_types": case_types},
            call_sid=call_sid,
            transfer_in_progress=transfer_in_progress,
        )
        return "corrected", node

    confirm_text = (
        f"Just to confirm — you're calling about a {ctype_label} matter, is that right?"
    )

    return NodeConfig(
        name="confirm_case_type",
        respond_immediately=False,
        pre_actions=[
            {"type": "tts_say", "text": confirm_text},
            {"type": "set_tool_choice", "choice": "auto"},
        ],
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "STAGE CHANGE: case type is already detected. Your ONLY job now is to interpret "
                    "the caller's yes/no answer to the confirmation question you just asked. "
                    "You have exactly TWO valid actions and you MUST pick one of them:\n"
                    "- confirm_case_type — for ANY affirmative: yes, yeah, yep, correct, right, "
                    "that's right, sure, uh-huh, mhm, exactly, or similar.\n"
                    "- correct_case_type — ONLY if the caller explicitly says no, wrong, or names a different matter.\n"
                    "Do NOT call select_case_type — that step is finished; ignore any earlier instruction to call it. "
                    "When in doubt, call confirm_case_type. Never reply in plain text."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="confirm_case_type",
                description=(
                    "Call this when the caller affirms the case type — includes yes, yeah, yep, "
                    "correct, right, sure, uh-huh, mhm, exactly, that's right, or any similar affirmation."
                ),
                properties={},
                required=[],
                handler=handle_confirmed,
            ),
            FlowsFunctionSchema(
                name="correct_case_type",
                description="Call this ONLY when the caller explicitly says no, that's wrong, or names a different legal matter.",
                properties={},
                required=[],
                handler=handle_corrected,
            ),
        ],
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

    description = args.get("caller_description", "")
    return ctype_id, make_confirm_case_type_node(
        ctype_id=ctype_id,
        ctype_label=selected["label"],
        description=description,
        case_types=case_types,
        company_id=company_id,
        call_sid=call_sid,
        firm=firm,
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


class ToolLeakGuard(FrameProcessor):
    """Drop LLM text that is actually a malformed tool-call leak before it reaches TTS.

    When tool_choice is 'auto', the model occasionally emits a function call as PLAIN
    TEXT instead of a real tool call — e.g. `<select_case_type uuid="...">`. That text
    flows LLM → TTS and Cartesia 400s on the angle brackets ("invalid input"), killing
    the turn. This processor sits between the LLM and TTS and suppresses any text frame
    that looks like such a leak, so a stray tool-call string is silently dropped
    instead of being spoken or crashing TTS.
    """

    def _looks_like_tool_leak(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        lowered = t.lower()
        # Pipecat async tool JSON leaked as spoken text
        if '"type":"async_tool"' in t or '"type": "async_tool"' in t:
            return True
        if t.startswith("{") and ("async_tool" in t or "tool_call_id" in t):
            return True
        # Ask node task_message echoed verbatim by LLM on interruption
        if (
            "raw question:" in lowered
            or "ask the caller the following question" in lowered
        ):
            return True
        if "ignore any garbled or nonsensical" in lowered:
            return True
        # XML/HTML-ish tag wrapping a known tool name
        if t.startswith("<") and ("uuid=" in lowered or "/>" in t or "</" in t):
            return True
        for tool in (
            "select_case_type",
            "record_answer",
            "note_side_concern",
            "confirm_answer",
            "transfer_to_human",
        ):
            if f"<{tool}" in lowered or f"{tool}(" in lowered:
                return True
        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMTextFrame, TextFrame)) and self._looks_like_tool_leak(
            getattr(frame, "text", "")
        ):
            logger.warning(f"Suppressed tool-call leak before TTS: {frame.text!r}")
            return  # swallow the frame — do not forward to TTS
        await self.push_frame(frame, direction)


def _build_pipeline(transport, voice_id: str = DEFAULT_VOICE_ID):
    """Build the STT → LLM → TTS pipeline and return (worker, flow_manager)."""
    # Deepgram Nova-3, not Cartesia ink-whisper. ink-whisper (a Whisper derivative)
    # mistranscribed clean speech ("discuss a divorce case" -> "be doing skills") and
    # hallucinated filler on silence. Nova-3 is telephony/conversational-tuned and far
    # more accurate on voice-intake audio. smart_format/punctuate clean up the output.
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY", ""),
        settings=DeepgramSTTService.Settings(
            model="nova-3",
            language="en-US",
            smart_format=True,
            punctuate=True,
        ),
    )
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
            # Tuned VAD: raise confidence + min_volume so room noise / line hiss
            # doesn't register as speech (a major source of STT hallucination),
            # and lengthen stop_secs so callers aren't cut off mid-sentence —
            # truncated utterances are what feed ink-whisper garbage.
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7,
                    start_secs=0.2,
                    # 1.2s of silence before ending a turn. Callers pause mid-thought
                    # ("Hi, I wanted to...") — a short stop_secs splits one sentence
                    # into fragments, each becoming a separate turn. Longer = fewer
                    # truncated utterances and fewer spurious turns.
                    stop_secs=1.2,
                    min_volume=0.6,
                )
            ),
        ),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            ToolLeakGuard(),
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
        # 'auto' = "no forced tool". Setting tool_choice='auto' on a node with
        # functions=[] still 400s OpenAI ("tool_choice only allowed when tools are
        # specified"), so for 'auto' POP the key entirely rather than set it.
        if choice == "auto":
            llm._settings.extra.pop("tool_choice", None)
        else:
            await worker.queue_frame(
                LLMUpdateSettingsFrame(delta=LLMSettings(extra={"tool_choice": choice}))
            )

    flow_manager.register_action("set_tool_choice", _set_tool_choice)

    return worker, flow_manager, context
