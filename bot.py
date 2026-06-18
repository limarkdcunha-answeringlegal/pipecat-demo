import asyncio
import os
import signal
import xml.sax.saxutils as xml_utils
from functools import partial

import aiohttp
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner
from pipecat_flows import FlowManager, FlowsFunctionSchema, NodeConfig

from api_client import fetch_case_type_questions, fetch_company_details

load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_VOICE_ID = "32b3f3c5-7171-46aa-abe7-b598964aa793"
HOLD_MUSIC_URL = "https://demo.twilio.com/docs/classic.mp3"  # reliable Twilio-hosted URL
ATTORNEY_BOT_WEBSOCKET_PATH = "/twilio/attorney-ws"

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")


# ─────────────────────────────────────────────────────────────────────────────
# TransferRegistry
# Coordinates state between the caller bot and the attorney bot WebSocket,
# which run as completely separate async contexts. The registry is the shared
# lookup bridge between them, keyed by caller call_sid.
# ─────────────────────────────────────────────────────────────────────────────

class TransferRegistry:
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


# ─────────────────────────────────────────────────────────────────────────────
# In-memory transfer context store
# Bot 1 (caller leg) stores full collected data here before transferring.
# Bot 2 (attorney leg) reads it to brief the attorney with everything collected.
# Swap store_transfer_context / fetch_transfer_context for Redis calls when
# scaling to multiple servers.
# ─────────────────────────────────────────────────────────────────────────────

_transfer_contexts: dict[str, dict] = {}


def store_transfer_context(call_sid: str, context: dict) -> None:
    """Bot 1 stores full collected intake data before initiating transfer."""
    _transfer_contexts[call_sid] = context
    logger.info(f"Stored transfer context for {call_sid}: {list(context.keys())}")


def fetch_transfer_context(call_sid: str) -> dict:
    """Bot 2 reads full context to brief the attorney. Pops to auto-clean."""
    return _transfer_contexts.pop(call_sid, {})


# ─────────────────────────────────────────────────────────────────────────────
# Twilio helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _twilio_post(path: str, data: dict) -> dict:
    """Authenticated POST to the Twilio REST API. Raises RuntimeError on 4xx/5xx."""
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}{path}"
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            data=data,
            auth=aiohttp.BasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
        ) as resp:
            result = await resp.json()
            if resp.status >= 400:
                logger.error(f"Twilio error {resp.status} on {path}: {result}")
                raise RuntimeError(
                    f"Twilio POST failed: {resp.status} — {result.get('message', '')}"
                )
            return result


async def _hold_caller(call_sid: str) -> None:
    """Redirect caller to hold music loop. No-op in browser/WebRTC mode."""
    if not call_sid:
        logger.info("[BROWSER] Skipping hold — no Twilio call_sid")
        return

    twiml = f"<Response><Play loop='0'>{HOLD_MUSIC_URL}</Play></Response>"
    try:
        await _twilio_post(f"/Calls/{call_sid}.json", {"Twiml": twiml})
        logger.info(f"Caller {call_sid} placed on hold")
    except RuntimeError as e:
        logger.error(f"Failed to hold caller {call_sid}: {e}")
        raise


async def _resume_caller(call_sid: str) -> None:
    """Take caller off hold and reconnect them to the bot WebSocket pipeline.
    Called when attorney declines or transfer fails — without this the caller
    would be stuck on hold music indefinitely.
    """
    if not call_sid:
        return

    ws_url = (
        PUBLIC_BASE_URL
        .replace("https://", "wss://")
        .replace("http://", "ws://")
        + "/twilio/ws"
    )
    twiml = (
        f"<Response><Connect>"
        f"<Stream url='{ws_url}'/>"
        f"</Connect></Response>"
    )
    try:
        await _twilio_post(f"/Calls/{call_sid}.json", {"Twiml": twiml})
        logger.info(f"Caller {call_sid} resumed on bot pipeline")
    except RuntimeError as e:
        logger.error(f"Failed to resume caller {call_sid}: {e}")
        raise


async def _bridge_calls(caller_sid: str, attorney_sid: str) -> None:
    """Put caller and attorney into a Twilio conference room so they speak directly.
    endConferenceOnExit='true' only on caller leg — attorney dropping off first
    won't kill the room.
    """
    if not caller_sid or attorney_sid == "MOCK_ATTORNEY_SID":
        logger.info(f"[MOCK] Would bridge caller {caller_sid or '(browser)'} ↔ attorney")
        if caller_sid:
            await _twilio_post(f"/Calls/{caller_sid}.json", {
                "Twiml": (
                    "<Response><Say>Mock bridge: attorney accepted. "
                    "Ending demo here.</Say><Hangup/></Response>"
                ),
            })
        return

    conference = f"bridge_{caller_sid}"

    caller_twiml = (
        f"<Response><Dial>"
        f"<Conference beep='false' startConferenceOnEnter='true' "
        f"endConferenceOnExit='true'>{conference}</Conference>"
        f"</Dial></Response>"
    )
    attorney_twiml = (
        f"<Response><Dial>"
        f"<Conference beep='false' startConferenceOnEnter='true' "
        f"endConferenceOnExit='false'>{conference}</Conference>"
        f"</Dial></Response>"
    )

    try:
        await _twilio_post(f"/Calls/{caller_sid}.json", {"Twiml": caller_twiml})
        await _twilio_post(f"/Calls/{attorney_sid}.json", {"Twiml": attorney_twiml})
        logger.info(f"Bridged caller {caller_sid} ↔ attorney {attorney_sid}")
    except RuntimeError as e:
        logger.error(f"Failed to bridge calls: {e}")
        raise


async def _dial_attorney(
    attorney_number: str,
    caller_sid: str,
    case_summary: str,
    contact_name: str,
    accepted_event: asyncio.Event,
    declined_event: asyncio.Event,
    attorney_sid_holder: list,
) -> None:
    """Dial attorney, connect them to the attorney bot pipeline, wait for accept/decline."""

    # ── MOCK MODE — set MOCK_ATTORNEY=true in .env to test without real calls ──
    # if os.getenv("MOCK_ATTORNEY", "false").lower() == "true":
    #     outcome = os.getenv("MOCK_ATTORNEY_OUTCOME", "accept")
    #     delay = float(os.getenv("MOCK_ATTORNEY_DELAY", "3"))
    #     logger.info(f"[MOCK] Simulating attorney {outcome} in {delay}s")
    #     attorney_sid_holder.append("MOCK_ATTORNEY_SID")
    #     await asyncio.sleep(delay)
    #     if outcome == "accept":
    #         accepted_event.set()
    #     else:
    #         declined_event.set()
    #     return
    # ──────────────────────────────────────────────────────────────────────────

    ws_url = (
        PUBLIC_BASE_URL
        .replace("https://", "wss://")
        .replace("http://", "ws://")
        + ATTORNEY_BOT_WEBSOCKET_PATH
    )

    safe_summary = xml_utils.escape(case_summary, entities={"'": "&apos;"})
    safe_name = xml_utils.escape(contact_name, entities={"'": "&apos;"})
    twiml = (
        f"<Response><Connect>"
        f"<Stream url='{ws_url}'>"
        f"<Parameter name='case_summary' value='{safe_summary}'/>"
        f"<Parameter name='caller_sid' value='{caller_sid}'/>"
        f"<Parameter name='contact_name' value='{safe_name}'/>"
        f"</Stream></Connect></Response>"
    )

    try:
        resp = await _twilio_post("/Calls.json", {
            "To": attorney_number,
            "From": TWILIO_PHONE_NUMBER,
            "Twiml": twiml,
        })
    except RuntimeError as e:
        logger.error(f"Failed to dial attorney: {e}")
        declined_event.set()
        return

    attorney_sid = resp.get("sid", "")
    if not attorney_sid:
        logger.error("Twilio returned no SID for attorney call")
        declined_event.set()
        return

    attorney_sid_holder.append(attorney_sid)
    logger.info(f"Dialed attorney {attorney_number} ({contact_name}), call_sid={attorney_sid}")

    t_accept = asyncio.create_task(accepted_event.wait())
    t_decline = asyncio.create_task(declined_event.wait())
    try:
        await asyncio.wait_for(
            asyncio.wait({t_accept, t_decline}, return_when=asyncio.FIRST_COMPLETED),
            timeout=45.0,
        )
    except asyncio.TimeoutError:
        logger.info("Attorney did not respond in 45s — treating as declined")
        declined_event.set()
    finally:
        t_accept.cancel()
        t_decline.cancel()


async def _warm_transfer(
    call_sid: str,
    transfer_number: str,
    contact_name: str,
    case_summary: str,
    collected: dict,
    transfer_in_progress: list,
) -> bool:
    """Full warm transfer: store context → hold caller → dial attorney bot → bridge or resume.

    transfer_in_progress is a list used as a mutable bool flag — non-empty means
    transfer is active. The on_client_disconnected handler checks this to avoid
    cancelling the worker when Twilio closes the WebSocket during hold music.

    Returns True if attorney accepted and bridge succeeded.
    Returns False if attorney declined, timed out, or any step failed.
    Caller is always either bridged or resumed — never left on hold.
    """
    store_transfer_context(call_sid, collected)

    state = transfer_registry.register(call_sid, case_summary)
    accepted_event: asyncio.Event = state["accepted"]
    declined_event: asyncio.Event = state["declined"]
    attorney_sid_holder: list[str] = state["attorney_sid"]

    transfer_in_progress.append(True)  # signal to on_client_disconnected
    try:
        try:
            await _hold_caller(call_sid)
        except RuntimeError as e:
            logger.error(f"Failed to hold caller {call_sid}: {e}")
            return False

        await _dial_attorney(
            transfer_number,
            caller_sid=call_sid,
            case_summary=case_summary,
            contact_name=contact_name,
            accepted_event=accepted_event,
            declined_event=declined_event,
            attorney_sid_holder=attorney_sid_holder,
        )

        if accepted_event.is_set():
            attorney_sid = attorney_sid_holder[0] if attorney_sid_holder else ""
            try:
                await _bridge_calls(call_sid, attorney_sid)
                return True
            except RuntimeError as e:
                logger.error(f"Bridge failed after accept: {e}")
                await _resume_caller(call_sid)
                return False
        else:
            logger.info(f"Transfer declined/timed out for {call_sid} — resuming caller")
            await _resume_caller(call_sid)
            return False

    finally:
        transfer_in_progress.clear()  # allow normal disconnect handling again
        transfer_registry.remove(call_sid)


# ─────────────────────────────────────────────────────────────────────────────
# Attorney briefing helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_attorney_briefing(
    context: dict,
    case_summary: str,
    contact_name: str,
) -> str:
    """Build the spoken briefing Bot 2 delivers when the attorney picks up.
    Uses full collected context from Bot 1 if available, falls back to case_summary.
    """
    caller_name = context.get("caller_name", "")
    case_type = context.get("case_type_slug", case_summary)
    answers = {
        k: v for k, v in context.items()
        if k not in ("case_type_id", "case_type_slug", "caller_name", "contact_name")
    }

    name_part = f"from {caller_name} " if caller_name else ""
    lines = [f"You have an incoming call {name_part}regarding a {case_type} case."]

    if answers:
        lines.append("Here is what was collected:")
        for field, value in answers.items():
            label = field.replace("_", " ").capitalize()
            lines.append(f"{label}: {value}.")

    lines.append("Would you like to accept this call?")
    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Flow node helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pick_transfer_rule(transfer_rules: list[dict]) -> dict | None:
    """Pick highest-priority patch-enabled transfer rule."""
    enabled = [r for r in transfer_rules if r.get("patch_enabled")]
    if not enabled:
        return None
    return sorted(enabled, key=lambda r: r.get("priority", 99))[0]


def make_save_node(collected: dict) -> NodeConfig:
    """Terminal node — thanks caller and ends the conversation."""
    return NodeConfig(
        name="save_and_end",
        task_messages=[{
            "role": "developer",
            "content": (
                "Thank the caller warmly, let them know their information has been received "
                "and someone will be in touch shortly. Then end the call."
            ),
        }],
        post_actions=[{"type": "end_conversation"}],
        functions=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Question node handlers
# ─────────────────────────────────────────────────────────────────────────────

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
    collected[q["field_id"]] = answer
    logger.info(f"Q[{q['field_id']}] = {answer!r}")
    return answer, make_question_node(
        questions, collected, index + 1, transfer_rule, call_sid, firm, case_types,
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
    success = await _warm_transfer(
        call_sid, number, name, case_summary, collected, transfer_in_progress
    )

    if success:
        return "transferred", make_save_node(collected)

    return "transfer_failed", NodeConfig(
        name="attorney_unavailable",
        task_messages=[{
            "role": "developer",
            "content": (
                "Tell the caller the attorney is currently unavailable and you were unable "
                "to connect them. Apologize briefly and end the call."
            ),
        }],
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
            questions, collected, index, transfer_rule, call_sid, firm, case_types,
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
        new_questions, new_collected, 0, new_transfer_rule, call_sid, firm, case_types,
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
                "Call this immediately after the caller says ANYTHING in response to the question — "
                "even one word, even if they refuse or go off-topic. "
                "Never respond conversationally before calling this."
            ),
            properties={
                "answer": {"type": "string", "description": "Exactly what the caller said"},
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
        functions.append(FlowsFunctionSchema(
            name="change_case_type",
            description=(
                "Call this if the caller mentions they want to discuss a different or additional "
                f"legal matter. Available case types:\n{case_list}"
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
        ))

    return NodeConfig(
        name=f"q_{q['field_id']}",
        task_messages=[{
            "role": "developer",
            "content": (
                f'Ask the caller this exact question, word for word: "{q["question"]}"{options_hint}'
            ),
        }],
        functions=functions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Case type node handlers
# ─────────────────────────────────────────────────────────────────────────────

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

    return ctype_id, make_question_node(
        questions, collected, 0, transfer_rule, call_sid, firm, case_types,
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

    success = await _warm_transfer(
        call_sid, number, name, "legal matter", {}, transfer_in_progress
    )

    if success:
        return "transferred", make_save_node({})

    return "transfer_failed", NodeConfig(
        name="attorney_unavailable",
        task_messages=[{
            "role": "developer",
            "content": (
                "Tell the caller the attorney is currently unavailable and you were unable "
                "to connect them. Apologize briefly and end the call."
            ),
        }],
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
        role_message=(
            f"You are a professional receptionist for {firm['name']}. "
            "Be warm, clear, and concise. Never skip a question."
        ),
        task_messages=[{
            "role": "developer",
            "content": (
                "The caller has already been greeted. Do NOT say hello or repeat the greeting. "
                "Wait silently for the caller to describe their situation. "
                "Listen to what they describe. DO NOT list or mention any case types out loud. "
                "Internally match their description to one of these case types and call select_case_type:\n"
                f"{case_list}"
            ),
        }],
        pre_actions=[{"type": "tts_say", "text": greeting}],
        functions=[
            FlowsFunctionSchema(
                name="select_case_type",
                description="Call this once you've identified what type of case the caller needs help with.",
                properties={
                    "case_type_id": {
                        "type": "string",
                        "description": "The id of the matching case type",
                    },
                },
                required=["case_type_id"],
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


# ─────────────────────────────────────────────────────────────────────────────
# Attorney bot pipeline handlers
# ─────────────────────────────────────────────────────────────────────────────

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
        task_messages=[{
            "role": "developer",
            "content": (
                "Tell the attorney the call has been declined and the caller will be notified. "
                "Be brief and polite."
            ),
        }],
        post_actions=[{"type": "end_conversation"}],
        functions=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder (shared by all bot variants)
# ─────────────────────────────────────────────────────────────────────────────

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
            filter_incomplete_user_turns=True,
        ),
    )
    pipeline = Pipeline([
        transport.input(),
        stt,
        aggregators.user(),
        llm,
        tts,
        transport.output(),
        aggregators.assistant(),
    ])
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
    return worker, flow_manager


# ─────────────────────────────────────────────────────────────────────────────
# Attorney bot — runs on the attorney's outbound call leg
# ─────────────────────────────────────────────────────────────────────────────

async def run_attorney_bot(
    websocket: WebSocket,
    stream_sid: str,
    attorney_call_sid: str,
    case_summary: str,
    caller_sid: str,
    contact_name: str = "",
    voice_id: str = DEFAULT_VOICE_ID,
) -> None:
    """Bot that runs on the attorney's call leg.
    Fetches full caller context, briefs the attorney, waits for accept/decline,
    then signals the caller bot via TransferRegistry events.
    """
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=attorney_call_sid,
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    context = fetch_transfer_context(caller_sid)
    state = transfer_registry.get(caller_sid) or {}
    accepted_event: asyncio.Event = state.get("accepted", asyncio.Event())
    declined_event: asyncio.Event = state.get("declined", asyncio.Event())

    briefing = _build_attorney_briefing(context, case_summary, contact_name)

    attorney_node = NodeConfig(
        name="attorney_intro",
        task_messages=[{
            "role": "developer",
            "content": (
                f'Say exactly: "{briefing}" '
                "Then wait for the attorney to respond. "
                "If they say yes or agree to take the call, call accept_call. "
                "If they say no, decline, or are unavailable, call decline_call."
            ),
        }],
        functions=[
            FlowsFunctionSchema(
                name="accept_call",
                description="Call this when the attorney agrees to take the call.",
                properties={},
                required=[],
                handler=partial(
                    handle_attorney_accept,
                    caller_sid=caller_sid,
                    accepted_event=accepted_event,
                ),
            ),
            FlowsFunctionSchema(
                name="decline_call",
                description="Call this when the attorney declines or is unavailable.",
                properties={},
                required=[],
                handler=partial(
                    handle_attorney_decline,
                    caller_sid=caller_sid,
                    declined_event=declined_event,
                ),
            ),
        ],
    )

    worker, flow_manager = _build_pipeline(transport, voice_id)

    @worker.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        await flow_manager.initialize(attorney_node)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await worker.cancel()
        if not accepted_event.is_set() and not declined_event.is_set():
            logger.info("Attorney disconnected without responding — treating as declined")
            declined_event.set()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


# ─────────────────────────────────────────────────────────────────────────────
# Caller bot — browser/WebRTC variant
# ─────────────────────────────────────────────────────────────────────────────

async def run_bot(connection: SmallWebRTCConnection) -> None:
    transport = SmallWebRTCTransport(
        connection,
        TransportParams(audio_out_enabled=True, audio_in_enabled=True),
    )
    phone = os.getenv("AL_BACKEND_FALLBACK_PHONE", "")

    try:
        company_data = await fetch_company_details(phone)
        voice_id = company_data.get("firm", {}).get("voice_id", DEFAULT_VOICE_ID)
        initial_node = make_case_type_node(company_data, call_sid="")
    except Exception as e:
        logger.error(f"Failed to fetch company details: {e}. Terminating.")
        os.kill(os.getpid(), signal.SIGTERM)
        return

    worker, flow_manager = _build_pipeline(transport, voice_id)

    @worker.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        await flow_manager.initialize(initial_node)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


# ─────────────────────────────────────────────────────────────────────────────
# Caller bot — Twilio telephony variant
# ─────────────────────────────────────────────────────────────────────────────

async def run_bot_twilio(
    websocket: WebSocket,
    stream_sid: str,
    call_sid: str,
    caller_phone: str,
) -> None:
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=TWILIO_ACCOUNT_SID,
        auth_token=TWILIO_AUTH_TOKEN,
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    # Always use fallback phone — caller_phone is the caller's number,
    # not the company lookup key
    phone = os.getenv("AL_BACKEND_FALLBACK_PHONE", "")
    logger.info(f"Fetching company details for phone: {phone!r}")

    try:
        company_data = await fetch_company_details(phone)
        voice_id = company_data.get("firm", {}).get("voice_id", DEFAULT_VOICE_ID)
    except Exception as e:
        logger.error(f"Failed to fetch company details: {e}. Terminating.")
        os.kill(os.getpid(), signal.SIGTERM)
        return

    # transfer_in_progress is a mutable list used as a flag.
    # Non-empty = transfer is active, on_client_disconnected should NOT cancel the worker.
    # This is necessary because Twilio closes the WebSocket when the caller is redirected
    # to hold music — without this flag that disconnect would kill the entire pipeline.
    transfer_in_progress: list = []

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