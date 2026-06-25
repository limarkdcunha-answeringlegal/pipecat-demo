import asyncio
from functools import partial

import aiohttp
from fastapi import WebSocket
from loguru import logger
import xml.sax.saxutils as xml_utils

from pipecat_flows import FlowsFunctionSchema, NodeConfig
from bot_core import (
    ATTORNEY_BOT_WEBSOCKET_PATH,
    DEFAULT_VOICE_ID,
    HOLD_MUSIC_URL,
    PUBLIC_BASE_URL,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER,
    transfer_registry,
    _build_pipeline,
    fetch_transfer_context,
    handle_attorney_accept,
    handle_attorney_decline,
    run_caller_bot,
    store_transfer_context,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner


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
        PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
        + "/twilio/ws"
    )
    twiml = f"<Response><Connect><Stream url='{ws_url}'/></Connect></Response>"
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
        logger.info(
            f"[MOCK] Would bridge caller {caller_sid or '(browser)'} ↔ attorney"
        )
        if caller_sid:
            await _twilio_post(
                f"/Calls/{caller_sid}.json",
                {
                    "Twiml": (
                        "<Response><Say>Mock bridge: attorney accepted. "
                        "Ending demo here.</Say><Hangup/></Response>"
                    ),
                },
            )
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


def _build_attorney_briefing(
    context: dict,
    case_summary: str,
) -> str:
    """Build the spoken briefing Bot 2 delivers when the attorney picks up.
    Uses full collected context from Bot 1 if available, falls back to case_summary.
    """
    caller_name = context.get("caller_name", "")
    case_type = context.get("case_type_slug", case_summary)
    answers = {
        k: v
        for k, v in context.items()
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
    ws_url = (
        PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
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
        resp = await _twilio_post(
            "/Calls.json",
            {
                "To": attorney_number,
                "From": TWILIO_PHONE_NUMBER,
                "Twiml": twiml,
            },
        )
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
    logger.info(
        f"Dialed attorney {attorney_number} ({contact_name}), call_sid={attorney_sid}"
    )

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
        transfer_in_progress.clear()
        transfer_registry.remove(call_sid)


async def run_attorney_side_bot(
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
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f'Say exactly: "{briefing}" '
                    "Then wait for the attorney to respond. "
                    "If they say yes or agree to take the call, call accept_call. "
                    "If they say no, decline, or are unavailable, call decline_call."
                ),
            }
        ],
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
            logger.info(
                "Attorney disconnected without responding — treating as declined"
            )
            declined_event.set()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


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
    transfer_in_progress: list = []
    await run_caller_bot(
        transport, call_sid=call_sid, transfer_in_progress=transfer_in_progress
    )
