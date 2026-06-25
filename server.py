import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat_ai_prebuilt.frontend import PipecatPrebuiltUI

from bot_webrtc import run_bot_webrtc as run_bot
from bot_twilio import run_bot_twilio, run_attorney_side_bot as run_attorney_bot

load_dotenv(override=True)

logger.add("bot.log", rotation="10 MB", retention="7 days", level="DEBUG", enqueue=True)

pipecat_handler = SmallWebRTCRequestHandler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await pipecat_handler.close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/debug", PipecatPrebuiltUI, name="prebuilt")


@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("static/index.html").read_text()


@app.post("/start")
async def start(request: Request):
    host = request.headers.get("host", "localhost:7860")
    scheme = "https" if request.url.scheme == "https" else "http"
    base = f"{scheme}://{host}"
    return {"webrtcRequestParams": {"endpoint": f"{base}/offer"}}


async def _handle_offer(data: dict):
    request = SmallWebRTCRequest.from_dict(data)

    async def on_connection(connection):
        asyncio.create_task(run_bot(connection))

    return await pipecat_handler.handle_web_request(request, on_connection)


@app.post("/offer")
async def offer_post(data: dict):
    return await _handle_offer(data)


@app.patch("/offer")
async def offer_patch(data: dict):
    return await _handle_offer(data)


@app.post("/ice")
async def ice_candidates(data: dict):
    pc_id = data.get("pc_id")
    if not pc_id:
        raise HTTPException(status_code=400, detail="pc_id required")

    candidates = [
        IceCandidate(
            candidate=c["candidate"],
            sdp_mid=c["sdpMid"],
            sdp_mline_index=c["sdpMLineIndex"],
        )
        for c in data.get("candidates", [])
    ]
    await pipecat_handler.handle_patch_request(
        SmallWebRTCPatchRequest(pc_id=pc_id, candidates=candidates)
    )
    return {"status": "ok"}


@app.patch("/sessions/{pc_id}/api/offer")
async def ice_candidates_prebuilt(pc_id: str, data: dict):
    candidates = [
        IceCandidate(
            candidate=c["candidate"],
            sdp_mid=c["sdpMid"],
            sdp_mline_index=c["sdpMLineIndex"],
        )
        for c in data.get("candidates", [])
    ]
    await pipecat_handler.handle_patch_request(
        SmallWebRTCPatchRequest(pc_id=pc_id, candidates=candidates)
    )
    return {"status": "ok"}


@app.post("/twilio/answer")
async def twilio_answer(request: Request):
    try:
        form = await request.form()
    except Exception as e:
        logger.error(f"Failed to parse Twilio webhook: {e}")
        raise HTTPException(status_code=400, detail="Bad request")

    caller_phone = form.get("From", "")
    call_sid = form.get("CallSid", "")

    public_base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if public_base:
        ws_url = (
            public_base.replace("https://", "wss://").replace("http://", "ws://")
            + "/twilio/ws"
        )
    else:
        host = request.headers.get("host", "localhost:7860")
        scheme = "wss" if request.url.scheme == "https" else "ws"
        ws_url = f"{scheme}://{host}/twilio/ws"

    logger.info(f"Inbound call from {caller_phone}, call_sid={call_sid}")

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f'<Stream url="{ws_url}">'
        f'<Parameter name="from" value="{caller_phone}"/>'
        "</Stream></Connect></Response>"
    )
    return PlainTextResponse(twiml, media_type="text/xml")


@app.websocket("/twilio/ws")
async def twilio_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("Twilio WebSocket connected")

    try:
        await websocket.receive_text()
        raw = await websocket.receive_text()
        msg = json.loads(raw)
    except Exception as e:
        logger.error(f"Failed to read Twilio start message: {e}")
        await websocket.close()
        return

    if msg.get("event") != "start":
        logger.error(f"Expected 'start' event, got: {msg.get('event')}")
        await websocket.close()
        return

    start = msg.get("start", {})
    stream_sid = start.get("streamSid", msg.get("streamSid", ""))
    call_sid = start.get("callSid", "")
    caller_phone = start.get("customParameters", {}).get("from")

    logger.info(
        f"Twilio stream started: stream_sid={stream_sid} call_sid={call_sid} from={caller_phone}"
    )

    await run_bot_twilio(
        websocket=websocket,
        stream_sid=stream_sid,
        call_sid=call_sid,
        caller_phone=caller_phone,
    )


@app.websocket("/twilio/attorney-ws")
async def attorney_websocket(websocket: WebSocket):
    await websocket.accept()

    try:
        await websocket.receive_text()
        raw = await websocket.receive_text()
        msg = json.loads(raw)
    except Exception as e:
        logger.error(f"Failed to read attorney stream start: {e}")
        await websocket.close()
        return

    if msg.get("event") != "start":
        logger.error(f"Attorney WS: expected 'start', got {msg.get('event')}")
        await websocket.close()
        return

    start = msg.get("start", {})
    stream_sid = start.get("streamSid", msg.get("streamSid", ""))
    call_sid = start.get("callSid", "")
    params = start.get("customParameters", {})
    case_summary = params.get("case_summary", "legal matter")
    caller_sid = params.get("caller_sid", "")
    contact_name = params.get("contact_name", "")

    logger.info(
        f"Attorney stream: stream_sid={stream_sid} call_sid={call_sid} caller_sid={caller_sid}"
    )

    await run_attorney_bot(
        websocket=websocket,
        stream_sid=stream_sid,
        attorney_call_sid=call_sid,
        case_summary=case_summary,
        caller_sid=caller_sid,
        contact_name=contact_name,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
