from bot_core import run_caller_bot
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams


async def run_bot_webrtc(connection: SmallWebRTCConnection) -> None:
    transport = SmallWebRTCTransport(
        connection,
        TransportParams(audio_out_enabled=True, audio_in_enabled=True),
    )
    await run_caller_bot(transport)
