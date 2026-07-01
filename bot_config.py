import os
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_VOICE_ID = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"
HOLD_MUSIC_URL = "https://demo.twilio.com/docs/classic.mp3"
ATTORNEY_BOT_WEBSOCKET_PATH = "/twilio/attorney-ws"

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
