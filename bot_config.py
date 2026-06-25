import os
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_VOICE_ID = "32b3f3c5-7171-46aa-abe7-b598964aa793"
HOLD_MUSIC_URL = "https://demo.twilio.com/docs/classic.mp3"
ATTORNEY_BOT_WEBSOCKET_PATH = "/twilio/attorney-ws"

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
