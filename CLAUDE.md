# AnsLegal Pipecat Voice Bot — Developer Guide

## Project Overview

Voice AI receptionist for law firms built on Pipecat + Twilio. Bot collects legal intake information from callers and optionally warm-transfers to an available attorney.

### File Map

| File | Role |
|---|---|
| `bot_core.py` | All business logic — flow graph, nodes, transfer registry, LLM/TTS/STT setup |
| `bot_config.py` | Shared constants (voice ID, env vars). Exists to break circular imports |
| `bot_twilio.py` | Twilio transport layer — call receive/dial, WebSocket handler, warm transfer dialling |
| `bot_webrtc.py` | WebRTC transport layer — browser-based testing via WHIP |
| `server.py` | FastAPI server — mounts Twilio and WebRTC routes |
| `api_client.py` | HTTP calls to the AnsLegal backend (`fetch_company_case_types`, `fetch_case_type_questions`) |

### Stack

- **LLM:** OpenAI (gpt-4o by default)
- **TTS:** Cartesia
- **STT:** Cartesia
- **VAD:** Silero
- **Flow engine:** `pipecat_flows` (`FlowManager`, `NodeConfig`)
- **Transport:** Twilio (prod) / WebRTC (dev)

---

## Intended Bot Behaviour

> **This section is the canonical specification for how the bot should behave.**
> All prompt changes, node changes, and LLM setting changes must be validated against it.
> Edit this section first when requirements change; then update code to match.

### 1. Greeting & Opening

- Bot greets the caller immediately on connection using the firm's configured greeting (fetched from `fetch_company_case_types`).
- After greeting, bot asks an open question: *"How can I help you today?"* or similar.
- Bot does **not** generate a second LLM greeting — the `tts_say` action fires the greeting, and `respond_immediately=False` suppresses the LLM from speaking again unprompted.

### 2. Background Case-Type Fetch

- On connection, `fetch_company_case_types(phone)` is called immediately in the background before the caller speaks.
- The case type list is embedded into the `select_case_type` node's LLM prompt so the model can map the caller's description to a known case type ID.

### 3. Case-Type Detection from First Answer

- The caller's first free-form answer (which may be long) is analysed by the LLM to detect intent.
- The LLM calls `select_case_type(case_type_id, caller_description)` with the matched type.
- If the detected case type is not handled by the firm, the bot politely informs the caller and ends the call. It does not transfer or continue intake.
- Once a case type is confirmed, `fetch_case_type_questions(firm_id, case_type_id)` is called to retrieve the ordered question list for that case type.

### 4. Information Pre-Fill from First Answer (Intro-Dump Memory)

- The caller's first answer is often information-rich. They may answer future questions without being asked.
- The LLM **must** extract any answers to future questions that are present in the caller's first message and store them.
- When the bot later reaches a question that was already answered:
  - It does **not** skip the question silently.
  - It asks a **confirmation question** instead. Example:
    > *"You mentioned that you and your partner don't own any property together — is that correct?"*
  - This acts as a guardrail: it prevents silently recording wrong extractions and gives the caller a chance to correct or expand.
- This applies for every subsequent question in the flow, not just the first.

### 5. Question Flow

- Questions are asked one at a time in the order returned by the API.
- Bot waits for a confident answer before advancing.
- If the bot cannot understand the caller's answer with confidence (e.g. heavy background noise, unclear speech, ambiguous answer) it asks again — up to a configurable number of retries — before saving a low-confidence answer.
- Bot never rushes through questions. Pacing should feel natural and conversational, not form-like.

### 6. Noise Handling & Confidence

- Silero VAD filters silence and non-speech noise before STT.
- If the STT transcript is empty, too short, or clearly garbled, the bot asks the caller to repeat rather than saving junk.
- The bot should be able to handle accents, hesitations, filler words ("um", "uh", "you know") without misclassifying them as meaningful answers.

### 7. Latency & Human Feel

- Responses should feel immediate but not robotic. Target end-of-utterance to first TTS byte < 1.5 s.
- The bot uses short, natural acknowledgement phrases between questions (*"Got it"*, *"Thanks for that"*, *"I see"*) to signal it heard the caller before asking the next question.
- Avoid long monologue responses — keep each turn concise.

### 8. Emotional Awareness

- The bot must detect frustration, distress, or impatience in the caller's tone or language.
- On frustration: slow down, acknowledge the emotion briefly, and continue. Example:
  > *"I understand this is a stressful situation. I'm going to make sure we have everything we need."*
- On distress (e.g. domestic violence, custody crisis): respond with empathy, do not rush, and consider fast-tracking to attorney transfer if available.
- Never dismiss or minimise the caller's emotional state.

### 9. Role Boundaries — Receptionist Only

- The bot is a **receptionist**, not a legal advisor.
- It collects information. It does not:
  - Interpret laws or statutes
  - Comment on the merits or likely outcome of a case
  - Give any advice that could be construed as legal counsel
- If a caller asks for legal advice, the bot redirects:
  > *"I'm not able to give legal advice, but I can make sure the attorney has all the details they need when they speak with you."*
- The bot never drifts from intake collection. If the caller goes off-topic, it gently steers back.

### 10. Warm Transfer

- After intake is complete, the bot summarises collected information and attempts to transfer to an available attorney.
- The attorney receives a briefing before being connected to the caller.
- If no attorney is available, the bot takes a callback number and closes politely.

---

## Prompt Change Policy

> **Every prompt change must be treated as a high-risk code change.**

- Prompts control all bot behaviour. A small wording change can break intent detection, cause role drift, or produce double-greetings.
- Before editing any `role: system` content, `tts_say` text, or node message:
  1. Read the **Intended Bot Behaviour** section above and identify which behaviour the change affects.
  2. Make the smallest possible change that achieves the goal.
  3. Test on a real call (or WebRTC session) before merging.
  4. Document what changed and why in the PR description.
- Never add legal commentary or opinions to any prompt — the bot must remain a receptionist.
- Never add a second greeting to any node that follows the opening `tts_say`.

---

## Architecture Notes

### Circular Import Resolution

`bot_core.py` and `bot_twilio.py` had a circular dependency. Resolved by:
- `bot_config.py` holds shared constants (no runtime imports from other bot files).
- `bot_twilio.py` imports constants from `bot_config`, runtime objects from `bot_core`.
- `bot_core.py` uses a lazy import of `_warm_transfer` inside `handle_question_transfer` to avoid import-time circularity.

### TransferRegistry

`TransferRegistry` is a singleton instance (`transfer_registry`) in `bot_core.py`. It coordinates state between the caller bot pipeline and the attorney bot WebSocket. Both sides look up state by `call_sid`. Do not call methods on the class directly — always use the instance.

### Flow Manager

Flows are built with `pipecat_flows`. Each node is a `NodeConfig`. Key patterns:
- `respond_immediately=False` on `select_case_type` prevents double greeting.
- `LLMUpdateSettingsFrame` is queued via `worker.queue_frame` to update LLM settings (e.g. `tool_choice`) mid-call.
- Terminal nodes (`save_and_end`, `attorney_unavailable`) must reset `tool_choice` to `"auto"` and clear `functions=[]` to avoid OpenAI 400 errors.

### In-Memory Transfer Context

`_transfer_contexts` (dict in `bot_core.py`) stores collected intake data between the caller bot and attorney bot. Keyed by `call_sid`. This is in-memory only — a Redis replacement is planned for multi-instance deployments.

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM |
| `CARTESIA_API_KEY` | TTS + STT |
| `TWILIO_ACCOUNT_SID` | Twilio auth |
| `TWILIO_AUTH_TOKEN` | Twilio auth |
| `ANSLEGAL_API_BASE` | Backend API base URL |
| `ANSLEGAL_API_KEY` | Backend API key |
| `PORT` | FastAPI server port (default 8765) |

---

## Running Locally

```bash
# Install deps
uv sync

# WebRTC dev mode (browser)
python server.py

# Twilio prod mode requires ngrok or a public URL configured in Twilio console
```
