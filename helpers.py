# Short, natural acknowledgements spoken before the NEXT question so the bot signals
# it heard the caller and doesn't feel form-like Rotated by question
# index so consecutive acks differ.
# TODO this can be enhanced but for now we are sticking to predetermined language
import json
import os

import httpx
from loguru import logger


# Warm, varied acks. Blank entries mean "no ack this turn" — acknowledging EVERY
# question made the bot feel form-like and repetitive ("Acknowledged" every time).
# Stiff words (Acknowledged/Understood) removed in favour of natural ones.
_ACK_PHRASES = [
    "",
    "Got it.",
    "",
    "Thanks for that.",
    "",
    "Okay, thank you.",
]


def _ack_for(index: int) -> str:
    """Pick a rotating acknowledgement phrase. Deterministic (by index) so it varies
    across questions without randomness. ~half are blank, so the bot acknowledges
    only occasionally rather than before every single question."""
    return _ACK_PHRASES[index % len(_ACK_PHRASES)]


# Phrases ink-whisper hallucinates on silence/noise. If the entire transcript is
# one of these (case-insensitive, stripped of punctuation), treat it as no answer.
_STT_HALLUCINATION_PHRASES = {
    "you",
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "whatever",
    "amen",
    "bye",
    "okay",
    "ok",
    ".",
}


def _is_low_confidence(answer: str) -> bool:
    """True when a transcript is empty, too short, or a known STT hallucination —
    i.e. not a real answer worth saving. Used to trigger a re-ask (CLAUDE.md §6).
    """
    cleaned = answer.strip().lower().strip(".!?,- ")
    if not cleaned:
        return True
    # Single stray character is almost always noise, never a real answer.
    if len(cleaned) < 2:
        return True
    if cleaned in _STT_HALLUCINATION_PHRASES:
        return True
    return False


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
