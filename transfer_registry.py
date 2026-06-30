import asyncio


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
