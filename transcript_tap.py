"""A pass-through processor that reports transcript text to the TUI.

Pipecat 1.6 here has no TranscriptProcessor, so we tap the two frame types we
care about and forward everything untouched:

  - TranscriptionFrame  — finalized user speech (place after STT)
  - TTSTextFrame        — a chunk of text the bot is about to speak (place after TTS)

Callbacks are invoked on the pipeline's event loop; keep them non-blocking
(the TUI just drops the text on an asyncio.Queue).
"""

from collections.abc import Callable

from pipecat.frames.frames import Frame, TranscriptionFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TranscriptTap(FrameProcessor):
    def __init__(
        self,
        *,
        on_user: Callable[[str], None] | None = None,
        on_bot: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._on_user = on_user
        self._on_bot = on_bot

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._on_user and isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if text and getattr(frame, "finalized", True):
                self._on_user(text)
        elif self._on_bot and isinstance(frame, TTSTextFrame):
            text = (frame.text or "").strip()
            if text:
                self._on_bot(text)

        await self.push_frame(frame, direction)
