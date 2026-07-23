"""Meeting mode: hold LLM runs until 'r' is pressed in the terminal.

The user aggregator adds each finished utterance to the shared LLMContext
BEFORE pushing the LLMContextFrame that triggers a run, so swallowing those
frames loses nothing — context keeps accumulating while the agent stays
silent. On 'r', main.py arms the gate and queues an LLMRunFrame at the
pipeline head; the user aggregator converts it into a fresh LLMContextFrame,
which the armed gate passes through exactly once.
"""

import os
import sys
import termios
import tty
from asyncio import AbstractEventLoop
from collections.abc import Callable

from loguru import logger

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MeetingGate(FrameProcessor):
    """Swallows downstream LLMContextFrames while holding; arm() lets the next
    one through. Sits between the user aggregator and the LLM."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._held = 0
        self._release_next = False

    @property
    def held(self) -> int:
        return self._held

    def arm(self):
        self._release_next = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._release_next:
                self._release_next = False
                self._held = 0
                await self.push_frame(frame, direction)
            else:
                self._held += 1
                logger.info(
                    f"meeting mode: holding ({self._held} utterance(s) pending — press 'r')"
                )
            return
        await self.push_frame(frame, direction)


def install_key_listener(loop: AbstractEventLoop, on_r: Callable[[], None]) -> Callable[[], None]:
    """Single-keypress stdin reader; on_r fires on 'r' with no Enter needed.

    Returns a restore() the caller must run in a finally block — it removes
    the reader and puts the terminal back. cbreak keeps ISIG, so Ctrl-C still
    works. Caller must check stdin.isatty() first, and install only after all
    input() prompts are done.
    """
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def _read():
        if os.read(fd, 1) in (b"r", b"R"):
            on_r()

    loop.add_reader(fd, _read)

    def restore():
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

    return restore
