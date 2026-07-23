"""Runtime mic mute: a pass-through processor that drops input audio while muted.

Placed right after transport.input() and before STT/VAD, so flipping `muted`
makes the agent deaf to the mic without tearing down the audio stream — toggling
is instant and reversible. Only the raw mic-audio frames are swallowed; every
other frame (StartFrame, cancellation, etc.) always passes so the pipeline keeps
running normally.
"""

from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MicGate(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.muted = False

    def toggle(self) -> bool:
        """Flip mute; returns the new muted state."""
        self.muted = not self.muted
        return self.muted

    def _should_drop(self, frame: Frame) -> bool:
        """Drop only raw mic audio, and only while muted; everything else flows."""
        return self.muted and isinstance(frame, InputAudioRawFrame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._should_drop(frame):
            return  # swallow mic audio — STT and VAD downstream see silence
        await self.push_frame(frame, direction)
