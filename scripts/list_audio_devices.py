"""List audio device indices for AUDIO_IN_DEVICE / AUDIO_OUT_DEVICE routing.

    uv run python scripts/list_audio_devices.py
"""

import pyaudio


def main():
    pa = pyaudio.PyAudio()
    default_in = pa.get_default_input_device_info()["index"]
    default_out = pa.get_default_output_device_info()["index"]
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        tags = []
        if d["maxInputChannels"]:
            tags.append(f"in:{d['maxInputChannels']}ch")
        if d["maxOutputChannels"]:
            tags.append(f"out:{d['maxOutputChannels']}ch")
        if i == default_in:
            tags.append("DEFAULT-IN")
        if i == default_out:
            tags.append("DEFAULT-OUT")
        print(f"{i:3d}: {d['name']}  ({', '.join(tags)})")
    pa.terminate()


if __name__ == "__main__":
    main()
