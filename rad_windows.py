#!/usr/bin/env python3
"""
RadiologySTT - Windows Version
================================
Medical dictation tool using Google Cloud Speech-to-Text (medical_dictation model).
Hold F9 to record; release F9 to stop. Transcribed text is pasted at the active
cursor position in whichever window currently has focus.

Requirements:
  - Python 3.9+
  - GOOGLE_APPLICATION_CREDENTIALS env var pointing to your service-account JSON
  - See requirements.txt for package dependencies
"""

import os
import sys
import queue
import threading
import time
import winsound
from typing import Optional

import keyboard
import pyaudio
import pyautogui
import pyperclip
from google.cloud import speech

# ── Audio configuration ───────────────────────────────────────────────────────
RATE = 16_000
CHUNK = int(RATE / 10)      # 1 600 samples = 100 ms per chunk
CHANNELS = 1
FORMAT = pyaudio.paInt16

# ── Auditory cues (frequency Hz, duration ms) ─────────────────────────────────
BEEP_START = (880, 150)     # high tone — recording starts
BEEP_STOP  = (440, 150)     # low  tone — recording stops

# Prevent pyautogui from raising an exception if the mouse reaches a screen corner
pyautogui.FAILSAFE = False


# ── Device discovery ──────────────────────────────────────────────────────────

def find_philips_device(pa: pyaudio.PyAudio) -> Optional[int]:
    """
    Scan PyAudio input devices for a Philips SpeechMike.
    Returns the device index, or None to fall back to the system default.
    """
    target_keywords = ("philips", "speechmike")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) < 1:
            continue
        name_lower = info.get("name", "").lower()
        if any(kw in name_lower for kw in target_keywords):
            print(f"[device] Philips SpeechMike found: \"{info['name']}\" (index {i})")
            return i
    print("[device] Philips SpeechMike not detected — using system default microphone.")
    return None


# ── Microphone stream ─────────────────────────────────────────────────────────

class MicrophoneStream:
    """
    Opens a non-blocking PyAudio input stream and exposes a generator that
    yields raw PCM bytes. Thread-safe via an internal queue.
    """

    def __init__(self, rate: int, chunk: int, device_index: Optional[int] = None):
        self.rate = rate
        self.chunk = chunk
        self.device_index = device_index
        self._buff: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self.closed = True

    def __enter__(self) -> "MicrophoneStream":
        self._pa = pyaudio.PyAudio()
        kwargs = dict(
            format=FORMAT,
            channels=CHANNELS,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._fill_buffer,
        )
        if self.device_index is not None:
            kwargs["input_device_index"] = self.device_index
        self._stream = self._pa.open(**kwargs)
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self.closed = True
        self._buff.put(None)        # sentinel — unblocks the generator
        if self._pa:
            self._pa.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self, stop_event: threading.Event):
        """Yield audio chunks until the stream closes or stop_event is set."""
        while not self.closed:
            if stop_event.is_set():
                return
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            # Drain any extra buffered chunks without blocking
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)


# ── Text injection ─────────────────────────────────────────────────────────────

def type_text(text: str) -> None:
    """
    Paste text at the current cursor position via the system clipboard.
    Clipboard-paste is faster and handles all Unicode / medical symbols
    reliably, unlike pyautogui.typewrite() which is ASCII-only.
    The original clipboard content is restored afterwards.
    """
    previous = pyperclip.paste()
    try:
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.05)            # brief pause so the paste registers
    finally:
        pyperclip.copy(previous)    # always restore the clipboard


# ── Transcription session ─────────────────────────────────────────────────────

def run_transcription(
    client: speech.SpeechClient,
    streaming_config: speech.StreamingRecognitionConfig,
    device_index: Optional[int],
) -> None:
    """
    Open a single push-to-talk recording session.
    Streams audio to Google Cloud Speech-to-Text until F9 is released.
    Types each is_final transcript at the active cursor position.
    """
    stop_event = threading.Event()

    with MicrophoneStream(RATE, CHUNK, device_index) as stream:

        def _monitor_f9():
            """Signal stop when the user releases F9."""
            while keyboard.is_pressed("F9"):
                time.sleep(0.02)
            stop_event.set()
            stream.closed = True
            stream._buff.put(None)

        threading.Thread(target=_monitor_f9, daemon=True).start()

        requests = (
            speech.StreamingRecognizeRequest(audio_content=audio)
            for audio in stream.generator(stop_event)
        )

        try:
            for response in client.streaming_recognize(streaming_config, requests):
                if stop_event.is_set():
                    break
                if not response.results:
                    continue
                result = response.results[0]
                if result.is_final and result.alternatives:
                    transcript = result.alternatives[0].transcript.strip()
                    if transcript:
                        confidence = result.alternatives[0].confidence
                        print(f"[transcript] (conf={confidence:.2f}) {transcript}")
                        type_text(transcript + " ")
        except Exception as exc:
            # Streaming errors (e.g. 5-minute limit) are non-fatal;
            # the user simply presses F9 again to start a fresh session.
            print(f"[error] Streaming ended: {exc}", file=sys.stderr)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Discover Philips SpeechMike once at startup
    pa = pyaudio.PyAudio()
    device_index = find_philips_device(pa)
    pa.terminate()

    # Build the Google Cloud Speech client and config once (reused every session)
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
        model="medical_dictation",
        use_enhanced=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,    # stream partial results (only finals are typed)
    )

    print("=" * 58)
    print("  RadiologySTT — Windows  |  Hold F9 to dictate")
    print("  Ctrl+C to exit")
    print("=" * 58)

    try:
        while True:
            keyboard.wait("F9")                     # block until F9 is pressed
            winsound.Beep(*BEEP_START)              # high beep — recording starts
            print("[status] Recording …")

            run_transcription(client, streaming_config, device_index)

            winsound.Beep(*BEEP_STOP)               # low beep  — recording stops
            print("[status] Stopped.\n")
            time.sleep(0.15)                        # debounce before next press

    except KeyboardInterrupt:
        print("\n[status] Exiting.")


if __name__ == "__main__":
    main()
