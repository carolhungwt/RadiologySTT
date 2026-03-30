#!/usr/bin/env python3
"""
RadiologySTT - macOS Version
==============================
Medical dictation tool using Google Cloud Speech-to-Text (medical_dictation model).
Hold F9 to record; release F9 to stop. Transcribed text is pasted at the active
cursor position in whichever window currently has focus.

Requirements:
  - Python 3.9+
  - GOOGLE_APPLICATION_CREDENTIALS env var pointing to your service-account JSON
  - See requirements.txt for package dependencies

macOS Accessibility Note:
  The app (or Terminal / your IDE) must be granted Accessibility access in:
  System Settings → Privacy & Security → Accessibility
  Otherwise pynput cannot intercept global key events.
"""

import os
import sys
import queue
import threading
import time
from typing import Optional

import pyaudio
import pyautogui
import pyperclip
from pynput import keyboard as pynput_keyboard
from google.cloud import speech

# ── Audio configuration ───────────────────────────────────────────────────────
RATE = 16_000
CHUNK = int(RATE / 10)      # 1 600 samples = 100 ms per chunk
CHANNELS = 1
FORMAT = pyaudio.paInt16

# ── Auditory cues — built-in macOS system sounds ──────────────────────────────
# afplay is called with & so it does not block the main thread.
SOUND_START = "/System/Library/Sounds/Tink.aiff"   # high tone — recording starts
SOUND_STOP  = "/System/Library/Sounds/Pop.aiff"    # low  tone — recording stops

# Prevent pyautogui from raising an exception if the mouse reaches a screen corner
pyautogui.FAILSAFE = False


# ── Microphone stream ─────────────────────────────────────────────────────────

class MicrophoneStream:
    """
    Opens a non-blocking PyAudio input stream using the system default
    microphone and exposes a generator that yields raw PCM bytes.
    Thread-safe via an internal queue.
    """

    def __init__(self, rate: int, chunk: int):
        self.rate = rate
        self.chunk = chunk
        self._buff: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self.closed = True

    def __enter__(self) -> "MicrophoneStream":
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._fill_buffer,
        )
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
    Clipboard-paste handles all Unicode / medical symbols reliably.
    The original clipboard content is restored afterwards.
    """
    previous = pyperclip.paste()
    try:
        pyperclip.copy(text)
        pyautogui.hotkey("command", "v")
        time.sleep(0.05)            # brief pause so the paste registers
    finally:
        pyperclip.copy(previous)    # always restore the clipboard


# ── Transcription session ─────────────────────────────────────────────────────

def run_transcription(
    client: speech.SpeechClient,
    streaming_config: speech.StreamingRecognitionConfig,
    stop_event: threading.Event,
) -> None:
    """
    Open a single push-to-talk recording session.
    Streams audio to Google Cloud Speech-to-Text until stop_event is set
    (i.e. when the pynput on_release callback fires for F9).
    Types each is_final transcript at the active cursor position.
    """
    with MicrophoneStream(RATE, CHUNK) as stream:

        def _watch_stop():
            """Close the stream as soon as the stop signal arrives."""
            stop_event.wait()
            stream.closed = True
            stream._buff.put(None)

        threading.Thread(target=_watch_stop, daemon=True).start()

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
        interim_results=True,   # stream partial results (only finals are typed)
    )

    print("=" * 58)
    print("  RadiologySTT — macOS  |  Hold F9 to dictate")
    print("  Ctrl+C to exit")
    print("=" * 58)

    # Shared state between the pynput listener callbacks and the main loop
    f9_pressed = threading.Event()
    stop_event  = threading.Event()
    session_active = threading.Event()     # guard against re-entrant presses

    def on_press(key):
        if key == pynput_keyboard.Key.f9 and not session_active.is_set():
            f9_pressed.set()

    def on_release(key):
        if key == pynput_keyboard.Key.f9:
            stop_event.set()

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while True:
            # Block until F9 is pressed
            f9_pressed.wait()
            f9_pressed.clear()
            stop_event.clear()
            session_active.set()

            os.system(f"afplay '{SOUND_START}' &")      # non-blocking start beep
            print("[status] Recording …")

            run_transcription(client, streaming_config, stop_event)

            os.system(f"afplay '{SOUND_STOP}' &")       # non-blocking stop beep
            print("[status] Stopped.\n")
            session_active.clear()
            time.sleep(0.15)                            # debounce before next press

    except KeyboardInterrupt:
        listener.stop()
        print("\n[status] Exiting.")


if __name__ == "__main__":
    main()
