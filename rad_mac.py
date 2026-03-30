#!/usr/bin/env python3
"""
RadiologySTT - macOS Version
==============================
Medical dictation tool using Google Cloud Speech-to-Text (medical_dictation model).
Hold F9 to record; release F9 to stop. Transcribed text is pasted at the active
cursor position in whichever window currently has focus.

Requirements:
  - Python 3.9+
  - See requirements.txt for package dependencies

macOS Accessibility Note:
  The app (or Terminal / your IDE) must be granted Accessibility access in:
  System Settings → Privacy & Security → Accessibility
  Otherwise pynput cannot intercept global key events.
"""

import os
import re
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
from google.protobuf import wrappers_pb2

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# Path to your Google Cloud service-account JSON key.
# Fill this in directly, or leave empty and set GOOGLE_APPLICATION_CREDENTIALS
# as an environment variable before running.
CREDENTIALS_PATH = ""   # e.g. "/Users/carol/.credentials/radiology-stt-key.json"

# ── Audio configuration ───────────────────────────────────────────────────────
RATE = 16_000
CHUNK = int(RATE / 10)      # 1 600 samples = 100 ms per chunk
CHANNELS = 1
FORMAT = pyaudio.paInt16

# ── Auditory cues — built-in macOS system sounds ──────────────────────────────
# afplay is called with & so it does not block the main thread.
SOUND_START = "/System/Library/Sounds/Tink.aiff"   # high tone — recording starts
SOUND_STOP  = "/System/Library/Sounds/Pop.aiff"    # low  tone — recording stops

pyautogui.FAILSAFE = False

if CREDENTIALS_PATH:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH


# ── Spoken punctuation map ─────────────────────────────────────────────────────
# Google's medical_dictation model tags spoken punctuation as [full stop] etc.
# enable_spoken_punctuation asks the API to convert them, but we also post-process
# locally as a reliable fallback for any that slip through.
#
# \s* inside brackets handles the spaces the model sometimes inserts:
# [ full stop ] or [full stop] — both are matched.

_PUNCT = [
    (r"\[\s*full\s*stop\s*\]",          "."),
    (r"\[\s*period\s*\]",               "."),
    (r"\[\s*comma\s*\]",                ","),
    (r"\[\s*semicolon\s*\]",            ";"),
    (r"\[\s*colon\s*\]",                ":"),
    (r"\[\s*question\s*mark\s*\]",      "?"),
    (r"\[\s*exclamation\s*mark\s*\]",   "!"),
    (r"\[\s*exclamation\s*point\s*\]",  "!"),
    (r"\[\s*hyphen\s*\]",               "-"),
    (r"\[\s*dash\s*\]",                 " \u2014 "),   # em dash
    (r"\[\s*slash\s*\]",                "/"),
    (r"\[\s*open\s*bracket\s*\]",       "("),
    (r"\[\s*close\s*bracket\s*\]",      ")"),
    (r"\[\s*open\s*parenthesis\s*\]",   "("),
    (r"\[\s*close\s*parenthesis\s*\]",  ")"),
    (r"\[\s*new\s*line\s*\]",           "\n"),
    (r"\[\s*next\s*line\s*\]",          "\n"),
    (r"\[\s*new\s*paragraph\s*\]",      "\n\n"),
    (r"\[\s*next\s*paragraph\s*\]",     "\n\n"),
    # Plain spoken phrases (unambiguous in a radiology context)
    (r"\bfull\s+stop\b",                "."),
    (r"\bnew\s+line\b",                 "\n"),
    (r"\bnext\s+line\b",                "\n"),
    (r"\bnew\s+paragraph\b",            "\n\n"),
    (r"\bnext\s+paragraph\b",           "\n\n"),
]

def process_punctuation(text: str) -> str:
    """Convert spoken/tagged punctuation commands to their symbols,
    then capitalise the first letter of the transcript and the first
    letter after any sentence-ending punctuation (. ? !)."""
    for pattern, replacement in _PUNCT:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Remove spurious spaces immediately before punctuation marks
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text).strip()
    # Capitalise after . ? ! (followed by a space and a lowercase letter)
    text = re.sub(r"([.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    # Capitalise the very first character
    if text:
        text = text[0].upper() + text[1:]
    return text


# ── Measurement / dimension normalisation ─────────────────────────────────────
# Handles the most common radiology measurement patterns that the model either
# leaves in spoken form or doesn't abbreviate.

_UNIT_MAP = [
    (r"\bcentimeters?\b",       "cm"),
    (r"\bcentimetres?\b",       "cm"),
    (r"\bmillimeters?\b",       "mm"),
    (r"\bmillimetres?\b",       "mm"),
    (r"\bkilometers?\b",        "km"),
    (r"\bkilometres?\b",        "km"),
    (r"\bmilligrams?\b",        "mg"),
    (r"\bmicrograms?\b",        "mcg"),
    (r"\bnanograms?\b",         "ng"),
    (r"\bkilograms?\b",         "kg"),
    (r"\bgrams?\b",             "g"),
    (r"\bmilliliters?\b",       "mL"),
    (r"\bmillilitres?\b",       "mL"),
    (r"\bliters?\b",            "L"),
    (r"\blitres?\b",            "L"),
    (r"\bmegahertz\b",          "MHz"),
    (r"\bkilohertz\b",          "kHz"),
    (r"\bhounsfield\s+units?\b","HU"),
    (r"\bsecond[s]?\b",         "s"),
    (r"\bmilliseconds?\b",      "ms"),
]

# Matches integers and decimals (e.g. 2, 2.5, 0.8)
_NUM = r"\d+(?:\.\d+)?"

def process_measurements(text: str) -> str:
    """
    1. Spoken decimal  : "2 point 5"       → "2.5"
    2. Dimension string: "2 by 3 by 1.5"   → "2 × 3 × 1.5"
                         "2 x 3"           → "2 × 3"
    3. Unit names      : "centimeters"     → "cm", "milligrams" → "mg", etc.
    """
    # Spoken decimal: must be digits on both sides so "the point is" is unaffected
    text = re.sub(r"\b(\d+)\s+point\s+(\d+)\b", r"\1.\2", text, flags=re.IGNORECASE)

    # Dimension separators: iterate until stable (handles 3-D chains like 2×3×4)
    _dim = re.compile(rf"({_NUM})\s+(?:by|x)\s+({_NUM})", re.IGNORECASE)
    while True:
        updated = _dim.sub(r"\1 × \2", text)
        if updated == text:
            break
        text = updated

    # Unit abbreviation
    for pattern, abbrev in _UNIT_MAP:
        text = re.sub(pattern, abbrev, text, flags=re.IGNORECASE)

    return text


# ── Microphone stream ─────────────────────────────────────────────────────────

class MicrophoneStream:
    """
    Opens a non-blocking PyAudio input stream using the system default
    microphone and exposes a generator that yields raw PCM bytes.

    Stopping contract:
      Caller sets self.closed = True and puts None in self._buff.
      The generator terminates, the gRPC request stream ends, Google Speech
      flushes remaining audio and returns the final is_final result, then the
      response stream closes naturally — no forced break needed.
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
        self._buff.put(None)
        if self._pa:
            self._pa.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Yield audio chunks until the stream closes."""
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
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
        time.sleep(0.05)
    finally:
        pyperclip.copy(previous)


# ── Transcription session ─────────────────────────────────────────────────────

def listen_and_type(
    client: speech.SpeechClient,
    streaming_config: speech.StreamingRecognitionConfig,
    stop_event: threading.Event,
) -> None:
    """
    One push-to-talk recording session.

    Flow:
      F9 held     → audio streams to Google Speech in real time.
      F9 released → stop_event is set by the pynput on_release callback.
                    _watch_stop thread closes the mic only (stops sending audio).
                    Google flushes the buffer and returns the final is_final result,
                    then closes the response stream on its own.
                    The for-loop ends naturally — no forced break.
      Stop sound plays AFTER the last word has been typed.
    """
    with MicrophoneStream(RATE, CHUNK) as stream:

        def _watch_stop():
            """Wait for F9 release, then close audio only — don't break the loop."""
            stop_event.wait()
            stream.closed = True
            stream._buff.put(None)

        threading.Thread(target=_watch_stop, daemon=True).start()

        requests = (
            speech.StreamingRecognizeRequest(audio_content=audio)
            for audio in stream.generator()
        )

        try:
            for response in client.streaming_recognize(streaming_config, requests):
                if not response.results:
                    continue
                result = response.results[0]
                if result.is_final and result.alternatives:
                    raw = result.alternatives[0].transcript.strip()
                    transcript = process_measurements(process_punctuation(raw))
                    if transcript:
                        confidence = result.alternatives[0].confidence
                        print(f"  --> (conf={confidence:.2f}) {transcript}")
                        type_text(transcript + " ")
        except Exception as exc:
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
        enable_automatic_punctuation=True,
        enable_spoken_punctuation=wrappers_pb2.BoolValue(value=True),
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    print("\n--- RADIOLOGY DICTATION (macOS) ---")
    print("1. Open your Radiology Reporting Window.")
    print("2. Hold [F9] to dictate.")
    print("3. Release to stop — text types after the last word. Ctrl+C to exit.\n")

    f9_pressed   = threading.Event()
    stop_event   = threading.Event()
    session_active = threading.Event()   # guard against re-entrant presses

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
            f9_pressed.wait()
            f9_pressed.clear()
            stop_event.clear()
            session_active.set()

            os.system(f"afplay '{SOUND_START}' &")
            print("  [ON] Listening... Release F9 to stop.")

            listen_and_type(client, streaming_config, stop_event)

            # Sound plays only after gRPC has closed and all text has been typed
            os.system(f"afplay '{SOUND_STOP}' &")
            print("  [OFF] Done.\n")
            session_active.clear()
            time.sleep(0.15)

    except KeyboardInterrupt:
        listener.stop()
        print("\n[status] Exiting.")


if __name__ == "__main__":
    main()
