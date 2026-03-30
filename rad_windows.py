#!/usr/bin/env python3
"""
RadiologySTT - Windows Version
================================
Medical dictation tool using Google Cloud Speech-to-Text (medical_dictation model).
Hold F9 to record; release F9 to stop. Transcribed text is pasted at the active
cursor position in whichever window currently has focus.

Requirements:
  - Python 3.9+
  - See requirements.txt for package dependencies
"""

import os
import re
import sys
import queue
import time
import winsound
from typing import Optional

import keyboard
import pyaudio
import pyautogui
import pyperclip
from google.cloud import speech
from google.protobuf import wrappers_pb2

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# Path to your Google Cloud service-account JSON key.
# Fill this in directly, or leave empty and set GOOGLE_APPLICATION_CREDENTIALS
# as an environment variable before running.
CREDENTIALS_PATH = r""   # e.g. r"C:\Users\carol\credentials\radiology-stt-key.json"

RECORD_HOTKEY = "F9"

# ── Audio settings ─────────────────────────────────────────────────────────────
RATE = 16_000
CHUNK = int(RATE / 10)   # 1 600 samples = 100 ms per chunk

# ── Auditory cues (frequency Hz, duration ms) ─────────────────────────────────
BEEP_START = (700, 200)  # high tone — recording starts
BEEP_STOP  = (500, 200)  # low  tone — recording stops (plays after last word types)

pyautogui.FAILSAFE = False

if CREDENTIALS_PATH:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH


# ── Spoken punctuation map ─────────────────────────────────────────────────────
# Google's medical_dictation model tags spoken punctuation as [full stop] etc.
# enable_spoken_punctuation asks the API to convert them, but we also post-process
# locally as a reliable fallback for any that slip through.
#
# Patterns are tried in order — most specific first.
# Only unambiguous commands are included to avoid corrupting real medical words.

_PUNCT = [
    # Google's bracket-tag format.
    # \s* handles the spaces the model sometimes inserts: [ full stop ] or [full stop]
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
    (r"\[\s*next\s*line\s*\]",          "\n"),   # "next line" is the common spoken form
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
    yields raw PCM bytes.

    Stopping contract:
      Caller sets self.closed = True and puts None in self._buff.
      The generator then terminates, the gRPC request stream ends, Google Speech
      flushes any remaining audio and returns the final is_final result, and the
      response stream closes naturally — no forced break needed.
    """

    def __init__(self, rate: int, chunk: int, device_index: Optional[int] = None):
        self._rate = rate
        self._chunk = chunk
        self._device_index = device_index
        self._buff: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._audio_interface: Optional[pyaudio.PyAudio] = None
        self._audio_stream = None
        self.closed = True

    def __enter__(self) -> "MicrophoneStream":
        self._audio_interface = pyaudio.PyAudio()
        kwargs = dict(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        if self._device_index is not None:
            kwargs["input_device_index"] = self._device_index
        self._audio_stream = self._audio_interface.open(**kwargs)
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

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
    Paste text at the current cursor position via the clipboard.
    Faster and more reliable than pyautogui.write() for medical terminology,
    punctuation, and Unicode. Restores the previous clipboard content.
    """
    previous = pyperclip.paste()
    try:
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.05)
    finally:
        pyperclip.copy(previous)


# ── Transcription session ─────────────────────────────────────────────────────

def listen_and_type(
    client: speech.SpeechClient,
    streaming_config: speech.StreamingRecognitionConfig,
    device_index: Optional[int],
) -> None:
    """
    One push-to-talk recording session.

    Flow:
      F9 held   → audio streams to Google Speech in real time
      F9 released → on_f9_release closes the mic only (stops sending audio)
                    Google flushes the remaining audio buffer and returns the
                    final is_final result, then closes the response stream.
                    The for-loop ends naturally — no forced break.
      Beep plays AFTER the last word has been typed, not before.
    """
    with MicrophoneStream(RATE, CHUNK, device_index) as stream:

        def on_f9_release(e):
            # Stop sending audio — do NOT break the response loop yet.
            # Google Speech will process whatever it already received, return
            # the is_final result, then close the stream on its own.
            stream.closed = True
            stream._buff.put(None)

        hook = keyboard.on_release_key(RECORD_HOTKEY, on_f9_release)

        try:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)

            # Iterate until gRPC closes the stream (happens naturally after
            # the request stream ends and Google returns the final result).
            for response in responses:
                if not response.results:
                    continue
                result = response.results[0]
                if not result.alternatives:
                    continue

                if result.is_final:
                    raw = result.alternatives[0].transcript.strip()
                    transcript = process_measurements(process_punctuation(raw))
                    if transcript:
                        confidence = result.alternatives[0].confidence
                        print(f"  --> (conf={confidence:.2f}) {transcript}")
                        type_text(transcript + " ")

        except Exception as exc:
            print(f"[error] Streaming ended: {exc}", file=sys.stderr)
        finally:
            keyboard.unhook(hook)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Discover Philips SpeechMike once at startup
    pa = pyaudio.PyAudio()
    device_index = find_philips_device(pa)
    pa.terminate()

    # Build Google Speech client and config once; reused on every F9 press
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
        model="medical_dictation",
        use_enhanced=True,
        enable_automatic_punctuation=True,
        # Ask Google to convert spoken punctuation ("full stop" → ".") at source
        enable_spoken_punctuation=wrappers_pb2.BoolValue(value=True),
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    print("\n--- RADIOLOGY DICTATION (WINDOWS) ---")
    print("1. Open your Radiology Reporting Window.")
    print(f"2. Hold [{RECORD_HOTKEY}] to dictate.")
    print("3. Release to stop — text types after the last word. Ctrl+C to exit.\n")

    while True:
        try:
            keyboard.wait(RECORD_HOTKEY)
            winsound.Beep(*BEEP_START)
            print(f"  [ON] Listening... Release {RECORD_HOTKEY} to stop.")

            listen_and_type(client, streaming_config, device_index)

            # Beep plays only after gRPC has closed and all text has been typed
            winsound.Beep(*BEEP_STOP)
            print("  [OFF] Done.\n")
            time.sleep(0.2)

        except KeyboardInterrupt:
            print("\n[status] Exiting.")
            break


if __name__ == "__main__":
    main()
