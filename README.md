# RadiologySTT

A lightweight, push-to-talk medical dictation tool for radiology practices.
Streams microphone audio to the **Google Cloud Speech-to-Text** API using the
`medical_dictation` enhanced model and pastes the transcribed text directly into
whichever reporting window is currently active.

---

## How it works

| Action | Result |
|---|---|
| Hold **F9** | High beep → microphone opens → audio streams to Google Speech |
| Speak | Finalised sentences are pasted at the cursor in real time |
| Release **F9** | Low beep → stream closes |
| Press **Ctrl+C** in the terminal | Graceful exit |

---

## Prerequisites

### 1. Python

Python **3.9 or later** is required.  
Download from [python.org](https://www.python.org/downloads/).

### 2. Google Cloud project & credentials

#### a) Create / select a project

1. Go to [console.cloud.google.com](https://console.cloud.google.com).
2. Create a new project (e.g. `radiology-stt`) or select an existing one.

#### b) Enable the Speech-to-Text API

```
Navigation menu → APIs & Services → Library
→ Search "Cloud Speech-to-Text API" → Enable
```

#### c) Create a Service Account and download the JSON key

1. **IAM & Admin → Service Accounts → Create Service Account**
2. Give it a name (e.g. `radiology-stt-sa`) and grant the role
   **Cloud Speech Client** (or **Editor** for simplicity during development).
3. Click **Keys → Add Key → Create new key → JSON** → Download the file.
4. Save it somewhere safe, e.g.:
   - Windows: `C:\credentials\radiology-stt-sa.json`
   - macOS:   `/Users/yourname/.credentials/radiology-stt-sa.json`

#### d) Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable

**Windows (PowerShell — per session):**
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\credentials\radiology-stt-sa.json"
```

**Windows (permanent — System Environment Variables):**
```
Control Panel → System → Advanced system settings
→ Environment Variables → New (User variable)
  Variable name:  GOOGLE_APPLICATION_CREDENTIALS
  Variable value: C:\credentials\radiology-stt-sa.json
```

**macOS (per session):**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/Users/yourname/.credentials/radiology-stt-sa.json"
```

**macOS (permanent — add to `~/.zshrc` or `~/.bash_profile`):**
```bash
echo 'export GOOGLE_APPLICATION_CREDENTIALS="/Users/yourname/.credentials/radiology-stt-sa.json"' >> ~/.zshrc
source ~/.zshrc
```

---

## Installation

### Windows

> Run PowerShell as **Administrator** — the `keyboard` library requires elevated
> privileges to install a global hook.

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

> **PyAudio note:** If `pip install PyAudio` fails on Windows, install the
> pre-built wheel directly:
> ```powershell
> pip install pipwin
> pipwin install pyaudio
> ```

### macOS

```bash
# 1. Install PortAudio (PyAudio's native dependency)
brew install portaudio

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Accessibility permission:**  
> Go to **System Settings → Privacy & Security → Accessibility** and add your
> Terminal application (or whichever app you use to run the script).  
> Without this, `pynput` cannot intercept the global F9 keypress.

---

## Running the scripts

### Windows

```powershell
# Make sure your virtual environment is active and credentials are set
.venv\Scripts\Activate.ps1
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\credentials\radiology-stt-sa.json"

python rad_windows.py
```

### macOS

```bash
source .venv/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS="/Users/yourname/.credentials/radiology-stt-sa.json"

python rad_mac.py
```

---

## Philips SpeechMike (Windows)

The Windows script automatically scans all PyAudio input devices on startup and
selects one whose name contains `"philips"` or `"speechmike"` (case-insensitive).
If none is found it falls back to the system default microphone — no configuration
needed. The selected device index is printed to the console at launch.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `PermissionError` on Windows | Run PowerShell / Terminal as Administrator |
| F9 not detected on macOS | Grant Accessibility permission (see above) |
| `PyAudio` install fails on Windows | Use `pipwin install pyaudio` |
| `PyAudio` install fails on macOS | `brew install portaudio` first |
| `INVALID_ARGUMENT` from Google | Check that `medical_dictation` model is available in your region; try `latest_long` as a fallback |
| Google API 5-minute limit | Each F9 press starts a new stream — just release and re-press F9 |
| Text pastes in wrong window | Click into your reporting window before pressing F9 |
| Clipboard briefly replaced | Normal — the script restores your clipboard after each paste |

---

## Architecture overview

```
F9 held down
    │
    ▼
MicrophoneStream (PyAudio, 16 kHz, 100 ms chunks)
    │  PCM bytes via queue
    ▼
generator() ──► StreamingRecognizeRequest stream
    │
    ▼
Google Cloud Speech-to-Text
  model = medical_dictation
  use_enhanced = True
    │  is_final responses only
    ▼
type_text()  (clipboard paste → Ctrl/Cmd+V)
    │
    ▼
Active radiology reporting window
```

---

## License

For internal/personal use within your practice. Review Google Cloud Speech-to-Text
[pricing](https://cloud.google.com/speech-to-text/pricing) before deploying.
