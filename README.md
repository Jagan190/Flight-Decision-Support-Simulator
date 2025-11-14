# Flight Decision Support Simulator

A Flask-based Flight Decision Support Simulator for demonstration and training purposes. The app:

* Evaluates simple aircraft/field performance (density altitude, runway suitability, fuel margin, risk score)
* Fetches METAR (multi-source fallback) or accepts manual METAR input
* Decodes METAR into an ATIS-style spoken format
* Generates audio briefings (gTTS) for the combined flight briefing and decoded METAR
* Provides a web UI for input and results

> **WARNING:** This is a demo/proof-of-concept. Do **not** use this tool for real flight planning. Always validate with official charts, manufacturer performance data, and certified flight planning tools.

---

## Repository layout

```
├── app.py               # Main Flask app
├── templates/
│   ├── index.html       # Input form
│   └── result.html      # Results page (assessment, METAR, audio)
├── static/              # Generated MP3 files (created at runtime)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Quick start

1. Clone the repository (or place files locally):

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
# venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) Add AVWX API key for improved METAR fallback:

```bash
# macOS / Linux
export AVWX_API_KEY="Token <your_token_here>"
# Windows (PowerShell)
# $env:AVWX_API_KEY="Token <your_token_here>"
```

5. Run the app:

```bash
python app.py
```

6. Open the app in a browser: `http://localhost:7860`

---

## Usage

* Enter origin and destination fields in the web form. Provide a Destination ICAO (e.g., `VOBG`) to fetch METAR automatically, or paste a raw METAR into the **Manual METAR** box (manual input takes priority).
* Submit the form to view:

  * Flight assessment and risk score
  * Raw METAR (if available)
  * Decoded ATIS-style METAR text
  * Audio players for: combined briefing, raw METAR audio, decoded METAR audio

**Example manual METAR**:

```
VOBG 142300Z 12007KT 9999 SCT020 29/18 Q1010 NOSIG
```

---

## Key functions / files

* `app.py`

  * `fetch_metar_for_icao()` — robust METAR fetcher (TGFTP → AviationWeather CGI → optional AVWX)
  * `simple_parse_metar()` — minimal parsing to auto-fill wind/visibility
  * `decode_metar_to_text()` — METAR → ATIS-style decoder (Option B)
  * `generate_audio_briefing()` — creates MP3 files using gTTS
  * `FlightDecisionSupport` — decision logic (density altitude, runway suitability, fuel margins, risk score)

* `templates/index.html` — user input form

* `templates/result.html` — results and audio players

---

## Environment variables

* `AVWX_API_KEY` (optional) — provides AVWX REST API fallback. Keep this **secret** and do not commit it to the repository.

---

## .gitignore (recommended)

Add a `.gitignore` at the repo root and include:

```
# Python
__pycache__/
*.pyc
venv/
.venv/

# IDEs
.vscode/
.idea/

# Generated audio
static/*.mp3

# OS
.DS_Store
Thumbs.db
```

---

## Troubleshooting

* **No audio:** Ensure `gTTS` is installed and the machine has internet access (gTTS uses Google TTS service).
* **METAR fetch fails:** The app tries multiple sources. If remote servers block requests, use the Manual METAR box or provide an `AVWX_API_KEY`.
* **Port conflict:** Change the port in `app.py` (`app.run(..., port=7860)`).

---





