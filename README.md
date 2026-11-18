# ClipForge — AI YouTube Clipper

A Render-ready MVP: YouTube URL or direct video upload → Gemini video analysis → at least 3 highlight clips → FFmpeg 9:16 720×1280 MP4 with white/yellow captions.

## Deploy to Render
1. Create a GitHub repository and upload these files.
2. In Render, create a **Blueprint** from the repository (`render.yaml`).
3. Add `GEMINI_API_KEY` in the Render service environment.
4. Deploy and open the service URL.

The Render build installs FFmpeg. The app uses a background thread and a polling endpoint so the browser does not wait for the entire render request.

## Run locally
Requires Python 3.11+, FFmpeg and ffprobe installed and on PATH.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000

## Important MVP limitations
- Gemini video understanding returns approximate segment timestamps; verify clips before publishing.
- The initial implementation colors a whole caption segment yellow if it contains a model-selected highlight phrase; word-level karaoke timing is not included yet.
- Render free instances have limited CPU/RAM, ephemeral disk, and may sleep/restart. A restart loses in-memory job status and temporary output. For reliable production, use a paid worker and object storage.
- The configured Gemini model order is a best-effort free-tier preference, not a guarantee of free quota or availability. Check the current Gemini API pricing/model availability for your API key. The code falls back to the next model if the first fails.
- YouTube downloading is subject to YouTube terms and copyright. Use content you own or are authorized to download. Some YouTube URLs may not be downloadable; direct upload is available as a fallback.
- For large videos, request upload limits, Render request limits and service resources may need tuning.
