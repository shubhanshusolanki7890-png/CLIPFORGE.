import os, json, uuid, shutil, subprocess, threading, time, re
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.getenv("WORK_DIR", "/tmp/clipforge"))
WORK.mkdir(parents=True, exist_ok=True)
JOBS = {}
app = FastAPI(title="AI YouTube Clipper")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]

def run(cmd, timeout=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-1800:])
    return p.stdout

def get_video(source: str, dest: Path):
    # Accept direct media URLs or YouTube URLs. yt-dlp is used only for content
    # the user owns or is authorized to download.
    if re.match(r"^https?://", source):
        try:
            run(["yt-dlp", "--no-playlist", "-f", "bv*[height<=720]+ba/b[height<=720]/b",
                 "--merge-output-format", "mp4", "-o", str(dest), source], timeout=900)
        except Exception as e:
            raise RuntimeError("Could not fetch this URL. Use a video you own or have permission to download. " + str(e))
    else:
        raise RuntimeError("Invalid video URL.")
    if not dest.exists():
        # yt-dlp may append extension
        candidates = list(dest.parent.glob(dest.stem + "*"))
        if candidates: candidates[0].rename(dest)
    if not dest.exists():
        raise RuntimeError("Video download did not produce an MP4.")

def probe_duration(path: Path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(out.strip())

def gemini_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key: raise RuntimeError("Set GEMINI_API_KEY in Render environment variables.")
    from google import genai
    return genai.Client(api_key=key)

def upload_gemini(path: Path, client):
    f = client.files.upload(file=str(path))
    # Wait until the uploaded video is ready for inference.
    for _ in range(90):
        f = client.files.get(name=f.name)
        state = getattr(f, "state", None)
        state_name = getattr(state, "name", str(state))
        if state_name == "ACTIVE": return f
        if state_name == "FAILED": raise RuntimeError("Gemini could not process the uploaded video.")
        time.sleep(2)
    raise RuntimeError("Timed out waiting for Gemini video processing.")

def ask_gemini(path: Path, duration: float):
    from google.genai import types
    client = gemini_client()
    uploaded = upload_gemini(path, client)
    prompt = f"""
Analyze this video (duration about {duration:.1f} seconds). Return ONLY valid JSON:
{{
 "transcript": [{{"start":0.0,"end":2.0,"text":"spoken words"}}],
 "clips": [
  {{"start":10.0,"end":42.0,"title":"short title","reason":"why this moment is engaging",
    "highlights":["word or short phrase to color yellow"]}}
 ]
}}
Requirements:
- Produce a useful timestamped transcript covering the whole video. Transcript segments should be short (roughly 2-6 seconds) and timestamps must be seconds from video start.
- Select at least 3 distinct compelling short-form moments, ideally 20-55 seconds each, no overlap, with a strong hook and a satisfying payoff.
- Never invent speech. Use exact wording from the video.
- Clip start/end must be within 0 and {duration:.1f}; end must be after start.
- Include 1-4 exact highlight words/phrases per clip. Return JSON only.
"""
    last = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[uploaded, prompt],
                config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json")
            )
            data = json.loads(response.text)
            data["model_used"] = model
            return data
        except Exception as e:
            last = e
            # Try next model for unavailable/quota/transient failures.
            continue
    raise RuntimeError("All configured Gemini models failed. Check API access/quota. " + str(last))

def escape_text(s):
    return s.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\\'").replace("%", r"\%").replace(",", r"\,")

def render_clip(video: Path, out: Path, clip: dict, transcript: list, index: int):
    start, end = max(0, float(clip["start"])), float(clip["end"])
    if end <= start: raise RuntimeError("Gemini returned an invalid clip range.")
    # Build ASS subtitles: white default, yellow for words/phrases the model flags.
    # Gemini transcript timing is segment-level; each segment is shown as a short caption.
    ass = out.with_suffix(".ass")
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
ScaledBorderAndShadow: yes
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,1,0,1,3,1,2,40,40,115,1
Style: Highlight,Arial,52,&H0000D7FF,&H0000D7FF,&H00101010,&H80000000,1,0,1,3,1,2,40,40,115,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    rows = [header]
    highlights = [str(x).lower() for x in clip.get("highlights", [])]
    for seg in transcript:
        try:
            a, b = float(seg["start"]), float(seg["end"])
            txt = str(seg["text"]).strip()
        except Exception: continue
        a2, b2 = max(a, start), min(b, end)
        if b2 <= a2 or not txt: continue
        # Whole caption turns yellow if it contains a highlighted phrase.
        style = "Highlight" if any(h and h in txt.lower() for h in highlights) else "Default"
        rel_a, rel_b = a2-start, b2-start
        def ass_time(v):
            cs = int(v*100)
            return f"{cs//360000}:{(cs//6000)%60:02}:{(cs//100)%60:02}.{cs%100:02}"
        safe = txt.replace("{", r"\{").replace("}", r"\}").replace("\n", " ")
        rows.append(f"Dialogue: 0,{ass_time(rel_a)},{ass_time(rel_b)},{style},,0,0,0,,{safe}\n")
    ass.write_text("".join(rows), encoding="utf-8")
    # Crop to vertical 9:16, fit 720x1280; output 720p vertical.
    vf = f"scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,setsar=1,subtitles={ass.as_posix()}"
    run(["ffmpeg", "-y", "-ss", str(start), "-i", str(video), "-t", str(end-start),
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)], timeout=1200)

def process_job(job_id: str, input_path: Path):
    job = JOBS[job_id]
    try:
        job.update(status="processing", progress="Checking video")
        duration = probe_duration(input_path)
        job["duration"] = duration
        job["progress"] = "Analyzing video with Gemini"
        analysis = ask_gemini(input_path, duration)
        clips = analysis.get("clips", [])[:6]
        transcript = analysis.get("transcript", [])
        if len(clips) < 3: raise RuntimeError("AI found fewer than 3 valid moments. Try a longer video.")
        result_dir = WORK / job_id
        result_dir.mkdir(exist_ok=True)
        outputs = []
        for i, clip in enumerate(clips, 1):
            job["progress"] = f"Rendering clip {i}/{len(clips)}"
            out = result_dir / f"clip-{i}.mp4"
            render_clip(input_path, out, clip, transcript, i)
            outputs.append({"name": clip.get("title", f"Clip {i}"), "reason": clip.get("reason", ""),
                            "url": f"/download/{job_id}/clip-{i}.mp4"})
        job.update(status="done", progress="Complete", clips=outputs, model=analysis.get("model_used"))
    except Exception as e:
        job.update(status="error", progress="Failed", error=str(e)[:1500])
    finally:
        try: input_path.unlink(missing_ok=True)
        except Exception: pass

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")

@app.post("/api/jobs")
async def create_job(file: Optional[UploadFile] = File(None), url: str = Form("")):
    if (file is None) == (not url.strip()):
        raise HTTPException(400, "Provide either a video file or a video URL.")
    job_id = uuid.uuid4().hex[:10]
    job_dir = WORK / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "source.mp4"
    if file:
        size = 0
        with path.open("wb") as f:
            while chunk := await file.read(1024*1024):
                size += len(chunk)
                if size > 1024*1024*1024:
                    raise HTTPException(413, "Upload limit is 1 GB.")
                f.write(chunk)
    else:
        try: get_video(url.strip(), path)
        except Exception as e: raise HTTPException(400, str(e))
    JOBS[job_id] = {"status":"queued", "progress":"Queued", "clips":[]}
    threading.Thread(target=process_job, args=(job_id, path), daemon=True).start()
    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS: raise HTTPException(404, "Job not found (Render free filesystem/process may have restarted).")
    return JOBS[job_id]

@app.get("/download/{job_id}/{filename}")
def download(job_id: str, filename: str):
    path = WORK / job_id / filename
    if not path.exists() or path.suffix != ".mp4": raise HTTPException(404, "Clip not found.")
    return FileResponse(path, media_type="video/mp4", filename=filename)
