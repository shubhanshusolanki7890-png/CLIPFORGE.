"""ClipForge FastAPI backend. Uvicorn expects the object named `app`."""
import os, json, re, shutil, subprocess, threading, time, uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
WORK = Path(os.getenv("WORK_DIR", "/tmp/clipforge"))
WORK.mkdir(parents=True, exist_ok=True)
JOBS = {}

# Required by: uvicorn app.main:app
app = FastAPI(title="ClipForge AI YouTube Clipper")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]

def run(args, timeout=900):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout or "Command failed")[-2000:])
    return p.stdout.strip()

def require_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg/ffprobe not found. Deploy with a Docker image that installs FFmpeg.")

def duration_of(path):
    require_ffmpeg()
    return float(run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)],60))

def get_video(url, dest):
    run(["yt-dlp","--no-playlist","-f","bv*[height<=720]+ba/b[height<=720]/b","--merge-output-format","mp4","-o",str(dest),url],1200)
    if not dest.exists():
        found=list(dest.parent.glob(dest.stem+"*"))
        if found: found[0].replace(dest)
    if not dest.exists(): raise RuntimeError("Could not download video. Try uploading the video file instead.")

def gemini_analysis(path, duration):
    key=os.getenv("GEMINI_API_KEY")
    if not key: raise RuntimeError("GEMINI_API_KEY is missing from Render Environment.")
    from google import genai
    from google.genai import types
    client=genai.Client(api_key=key)
    f=client.files.upload(file=str(path))
    for _ in range(90):
        f=client.files.get(name=f.name)
        state=getattr(getattr(f,"state",None),"name",str(getattr(f,"state",None)))
        if state=="ACTIVE": break
        if state=="FAILED": raise RuntimeError("Gemini video processing failed.")
        time.sleep(2)
    else: raise RuntimeError("Timed out waiting for Gemini video processing.")
    prompt=f"""Analyze this video ({duration:.1f} seconds). Return JSON only:
{{"transcript":[{{"start":0.0,"end":3.0,"text":"exact spoken words"}}],"clips":[{{"start":10.0,"end":42.0,"title":"title","reason":"why interesting","highlights":["word","phrase"]}}]}}
Create a timestamped transcript across the video. Select at least 3 distinct engaging moments, preferably 20-55 seconds, with hook and payoff. Do not invent speech. Timestamps are seconds from video start and must be within 0 and {duration:.1f}. Give 1-4 exact words/phrases to highlight yellow."""
    error=None
    for model in MODELS:
        try:
            response=client.models.generate_content(model=model,contents=[f,prompt],config=types.GenerateContentConfig(temperature=0.2,response_mime_type="application/json"))
            data=json.loads(response.text); data["model_used"]=model; return data
        except Exception as e: error=e
    raise RuntimeError(f"Gemini models failed; check API key/quota. Last error: {error}")

def ass_time(sec):
    n=max(0,int(sec*100)); return f"{n//360000}:{(n//6000)%60:02}:{(n//100)%60:02}.{n%100:02}"

def render_clip(video, output, clip, transcript):
    start=max(0,float(clip["start"])); end=float(clip["end"])
    if end<=start: raise RuntimeError("Invalid clip timestamps from Gemini.")
    ass=output.with_suffix(".ass")
    lines=["""[Script Info]\nScriptType: v4.00+\nPlayResX: 720\nPlayResY: 1280\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Normal,Arial,48,&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,1,0,1,3,1,2,36,36,120,1\nStyle: Highlight,Arial,50,&H0000D7FF,&H0000D7FF,&H00101010,&H80000000,1,0,1,3,1,2,36,36,120,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""]
    terms=[str(x).lower() for x in clip.get("highlights",[])]
    for seg in transcript:
        try: a,b,txt=float(seg["start"]),float(seg["end"]),str(seg["text"]).strip()
        except Exception: continue
        a2,b2=max(a,start),min(b,end)
        if b2<=a2 or not txt: continue
        style="Highlight" if any(t and t in txt.lower() for t in terms) else "Normal"
        txt=txt.replace("\\",r"\\").replace("{",r"\\{").replace("}",r"\\}").replace("\n"," ")
        lines.append(f"Dialogue: 0,{ass_time(a2-start)},{ass_time(b2-start)},{style},,0,0,0,,{txt}\n")
    ass.write_text("".join(lines),encoding="utf-8")
    vf=f"scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,setsar=1,subtitles={ass.as_posix()}"
    run(["ffmpeg","-y","-ss",str(start),"-i",str(video),"-t",str(end-start),"-vf",vf,"-c:v","libx264","-preset","veryfast","-crf","25","-c:a","aac","-b:a","128k","-movflags","+faststart",str(output)],1200)

def worker(job_id, video):
    job=JOBS[job_id]
    try:
        job.update(status="processing",progress="Checking video")
        dur=duration_of(video); job["progress"]="Analyzing with Gemini"
        data=gemini_analysis(video,dur); transcript=data.get("transcript",[]); clips=[]
        for c in data.get("clips",[]):
            try:
                if 0<=float(c["start"])<float(c["end"])<=dur: clips.append(c)
            except Exception: pass
        clips=clips[:5]
        if len(clips)<3: raise RuntimeError("Gemini found fewer than 3 valid clips. Try another video.")
        folder=WORK/job_id; folder.mkdir(exist_ok=True); result=[]
        for i,c in enumerate(clips,1):
            job["progress"]=f"Rendering clip {i}/{len(clips)}"
            name=f"clip-{i}.mp4"; render_clip(video,folder/name,c,transcript)
            result.append({"name":c.get("title",f"Clip {i}"),"reason":c.get("reason",""),"url":f"/download/{job_id}/{name}"})
        job.update(status="done",progress="Clips ready",clips=result,model=data.get("model_used"))
    except Exception as e: job.update(status="error",progress="Failed",error=str(e))
    finally:
        try: video.unlink(missing_ok=True)
        except Exception: pass

@app.get("/",response_class=HTMLResponse)
def home():
    index=STATIC_DIR/"index.html"
    return HTMLResponse(index.read_text(encoding="utf-8") if index.exists() else "<h1>ClipForge API is running</h1>")

@app.get("/health")
def health(): return {"ok":True,"service":"ClipForge"}

@app.post("/api/jobs")
async def create_job(file:Optional[UploadFile]=File(default=None),url:str=Form(default="")):
    if (file is None and not url.strip()) or (file is not None and url.strip()): raise HTTPException(400,"Provide either a video upload or a YouTube URL.")
    jid=uuid.uuid4().hex[:12]; folder=WORK/jid; folder.mkdir(parents=True,exist_ok=True); path=folder/"source.mp4"
    if file:
        size=0
        try:
            with path.open("wb") as out:
                while True:
                    chunk=await file.read(1024*1024)
                    if not chunk: break
                    size+=len(chunk)
                    if size>1024*1024*1024: raise HTTPException(413,"Maximum upload size is 1 GB.")
                    out.write(chunk)
        finally: await file.close()
    else:
        if not re.match(r"^https?://",url.strip(),re.I): raise HTTPException(400,"Invalid URL.")
        try: get_video(url.strip(),path)
        except Exception as e: raise HTTPException(400,str(e))
    JOBS[jid]={"status":"queued","progress":"Queued","clips":[]}
    threading.Thread(target=worker,args=(jid,path),daemon=True).start()
    return {"job_id":jid}

@app.get("/api/jobs/{job_id}")
def job_status(job_id:str):
    if job_id not in JOBS: raise HTTPException(404,"Job not found; service may have restarted.")
    return JOBS[job_id]

@app.get("/download/{job_id}/{filename}")
def download(job_id:str,filename:str):
    if not re.fullmatch(r"clip-\d+\.mp4",filename): raise HTTPException(400,"Invalid filename.")
    path=WORK/job_id/filename
    if not path.is_file(): raise HTTPException(404,"Clip not found.")
    return FileResponse(str(path),media_type="video/mp4",filename=filename)
