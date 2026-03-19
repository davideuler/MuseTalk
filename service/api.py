"""
MuseTalk Lipsync Service API  (new version — transformers WhisperModel)
POST /lipsync/sync  — upload video + audio → lip-synced mp4
GET  /lipsync/health
GET  /lipsync/jobs/{job_id}
GET  /lipsync/result/{job_id}  — download result
"""
import os
import sys
import uuid
import shutil
import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

MUSETALK_ROOT = Path(os.getenv("MUSETALK_ROOT", "/app"))
FFMPEG_PATH   = os.getenv("FFMPEG_PATH", str(MUSETALK_ROOT / "ffmpeg-4.4-amd64-static"))
RESULTS_DIR   = Path(os.getenv("RESULTS_DIR", "/tmp/musetalk_results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure CUDA/cuDNN libraries are visible in subprocess environments
_CUDA_LIB_PATHS = [
    "/venv/lib/python3.10/site-packages/nvidia/cudnn/lib",
    "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib",
    "/usr/local/cuda/lib64",
    "/usr/local/cuda-12.1/lib64",
    "/usr/lib/x86_64-linux-gnu",
]
_existing = [p for p in _CUDA_LIB_PATHS if os.path.isdir(p)]
_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(_existing + ([_ld] if _ld else []))

# job store
_jobs: dict = {}

app = FastAPI(title="MuseTalk Lipsync Service", root_path="/lipsync")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "musetalk_root": str(MUSETALK_ROOT),
        "jobs_total": len(_jobs),
        "jobs_running": sum(1 for j in _jobs.values() if j["status"] == "running"),
    }


@app.post("/sync")
async def sync(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Input video (.mp4)"),
    audio: UploadFile = File(..., description="Input audio (.wav/.mp3)"),
    bbox_shift: int = Form(0, description="Bounding box shift (-10 to 10)"),
    use_float16: bool = Form(True, description="Use float16 for faster inference"),
):
    """Submit a lipsync job. Returns {job_id, status} immediately."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True)

    vsuffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    asuffix = Path(audio.filename or "input.wav").suffix or ".wav"
    video_path = job_dir / f"input{vsuffix}"
    audio_path = job_dir / f"input{asuffix}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    _jobs[job_id] = {"status": "queued", "created_at": time.time()}
    background_tasks.add_task(
        _run_musetalk, job_id, job_dir, video_path, audio_path, bbox_shift, use_float16
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/result/{job_id}")
def download_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "success":
        raise HTTPException(400, f"Job not ready: {job['status']}")
    output = Path(job["output"])
    if not output.exists():
        raise HTTPException(500, "Output file missing")
    return FileResponse(
        str(output), media_type="video/mp4",
        filename=f"lipsync_{job_id}.mp4",
    )


# ── background worker ─────────────────────────────────────────────────────────

async def _run_musetalk(
    job_id: str, job_dir: Path,
    video_path: Path, audio_path: Path,
    bbox_shift: int, use_float16: bool,
):
    _jobs[job_id]["status"] = "running"
    t0 = time.time()
    _jobs[job_id]["started_at"] = t0

    result_dir = job_dir / "output"
    result_dir.mkdir()

    # 1. Convert audio to 16kHz mono WAV (AudioProcessor requires 16kHz)
    audio_16k = job_dir / "input_16k.wav"
    ffmpeg_bin = str(Path(FFMPEG_PATH) / "ffmpeg") if Path(FFMPEG_PATH).is_dir() else "ffmpeg"
    conv = await asyncio.create_subprocess_exec(
        ffmpeg_bin, "-y", "-i", str(audio_path),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        str(audio_16k),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await conv.wait()
    if audio_16k.exists() and audio_16k.stat().st_size > 0:
        audio_path = audio_16k

    # 2. Build per-job inference config yaml
    cfg_path = job_dir / "inference.yaml"
    cfg_path.write_text(f"task_0:\n  video_path: \"{video_path}\"\n  audio_path: \"{audio_path}\"\n")

    # 3. Determine model paths
    musetalk_v15  = MUSETALK_ROOT / "models" / "musetalkV15"
    musetalk_v10  = MUSETALK_ROOT / "models" / "musetalk"
    if musetalk_v15.exists() and (musetalk_v15 / "unet.pth").exists():
        unet_model = str(musetalk_v15 / "unet.pth")
        unet_cfg   = str(musetalk_v15 / "musetalk.json")
    else:
        unet_model = str(musetalk_v10 / "pytorch_model.bin")
        unet_cfg   = str(musetalk_v10 / "musetalk.json")

    whisper_dir = str(MUSETALK_ROOT / "models" / "whisper")
    vae_type    = "sd-vae-ft-mse"

    cmd = [
        sys.executable, "scripts/inference.py",
        "--ffmpeg_path",      FFMPEG_PATH,
        "--whisper_dir",      whisper_dir,
        "--unet_model_path",  unet_model,
        "--unet_config",      unet_cfg,
        "--vae_type",         vae_type,
        "--inference_config", str(cfg_path),
        "--result_dir",       str(result_dir),
        "--bbox_shift",       str(bbox_shift),
        "--output_vid_name",  "output.mp4",
    ]
    if use_float16:
        cmd.append("--use_float16")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(MUSETALK_ROOT) + ":" + env.get("PYTHONPATH", "")
    env["PYTHONNOUSERSITE"] = "1"

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(MUSETALK_ROOT),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=1800)
        stdout_s = stdout.decode(errors="replace")
        stderr_s = stderr.decode(errors="replace")

        if proc.returncode != 0:
            raise RuntimeError(f"exit {proc.returncode}: {stderr_s[-800:]}")

        # Find output mp4
        mp4s = sorted(result_dir.rglob("*.mp4"),
                      key=lambda x: x.stat().st_mtime, reverse=True)
        if not mp4s:
            raise FileNotFoundError(
                f"No mp4 in {result_dir}. stdout={stdout_s[-200:]} stderr={stderr_s[-300:]}"
            )

        final = job_dir / "result.mp4"
        shutil.move(str(mp4s[0]), str(final))

        _jobs[job_id].update({
            "status": "success",
            "output": str(final),
            "elapsed_s": round(time.time() - t0, 1),
            "size_mb": round(final.stat().st_size / 1024 / 1024, 2),
        })

    except Exception as e:
        _jobs[job_id].update({
            "status": "failed",
            "error": str(e)[-600:],
            "elapsed_s": round(time.time() - t0, 1),
        })
