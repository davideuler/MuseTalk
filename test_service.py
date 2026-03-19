#!/usr/bin/env python3
"""
Quick functional test for the MuseTalk lipsync service.
Usage:  python test_service.py [--base-url http://127.0.0.1:8400/lipsync]
"""
import sys, time, requests, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8400/lipsync")
    ap.add_argument("--video", default="/mnt/nvme/models/MuseTalk/data/video/sun.mp4")
    ap.add_argument("--audio", default="/mnt/nvme/models/MuseTalk/data/audio/sun.wav")
    ap.add_argument("--version", default="v1.0")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    # 1. Health
    print(f"[1] GET {base}/health ...")
    r = requests.get(f"{base}/health", timeout=10)
    r.raise_for_status()
    print(f"    {r.json()}")

    # 2. Submit job
    print(f"[2] POST {base}/sync ...")
    with open(args.video, "rb") as vf, open(args.audio, "rb") as af:
        r = requests.post(f"{base}/sync", files={
            "video": ("input.mp4", vf, "video/mp4"),
            "audio": ("input.wav", af, "audio/wav"),
        }, data={"version": args.version}, timeout=30)
    r.raise_for_status()
    job = r.json()
    job_id = job["job_id"]
    print(f"    job_id={job_id}")

    # 3. Poll
    print(f"[3] Polling job {job_id} ...")
    t0 = time.time()
    while True:
        time.sleep(5)
        r = requests.get(f"{base}/jobs/{job_id}", timeout=10)
        r.raise_for_status()
        status = r.json()
        elapsed = round(time.time() - t0)
        print(f"    {elapsed}s — {status['status']}")
        if status["status"] in ("success", "failed"):
            break
        if time.time() - t0 > args.timeout:
            print("TIMEOUT")
            sys.exit(2)

    if status["status"] != "success":
        print(f"FAILED: {status.get('error')}")
        sys.exit(1)

    print(f"    elapsed={status.get('elapsed_s')}s")

    # 4. Download
    out_path = f"/tmp/lipsync_result_{job_id}.mp4"
    print(f"[4] Downloading result → {out_path} ...")
    r = requests.get(f"{base}/result/{job_id}", timeout=30, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"    saved {size_mb:.1f} MB")
    if size_mb < 0.1:
        print("ERROR: output too small!")
        sys.exit(1)

    print("\n✅ ALL TESTS PASSED")

if __name__ == "__main__":
    main()
