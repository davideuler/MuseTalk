# MuseTalk Lipsync Service — 部署与调用文档

## 目录

- [项目概述](#项目概述)
- [环境要求](#环境要求)
- [模型准备](#模型准备)
- [Docker 部署](#docker-部署)
- [nginx 反向代理](#nginx-反向代理)
- [API 接口文档](#api-接口文档)
- [Web 页面使用](#web-页面使用)
- [故障排查](#故障排查)

---

## 项目概述

本服务将 MuseTalk 口型同步功能封装为 REST API，提供：

- 上传任意视频 + 音频 → AI 口型对齐 → 输出 mp4
- 异步任务队列（提交 → 轮询 → 下载）
- Web 体验页面
- Docker GPU 容器部署

**技术栈：** FastAPI · PyTorch 2.3.1+cu121 · mmcv 2.2.0 · mmpose 1.3.2 · RTX 4090

---

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA GPU，显存 ≥ 10GB（推荐 RTX 3090/4090） |
| CUDA | 12.1 |
| Docker | ≥ 24.0，已安装 `nvidia-container-toolkit` |
| 磁盘 | 模型目录约 10GB |

### 确认 nvidia-container-toolkit 已安装

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

---

## 模型准备

所有模型下载到 `/mnt/nvme/models/MuseTalk/models`（可按实际路径调整）：

```bash
cd /mnt/nvme/models/MuseTalk
bash download_weights.sh
```

脚本会自动下载：

| 模型 | 路径 | 大小 |
|------|------|------|
| MuseTalk V1.5 UNet | `models/musetalkV15/unet.pth` | 3.2 GB |
| MuseTalk V1.0 | `models/musetalk/pytorch_model.bin` | 500 MB |
| Whisper Tiny | `models/whisper/` | 145 MB |
| SD VAE | `models/sd-vae-ft-mse/` | 320 MB |
| DWPose | `models/dwpose/dw-ll_ucoco_384.pth` | 389 MB |
| RTMPose backbone | `models/dwpose/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth` | 255 MB |
| Face Parse | `models/face-parse-bisent/` | 96 MB |

> **注意：** `models/sd-vae` 需软链到 `sd-vae-ft-mse`：
> ```bash
> ln -sfn sd-vae-ft-mse models/sd-vae
> ```

### 预下载 s3fd 人脸检测模型

首次运行时 `face_detection` 库会自动从 `adrianbulat.com` 下载 s3fd 权重（86MB），网速慢时耗时极长。建议提前下载到 torch hub 缓存：

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
wget -c "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" \
     -O ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
```

---

## Docker 部署

### 1. 克隆代码

```bash
git clone https://github.com/davideuler/MuseTalk.git
cd MuseTalk
```

### 2. 构建镜像

```bash
cd /mnt/nvme/clawspace/musetalk-service   # 或将 docker-compose.service.yml 复制到此目录
docker compose -f docker-compose.service.yml build
```

> 首次构建约需 20-40 分钟（主要是 PyTorch 下载）。

### 3. 启动服务

```bash
docker compose -f docker-compose.service.yml up -d
```

### 4. 验证

```bash
curl http://localhost:8400/lipsync/health
# {"status":"ok","musetalk_root":"/app","jobs_total":0,"jobs_running":0}
```

### docker-compose.service.yml 关键配置说明

```yaml
services:
  musetalk-service:
    runtime: nvidia                        # 必须：启用 GPU
    volumes:
      - /mnt/nvme/models/MuseTalk:/app:ro  # MuseTalk 代码+模型（只读挂载）
      - musetalk_results:/tmp/musetalk_results  # 结果存储（可写）
      - /home/david/.cache/torch:/root/.cache/torch:ro  # torch hub 缓存（含 s3fd）
    ports:
      - "8400:8400"                        # 服务端口
    environment:
      - MUSETALK_ROOT=/app
      - FFMPEG_PATH=/app/ffmpeg-4.4-amd64-static
```

> 如果 MuseTalk 代码目录不在 `/mnt/nvme/models/MuseTalk`，修改 `volumes` 中的宿主机路径即可。

---

## nginx 反向代理

在已有 nginx `server {}` 块中添加以下配置，通过 `/lipsync/` 前缀代理服务：

```nginx
upstream musetalk_service {
    server 172.17.0.1:8400;   # Docker bridge 网关 IP，用于从容器内访问宿主机端口
}

location /lipsync/ {
    proxy_pass         http://musetalk_service/lipsync/;
    proxy_http_version 1.1;
    proxy_set_header   Host $host;
    proxy_set_header   X-Real-IP $remote_addr;
    proxy_read_timeout 1800s;      # 推理最长 30 分钟
    client_max_body_size 500m;     # 支持大视频上传
}
```

> **注意：** upstream 必须使用 Docker bridge 网关 IP（通常为 `172.17.0.1`），不能用 `127.0.0.1`（nginx 容器内的 localhost 不等于宿主机）。

重载 nginx：

```bash
nginx -t && nginx -s reload
# 或 Docker 容器内
docker exec <nginx_container> nginx -t && docker exec <nginx_container> nginx -s reload
```

---

## API 接口文档

交互式文档：`http://<host>/lipsync/docs`

### POST /lipsync/sync — 提交口型同步任务

**请求（multipart/form-data）：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `video` | file | ✅ | 输入视频（mp4/mov/avi） |
| `audio` | file | ✅ | 输入音频（wav/mp3/m4a） |
| `bbox_shift` | int | ❌ | 人脸边框偏移，范围 -10～10，默认 0 |
| `use_float16` | bool | ❌ | 是否使用 FP16 加速，默认 true |

**响应：**

```json
{"job_id": "959e1d34", "status": "queued"}
```

**curl 示例：**

```bash
curl -X POST http://10.0.0.145/lipsync/sync \
  -F "video=@/path/to/video.mp4" \
  -F "audio=@/path/to/audio.wav" \
  -F "bbox_shift=0" \
  -F "use_float16=true"
```

---

### GET /lipsync/jobs/{job_id} — 查询任务状态

```bash
curl http://10.0.0.145/lipsync/jobs/959e1d34
```

**响应（进行中）：**

```json
{"status": "running", "created_at": 1710000000.0, "started_at": 1710000001.0}
```

**响应（成功）：**

```json
{
  "status": "success",
  "output": "/tmp/musetalk_results/959e1d34/result.mp4",
  "elapsed_s": 58.5,
  "size_mb": 2.2
}
```

**响应（失败）：**

```json
{"status": "failed", "error": "...", "elapsed_s": 5.1}
```

---

### GET /lipsync/result/{job_id} — 下载结果视频

```bash
curl -O http://10.0.0.145/lipsync/result/959e1d34
# 或
wget -O output.mp4 http://10.0.0.145/lipsync/result/959e1d34
```

返回 `video/mp4` 文件流。

---

### GET /lipsync/health — 健康检查

```bash
curl http://10.0.0.145/lipsync/health
```

```json
{
  "status": "ok",
  "musetalk_root": "/app",
  "jobs_total": 10,
  "jobs_running": 1
}
```

---

## Python SDK 调用示例

```python
import requests, time

BASE = "http://10.0.0.145/lipsync"

# 1. 提交任务
with open("input.mp4", "rb") as vf, open("input.wav", "rb") as af:
    r = requests.post(f"{BASE}/sync", files={
        "video": ("input.mp4", vf, "video/mp4"),
        "audio": ("input.wav", af, "audio/wav"),
    }, data={"bbox_shift": 0, "use_float16": True})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"Job submitted: {job_id}")

# 2. 轮询状态
while True:
    time.sleep(5)
    status = requests.get(f"{BASE}/jobs/{job_id}").json()
    print(f"Status: {status['status']}")
    if status["status"] == "success":
        break
    if status["status"] == "failed":
        raise RuntimeError(status.get("error"))

# 3. 下载结果
r = requests.get(f"{BASE}/result/{job_id}", stream=True)
with open("output.mp4", "wb") as f:
    for chunk in r.iter_content(65536):
        f.write(chunk)
print(f"Saved output.mp4 ({status['size_mb']} MB, {status['elapsed_s']}s)")
```

---

## Web 页面使用

访问 `http://<host>/lipsync/` 即可打开体验页面：

1. 拖拽或点击上传视频文件（支持 mp4/mov/avi）
2. 拖拽或点击上传音频文件（支持 wav/mp3/m4a）
3. 可选调整参数：Float16 加速、BBox 偏移
4. 点击「🚀 开始生成」，进度条实时显示推理状态
5. 完成后直接在页面播放结果，点击「⬇️ 下载视频」保存

典型耗时：约 **60 秒**（RTX 4090，22 秒 1080p 视频）

---

## 故障排查

### 502 Bad Gateway

nginx upstream 使用了 `127.0.0.1:8400`。若 nginx 运行在 Docker 容器内，需改为 `172.17.0.1:8400`（宿主机 Docker bridge IP）。

```bash
# 确认 musetalk-service 端口绑定
ss -tlnp | grep 8400
# 应显示 0.0.0.0:8400，不是 127.0.0.1:8400
```

### ModuleNotFoundError: No module named 'munkres'

mmpose 运行时依赖未完整安装，执行：

```bash
docker exec musetalk-service pip install munkres chumpy json-tricks terminaltables
docker restart musetalk-service
```

### 推理卡住（长时间 running，无输出文件）

通常是运行时尝试下载模型导致（网络慢）：

**s3fd（人脸检测，86MB）：**
```bash
# 提前下载到宿主机 torch cache，然后 docker cp 到容器
mkdir -p ~/.cache/torch/hub/checkpoints
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" \
     -O ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
docker exec musetalk-service mkdir -p /root/.cache/torch/hub/checkpoints
docker cp ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth \
          musetalk-service:/root/.cache/torch/hub/checkpoints/
```

**RTMPose backbone（255MB）：**
```bash
wget "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth" \
     -O models/dwpose/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth
```
`musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py` 中的 checkpoint URL 已改为本地路径，无需额外配置。

### CUDA Out of Memory

- 确认其他 GPU 进程已退出：`nvidia-smi`
- 减小 `batch_size`（在 inference.py `--batch_size` 参数）
- 使用 `--use_float16` 节省显存

### OSError: whisper pytorch_model.bin not found

whisper 目录中的文件可能是 HuggingFace 缓存的**符号链接**，Docker 容器内无法访问源路径。解决：

```bash
cd models/whisper
for f in pytorch_model.bin model.safetensors; do
  if [ -L "$f" ]; then
    REAL=$(readlink -f "$f")
    cp "$REAL" "${f}.real" && mv "${f}.real" "$f"
    echo "Fixed: $f"
  fi
done
```

---

## 服务目录结构

```
MuseTalk/
├── scripts/inference.py          # 推理入口
├── musetalk/                     # 核心模型代码
├── models/                       # 模型权重（需手动下载）
│   ├── musetalkV15/unet.pth      # V1.5 UNet (3.2GB)
│   ├── musetalk/pytorch_model.bin # V1.0
│   ├── whisper/                  # Whisper Tiny
│   ├── sd-vae -> sd-vae-ft-mse   # VAE (软链)
│   ├── sd-vae-ft-mse/
│   ├── dwpose/                   # 姿态估计模型
│   └── face-parse-bisent/        # 人脸解析模型
├── service/                      # FastAPI 服务
│   ├── api.py                    # API 入口
│   └── templates/index.html      # Web 页面
├── Dockerfile.service            # Docker 镜像构建文件
├── docker-compose.service.yml    # Docker Compose 配置
├── test_service.py               # E2E 测试脚本
└── download_weights.sh           # 模型下载脚本
```
