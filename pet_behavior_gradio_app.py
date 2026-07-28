from __future__ import annotations

import argparse
import atexit
import base64
import json
import logging
import mimetypes
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections import deque
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from ultralytics import YOLO

try:
    import serial
    import serial.tools.list_ports

    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False


APP_TITLE = "智能宠物行为识别演示系统"
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHT_CANDIDATES = [
    ROOT_DIR / "runs" / "train" / "cat_behavior_yolo26n" / "weights" / "best.pt",
    ROOT_DIR / "runs" / "train" / "yolo-GDL" / "weights" / "best.pt",
]
DEFAULT_WEIGHT = next(
    (weight_path for weight_path in DEFAULT_WEIGHT_CANDIDATES if weight_path.exists()),
    DEFAULT_WEIGHT_CANDIDATES[0],
)
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_IMAGE_MODEL = "qwen3.5-flash"
DEFAULT_AUDIO_ASR_MODEL = "不使用ASR"
DEFAULT_AUDIO_REASONING_MODEL = "qwen3.5-omni-plus"
DEFAULT_IMAGE_PROMPT = (
    "你是宠物行为分析助手。请结合图片内容和YOLO检测结果，判断猫咪当前行为，"
    "说明置信依据、可能的健康风险，并给出主人建议。请使用中文分点输出。"
)
DEFAULT_AUDIO_PROMPT = (
    "你是一个宠物声音行为分析助手。请直接分析音频中的猫叫声，不要把猫叫当成人类语音转写。"
    "只能从以下标签中选择一个最可能的主标签："
    "hungry（饥饿/要食物）、attention（求关注/撒娇）、fear（害怕/紧张）、"
    "pain（疼痛/不适）、warning（生气/警告）、fighting（打架/攻击/激烈对峙）、mating（发情）、"
    "relaxed（放松/普通叫声）、unknown（无法判断）。"
    "判定规则：如果听到尖锐尖叫、低吼、哈气、咆哮、突然爆发、两只猫对峙或连续冲突声，优先判为fighting或warning，不要判为mating。"
    "只有在叫声持续拖长、反复求偶式嚎叫、没有明显攻击/恐惧/冲突特征时，才可判为mating。"
    "如果音频像猫咪吵架、抢地盘、互相威胁或攻击，请判为fighting。"
    "请只输出JSON，不要输出其他内容，格式为："
    '{"emotion":"hungry|attention|fear|pain|warning|fighting|mating|relaxed|unknown",'
    '"confidence":0.0,"urgency":1,"reason":"一句话说明判断依据",'
    '"suggestion":"一句话给主人建议"}'
)
MAX_LOG_LINES = 200
QWEN_IMAGE_MAX_SIDE = 1024
QWEN_IMAGE_JPEG_QUALITY = 82
CLIENT_DATA_DIR = ROOT_DIR / ".gradio" / "client_data"
QWEN_TEMP_DIR = ROOT_DIR / ".gradio" / "qwen_temp"
MAX_AUDIO_AGE_DAYS = 7


def cleanup_old_temp_files() -> None:
    for d in [QWEN_TEMP_DIR, CLIENT_DATA_DIR]:
        if not d.exists():
            continue
        now = time.time()
        removed = 0
        for p in d.rglob("*"):
            if p.is_file() and (now - p.stat().st_mtime) > MAX_AUDIO_AGE_DAYS * 86400:
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
        # remove empty dirs
        for p in sorted(d.rglob("*"), key=lambda x: len(str(x)), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                try:
                    p.rmdir()
                except OSError:
                    pass
        if removed:
            push_log(f"已清理 {removed} 个过期临时文件（{MAX_AUDIO_AGE_DAYS}天以上）", level="info")


def build_logger() -> tuple[logging.Logger, deque[str], threading.Lock]:
    logger = logging.getLogger("pet_behavior_demo")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    log_buffer: deque[str] = deque(maxlen=MAX_LOG_LINES)
    log_lock = threading.Lock()
    return logger, log_buffer, log_lock


LOGGER, LOG_BUFFER, LOG_LOCK = build_logger()


def push_log(message: str, level: str = "info") -> None:
    text = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {level.upper()} | {message}"
    with LOG_LOCK:
        LOG_BUFFER.append(text)
    getattr(LOGGER, level, LOGGER.info)(message)


def get_recent_logs() -> str:
    with LOG_LOCK:
        return "\n".join(LOG_BUFFER) if LOG_BUFFER else "暂无日志。"


# 应用启动时清理一次过期临时文件
cleanup_old_temp_files()


MAX_AUDIO_RECORDS = 100


def new_client_state() -> dict:
    client_id = uuid.uuid4().hex[:12]
    return {
        "client_id": client_id,
        "logs": [],
        "audio_records": [],
    }


def client_log(state: dict | None, message: str, level: str = "info") -> dict:
    state = state or new_client_state()
    item = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {level.upper()} | {message}"
    logs = state.setdefault("logs", [])
    logs.append(item)
    del logs[:-MAX_LOG_LINES]
    push_log(f"[client:{state.get('client_id', 'unknown')}] {message}", level=level)
    return state


def get_client_logs(state: dict | None) -> str:
    if not state:
        return "暂无本会话日志。"
    logs = state.get("logs") or []
    return "\n".join(logs) if logs else "暂无本会话日志。"


def get_client_audio_dir(state: dict) -> Path:
    client_id = state.get("client_id") or uuid.uuid4().hex[:12]
    state["client_id"] = client_id
    audio_dir = CLIENT_DATA_DIR / client_id / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


def store_client_audio(audio_path: str, state: dict | None) -> tuple[str, dict]:
    state = state or new_client_state()
    source = Path(audio_path)
    suffix = source.suffix or ".wav"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = get_client_audio_dir(state) / f"{timestamp}_{uuid.uuid4().hex[:8]}{suffix}"
    shutil.copy2(source, target)
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": source.name,
        "path": str(target),
        "size_kb": round(target.stat().st_size / 1024, 2),
    }
    records = state.setdefault("audio_records", [])
    records.append(record)
    del records[:-MAX_AUDIO_RECORDS]
    return str(target), state


def format_client_audio_records(state: dict | None) -> list[list]:
    records = (state or {}).get("audio_records") or []
    return [
        [idx, item.get("time", ""), item.get("name", ""), item.get("size_kb", 0)]
        for idx, item in enumerate(records, start=1)
    ]


class ModelManager:
    def __init__(self, weight_path: Path):
        self.weight_path = weight_path
        self._model = None
        self._lock = threading.Lock()

    def get_model(self) -> YOLO:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if not self.weight_path.exists():
                        raise FileNotFoundError(
                            f"YOLO权重文件不存在: {self.weight_path}\n"
                            f"请先使用 train.py 训练模型，或通过 --weight 参数指定正确的权重路径。"
                        )
                    push_log(f"开始加载YOLO权重: {self.weight_path}")
                    self._model = YOLO(str(self.weight_path))
                    push_log("YOLO权重加载完成")
        return self._model


MODEL_MANAGER = ModelManager(DEFAULT_WEIGHT)


DEFAULT_LOCAL_PORT = 7861

FRP_SERVER_ADDR = os.environ.get("FRP_SERVER_ADDR", "127.0.0.1")
FRP_SERVER_PORT = int(os.environ.get("FRP_SERVER_PORT", "7000"))
FRP_AUTH_TOKEN = os.environ.get("FRP_AUTH_TOKEN", "")
FRP_REMOTE_PORT = int(os.environ.get("FRP_REMOTE_PORT", "8081"))
FRP_PROXY_NAME = "gradio-pet-behavior"
PUBLIC_DOMAIN = os.environ.get("FRP_PUBLIC_DOMAIN", "localhost")
PUBLIC_URL = f"http://{PUBLIC_DOMAIN}:{FRP_REMOTE_PORT}"
FRP_STARTUP_WAIT = 4  # frpc 启动后等待连通时间(秒)
FRP_MAX_RETRIES = 3  # 启动失败重试次数


def find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """Return the first available port starting from *start_port*."""
    for offset in range(max_attempts):
        port = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"无法找到可用端口 (尝试范围 {start_port}–{start_port + max_attempts - 1})")


def _find_frpc_exe() -> Path | None:
    preferred = ROOT_DIR / "frp_windows" / "frp_0.68.1_windows_amd64" / "frpc.exe"
    if preferred.exists():
        return preferred

    candidates = []
    for sub in ROOT_DIR.glob("frp*/**/frpc.exe"):
        candidates.append(sub)
    return candidates[0] if candidates else None


def _write_frpc_config(local_port: int) -> Path:
    config_dir = ROOT_DIR / ".gradio"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "frpc.toml"
    config_path.write_text(
        "\n".join(
            [
                f'serverAddr = "{FRP_SERVER_ADDR}"',
                f"serverPort = {FRP_SERVER_PORT}",
                "",
                'auth.method = "token"',
                f'auth.token = "{FRP_AUTH_TOKEN}"',
                "",
                "[[proxies]]",
                f'name = "{FRP_PROXY_NAME}"',
                'type = "tcp"',
                'localIP = "127.0.0.1"',
                f"localPort = {local_port}",
                f"remotePort = {FRP_REMOTE_PORT}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _get_frpc_pid_file() -> Path:
    return ROOT_DIR / ".gradio" / "frpc_pet_behavior.pid"


def _get_frpc_log_path() -> Path:
    return ROOT_DIR / ".gradio" / "frpc_pet_behavior.log"


def _stop_own_frpc() -> None:
    pid_file = _get_frpc_pid_file()
    if not pid_file.exists():
        return
    try:
        old_pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(old_pid), "/F"],
                capture_output=True,
                text=True,
                encoding="gbk",
                errors="ignore",
            )
        else:
            os.kill(old_pid, 9)
        push_log(f"已清理本 App 旧 frpc 进程 (PID={old_pid})")
    except (ProcessLookupError, OSError) as _exc:
        push_log(f"清理 frpc 进程 (PID={old_pid}) 时忽略: {_exc}", level="debug")
    except Exception:
        pass
    finally:
        pid_file.unlink(missing_ok=True)


def _check_frpc_connected(log_path: Path, wait_seconds: int = FRP_STARTUP_WAIT) -> bool:
    time.sleep(wait_seconds)
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        return "start proxy success" in text.lower() or "success" in text.lower()
    except OSError:
        return False


def _start_frpc_tunnel(local_port: int) -> subprocess.Popen | None:
    frpc_exe = _find_frpc_exe()
    if frpc_exe is None:
        push_log("未找到 frpc.exe，跳过公网映射", level="warning")
        return None

    config_path = _write_frpc_config(local_port)
    log_path = _get_frpc_log_path()
    _stop_own_frpc()

    for attempt in range(1, FRP_MAX_RETRIES + 1):
        if attempt > 1:
            push_log(f"frpc 第 {attempt}/{FRP_MAX_RETRIES} 次重试启动...")
            time.sleep(2)

        proc = None
        log_file = None
        try:
            log_file = open(str(log_path), "w", encoding="utf-8")
            proc = subprocess.Popen(
                [str(frpc_exe), "-c", str(config_path)],
                cwd=str(frpc_exe.parent),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        except Exception as exc:
            push_log(f"frpc 启动失败: {exc}", level="warning")
            if log_file:
                log_file.close()
            return None

        _get_frpc_pid_file().write_text(str(proc.pid))

        if _check_frpc_connected(log_path):
            push_log(f"公网访问: {PUBLIC_URL}")
            push_log(f"公网排障地址: http://{FRP_SERVER_ADDR}:{FRP_REMOTE_PORT}")
            push_log(f"frpc 隧道已连通，本地端口 {local_port} -> 阿里云 {FRP_REMOTE_PORT}")
            return proc

        push_log(f"frpc 进程已启动但隧道未连通 (尝试 {attempt}/{FRP_MAX_RETRIES})", level="warning")
        try:
            proc.kill()
            log_file.close()
        except Exception:
            pass

    push_log(f"frpc 启动失败：已重试 {FRP_MAX_RETRIES} 次仍未连通，请检查日志 {log_path}", level="error")
    return None


def list_serial_ports() -> tuple[list[str], str]:
    if not SERIAL_AVAILABLE:
        return [], "pyserial 未安装，请先在 yolo 环境中安装 pyserial。"
    ports = list(serial.tools.list_ports.comports())
    choices = [port.device for port in ports]
    lines = [f"{port.device} | {port.description} | hwid={port.hwid}" for port in ports]
    return choices, "\n".join(lines) if lines else "未发现串口设备。"


def build_hardware_packet(cmd: str, **kwargs) -> str:
    payload = {"cmd": cmd, **kwargs}
    return json.dumps(payload, ensure_ascii=False)


def send_ch340_packet(
    port: str,
    baudrate: int,
    packet: str,
    append_newline: bool,
    read_wait: float,
    client_state: dict | None,
) -> tuple[str, str, dict]:
    client_state = client_state or new_client_state()
    if not SERIAL_AVAILABLE:
        client_state = client_log(client_state, "CH340 测试失败：pyserial 未安装", "warning")
        return "pyserial 未安装，无法打开 CH340 串口。", get_client_logs(client_state), client_state
    port = (port or "").strip()
    if not port:
        client_state = client_log(client_state, "CH340 测试失败：未选择串口", "warning")
        return "请先选择 CH340 对应的 COM 端口。", get_client_logs(client_state), client_state
    data = (packet or "").strip()
    if not data:
        client_state = client_log(client_state, "CH340 测试失败：发送包为空", "warning")
        return "发送包不能为空。", get_client_logs(client_state), client_state

    tx = data + ("\n" if append_newline else "")
    try:
        with serial.Serial(port=port, baudrate=int(baudrate), timeout=max(read_wait, 0.1)) as set:
            set.reset_input_buffer()
            set.write(tx.encode("utf-8"))
            set.flush()
            time.sleep(max(read_wait, 0.0))
            waiting = set.in_waiting
            raw = set.read(waiting or 256)
        rx = raw.decode("utf-8", errors="replace").strip() if raw else ""
        result = f"TX ({port} @ {baudrate}): {data}\nRX: {rx or '<无返回>'}"
        client_state = client_log(client_state, f"CH340 发包完成 | port={port} | tx={data} | rx={rx or '<empty>'}")
        return result, get_client_logs(client_state), client_state
    except Exception as exc:
        client_state = client_log(client_state, f"CH340 发包失败: {exc}", "error")
        return f"CH340 发包失败：{exc}", get_client_logs(client_state), client_state


def refresh_serial_ports(client_state: dict | None) -> tuple[gr.Dropdown, str, str, dict]:
    client_state = client_state or new_client_state()
    choices, detail = list_serial_ports()
    client_state = client_log(client_state, f"刷新串口列表，发现 {len(choices)} 个端口")
    return (
        gr.update(choices=choices, value=choices[0] if choices else None),
        detail,
        get_client_logs(client_state),
        client_state,
    )


def send_temperature_query(
    port: str,
    baudrate: int,
    read_wait: float,
    client_state: dict | None,
) -> tuple[str, str, dict]:
    packet = build_hardware_packet("temperature", action="read")
    return send_ch340_packet(port, baudrate, packet, True, read_wait, client_state)


def send_servo_command(
    port: str,
    baudrate: int,
    servo_id: int,
    angle: int,
    read_wait: float,
    client_state: dict | None,
) -> tuple[str, str, dict]:
    packet = build_hardware_packet("servo", id=int(servo_id), angle=int(angle))
    return send_ch340_packet(port, baudrate, packet, True, read_wait, client_state)


def image_file_to_data_uri(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "image/jpeg"
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def prepare_image_for_qwen(image_path: str) -> tuple[str, dict]:
    original_path = Path(image_path)
    original_size_kb = round(original_path.stat().st_size / 1024, 2)

    QWEN_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = str(QWEN_TEMP_DIR / f"qwen_img_{uuid.uuid4().hex[:12]}.jpg")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size
        max_side = max(original_width, original_height)

        if max_side > QWEN_IMAGE_MAX_SIDE:
            scale = QWEN_IMAGE_MAX_SIDE / float(max_side)
            resized_width = max(1, int(original_width * scale))
            resized_height = max(1, int(original_height * scale))
            image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        else:
            resized_width, resized_height = original_width, original_height

        image.save(
            temp_path,
            format="JPEG",
            quality=QWEN_IMAGE_JPEG_QUALITY,
            optimize=True,
        )

    compressed_size_kb = round(Path(temp_path).stat().st_size / 1024, 2)
    info = {
        "原始尺寸": f"{original_width}x{original_height}",
        "压缩尺寸": f"{resized_width}x{resized_height}",
        "原始大小KB": original_size_kb,
        "压缩后大小KB": compressed_size_kb,
        "压缩路径": temp_path,
    }
    push_log(
        "已完成Qwen图片压缩 | "
        f"size={info['原始尺寸']}->{info['压缩尺寸']} | "
        f"kb={original_size_kb}->{compressed_size_kb}"
    )
    return temp_path, info


def audio_file_to_base64(file_path: str) -> tuple[str, str]:
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "audio/wav"
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return mime_type, encoded


def audio_file_to_data_uri(file_path: str) -> str:
    mime_type, encoded = audio_file_to_base64(file_path)
    return f"data:{mime_type};base64,{encoded}"


def extract_audio_metadata(file_path: str) -> dict:
    path = Path(file_path)
    info = {
        "文件名": path.name,
        "文件大小KB": round(path.stat().st_size / 1024, 2),
        "扩展名": path.suffix.lower(),
    }
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate) if rate else 0.0
            info.update(
                {
                    "声道数": wav_file.getnchannels(),
                    "采样率Hz": rate,
                    "时长秒": round(duration, 2),
                }
            )
    return info


def normalize_api_base(api_base: str) -> str:
    api_base = (api_base or "").strip()
    return api_base or DEFAULT_API_BASE


def build_multimodal_message(
    prompt: str,
    context_text: str,
    image_path: str | None = None,
    audio_path: str | None = None,
) -> list[dict]:
    parts: list[dict] = [{"type": "text", "text": f"{prompt}\n\n上下文信息：\n{context_text}"}]
    if image_path:
        parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_file_to_data_uri(image_path),
                },
            }
        )
    if audio_path:
        audio_format = Path(audio_path).suffix.lower().replace(".", "") or "wav"
        parts.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_file_to_data_uri(audio_path),
                    "format": audio_format,
                },
            }
        )
    return [{"role": "user", "content": parts}]


def call_qwen_api(
    *,
    api_key: str,
    api_base: str,
    model_name: str,
    prompt: str,
    context_text: str,
    image_path: str | None = None,
    audio_path: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> tuple[str, str]:
    if not api_key.strip():
        raise ValueError("未提供API Key，请在页面中填写或提前设置环境变量 DASHSCOPE_API_KEY。")

    url = normalize_api_base(api_base)
    payload = {
        "model": model_name.strip() or DEFAULT_IMAGE_MODEL,
        "messages": build_multimodal_message(
            prompt=prompt,
            context_text=context_text,
            image_path=image_path,
            audio_path=audio_path,
        ),
        "temperature": temperature,
        "stream": False,
        "max_tokens": max_tokens,
    }
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    push_log(
        f"准备调用Qwen接口 | model={payload['model']} | "
        f"image={'yes' if image_path else 'no'} | audio={'yes' if audio_path else 'no'}"
    )
    request = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        push_log(f"Qwen接口HTTP错误: status={exc.code}, detail={detail}", level="error")
        raise RuntimeError(f"Qwen接口请求失败，HTTP {exc.code}:\n{detail}") from exc
    except urllib.error.URLError as exc:
        push_log(f"Qwen接口网络错误: {exc}", level="error")
        raise RuntimeError(f"Qwen接口网络错误: {exc}") from exc

    result = json.loads(raw)
    push_log(f"Qwen接口调用成功，返回字段: {list(result.keys())}")
    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError(f"Qwen返回结果缺少choices字段:\n{json.dumps(result, ensure_ascii=False, indent=2)}")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        content = "\n".join(part for part in text_parts if part)
    if not content:
        raise RuntimeError(f"Qwen返回内容为空:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
    return str(content).strip(), str(result.get("model", payload["model"]))


def call_qwen_asr_api(
    *,
    api_key: str,
    api_base: str,
    model_name: str,
    audio_path: str,
) -> tuple[str, dict, str]:
    if not api_key.strip():
        raise ValueError("未提供API Key，请在页面中填写或提前设置环境变量 DASHSCOPE_API_KEY。")

    url = normalize_api_base(api_base)
    payload = {
        "model": model_name.strip() or DEFAULT_AUDIO_ASR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_file_to_data_uri(audio_path),
                        },
                    }
                ],
            }
        ],
        "stream": False,
        "asr_options": {
            "enable_itn": False,
        },
    }
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    push_log(f"准备调用Qwen ASR接口 | model={payload['model']} | audio=yes")
    request = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        push_log(f"Qwen ASR接口HTTP错误: status={exc.code}, detail={detail}", level="error")
        raise RuntimeError(f"Qwen ASR接口请求失败，HTTP {exc.code}:\n{detail}") from exc
    except urllib.error.URLError as exc:
        push_log(f"Qwen ASR接口网络错误: {exc}", level="error")
        raise RuntimeError(f"Qwen ASR接口网络错误: {exc}") from exc

    result = json.loads(raw)
    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError(f"Qwen ASR返回结果缺少choices字段:\n{json.dumps(result, ensure_ascii=False, indent=2)}")

    message = choices[0].get("message", {})
    transcript = (message.get("content") or "").strip()
    annotations = message.get("annotations") or []
    annotation_info = annotations[0] if annotations else {}
    push_log(
        "Qwen ASR接口调用成功 | "
        f"model={result.get('model', payload['model'])} | "
        f"language={annotation_info.get('language', 'unknown')} | "
        f"emotion={annotation_info.get('emotion', 'unknown')}"
    )
    if not transcript:
        raise RuntimeError(f"Qwen ASR返回内容为空:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
    return transcript, annotation_info, str(result.get("model", payload["model"]))


def format_detection_summary(detections: list[dict]) -> str:
    if not detections:
        return "未检测到目标。"

    counts: dict[str, int] = {}
    for item in detections:
        label = item["类别"]
        counts[label] = counts.get(label, 0) + 1
    stats = "，".join(f"{label} {count}个" for label, count in counts.items())
    return f"检测到 {len(detections)} 个目标：{stats}。"


def run_yolo_detection(
    image_path: str,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    device: str = "",
) -> tuple[np.ndarray, list[list], list[dict], str]:
    start = time.perf_counter()
    model = MODEL_MANAGER.get_model()
    push_log(
        f"开始YOLO推理 | image={image_path} | conf={conf_threshold:.2f} | "
        f"iou={iou_threshold:.2f} | max_det={max_det} | device='{device or 'auto'}'"
    )
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        max_det=max_det,
        verbose=False,
        device=device or None,
    )
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        max_det=max_det,
        verbose=False,
    )
    elapsed = (time.perf_counter() - start) * 1000
    result = results[0]
    annotated = result.plot()
    annotated = annotated[:, :, ::-1]

    detections: list[dict] = []
    rows: list[list] = []
    names = result.names
    for idx, box in enumerate(result.boxes, start=1):
        cls_id = int(box.cls.item())
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        conf = float(box.conf.item())
        xyxy = [round(float(x), 1) for x in box.xyxy[0].tolist()]
        item = {
            "序号": idx,
            "类别": label,
            "置信度": round(conf, 4),
            "边界框": xyxy,
        }
        detections.append(item)
        rows.append([idx, label, round(conf, 4), str(xyxy)])

    summary = format_detection_summary(detections)
    push_log(f"YOLO推理完成 | det_count={len(detections)} | cost_ms={elapsed:.2f} | summary={summary}")
    return annotated, rows, detections, summary


def resolve_api_key(api_key_input: str) -> str:
    api_key_input = (api_key_input or "").strip()
    if api_key_input:
        return api_key_input
    return os.environ.get("DASHSCOPE_API_KEY", "").strip() or os.environ.get("QWEN_API_KEY", "").strip()


def analyze_image(
    image_path: str,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    device: str,
    api_key_input: str,
    api_base: str,
    image_model_name: str,
    image_prompt: str,
    client_state: dict | None,
) -> tuple[np.ndarray | None, list[list], str, str, str, dict]:
    client_state = client_state or new_client_state()
    if not image_path:
        client_state = client_log(client_state, "图片分析未执行：未上传图片", "warning")
        return None, [], "请先上传图片。", "", get_client_logs(client_state), client_state

    try:
        client_state = client_log(client_state, f"开始图片行为分析 | image={Path(image_path).name}")
        annotated, rows, detections, summary = run_yolo_detection(
            image_path=image_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            device=device,
        )
        api_key = resolve_api_key(api_key_input)
        if not api_key:
            client_state = client_log(client_state, "未提供API Key，跳过Qwen分析，仅返回YOLO检测结果", "warning")
            report = "未提供API Key，当前仅展示YOLO检测结果。填写 API Key 后可继续调用 Qwen 分析。"
            return annotated, rows, summary, report, get_client_logs(client_state), client_state

        qwen_image_path, image_compress_info = prepare_image_for_qwen(image_path)
        context_text = (
            f"图片文件: {Path(image_path).name}\n"
            f"YOLO摘要: {summary}\n"
            f"YOLO详情: {json.dumps(detections, ensure_ascii=False)}\n"
            f"发送给Qwen前的图片压缩信息: {json.dumps(image_compress_info, ensure_ascii=False)}"
        )
        try:
            report, used_model = call_qwen_api(
                api_key=api_key,
                api_base=api_base,
                model_name=image_model_name,
                prompt=image_prompt or DEFAULT_IMAGE_PROMPT,
                context_text=context_text,
                image_path=qwen_image_path,
                max_tokens=500,
            )
        finally:
            try:
                os.remove(qwen_image_path)
            except OSError:
                push_log(f"临时压缩图片清理失败: {qwen_image_path}", level="warning")

        report = (
            f"本次图片分析模型：{used_model}\n"
            f"Qwen图片压缩：{image_compress_info['原始尺寸']} -> {image_compress_info['压缩尺寸']}，"
            f"{image_compress_info['原始大小KB']}KB -> {image_compress_info['压缩后大小KB']}KB\n\n"
            f"{report}"
        )
        client_state = client_log(client_state, "图片行为分析流程完成")
        return annotated, rows, summary, report, get_client_logs(client_state), client_state
    except Exception as exc:
        client_state = client_log(client_state, f"图片行为分析失败: {exc}", "error")
        return None, [], "图片分析失败。", str(exc), get_client_logs(client_state), client_state


def analyze_audio(
    audio_path: str,
    api_key_input: str,
    api_base: str,
    audio_asr_model_name: str,
    audio_reasoning_model_name: str,
    audio_prompt: str,
    client_state: dict | None,
) -> tuple[str, str, list[list], str, dict]:
    client_state = client_state or new_client_state()
    if not audio_path:
        client_state = client_log(client_state, "音频分析未执行：未上传音频", "warning")
        return (
            "请先上传音频。",
            "",
            format_client_audio_records(client_state),
            get_client_logs(client_state),
            client_state,
        )

    try:
        private_audio_path, client_state = store_client_audio(audio_path, client_state)
        metadata = extract_audio_metadata(private_audio_path)
        client_state = client_log(
            client_state,
            f"开始音频分析 | audio={Path(private_audio_path).name} | metadata={metadata}",
        )

        api_key = resolve_api_key(api_key_input)
        if not api_key:
            client_state = client_log(client_state, "未提供API Key，跳过Qwen音频分析，仅返回音频元信息", "warning")
            report = "未提供API Key，当前仅解析音频基础信息。填写 API Key 后可调用 Qwen 分析猫叫情绪。"
            return (
                json.dumps(metadata, ensure_ascii=False, indent=2),
                report,
                format_client_audio_records(client_state),
                get_client_logs(client_state),
                client_state,
            )

        merged_metadata = {
            **metadata,
            "分析方式": "直接音频理解，不调用ASR转写",
            "音频识别模型": audio_asr_model_name or DEFAULT_AUDIO_ASR_MODEL,
        }
        metadata_text = json.dumps(merged_metadata, ensure_ascii=False, indent=2)

        report, reasoning_used_model = call_qwen_api(
            api_key=api_key,
            api_base=api_base,
            model_name=audio_reasoning_model_name,
            prompt=audio_prompt or DEFAULT_AUDIO_PROMPT,
            context_text=(
                "音频文件元信息如下，请结合原始猫叫音频直接判断猫咪情绪。"
                "特别注意区分猫咪吵架/攻击/对峙与发情叫声；有冲突特征时优先输出fighting。\n"
                f"{metadata_text}"
            ),
            audio_path=private_audio_path,
            temperature=0.1,
            max_tokens=480,
        )
        report = f"本次音频识别模型：未调用ASR\n本次猫叫情绪分析模型：{reasoning_used_model}\n\n{report}"
        client_state = client_log(client_state, "音频分析流程完成")
        return (
            metadata_text,
            report,
            format_client_audio_records(client_state),
            get_client_logs(client_state),
            client_state,
        )
    except Exception as exc:
        client_state = client_log(client_state, f"音频分析失败: {exc}", "error")
        return (
            "音频分析失败。",
            str(exc),
            format_client_audio_records(client_state),
            get_client_logs(client_state),
            client_state,
        )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        shared_key = gr.State(value="")
        client_state = gr.State(value=new_client_state())

        # ======================== 欢迎弹窗 ========================
        with gr.Column(visible=True, elem_classes=["welcome-container"]) as welcome_section:
            gr.Markdown(
                f"""
                # {APP_TITLE}

                ## 使用说明

                本系统提供两种猫咪行为分析能力：

                **1. 图片行为分析**
                - 上传猫咪照片，系统先使用 **YOLO 目标检测** 识别猫咪行为类别
                  （猫喝水、猫进食、猫玩耍、猫睡觉、猫呕吐、猫如厕）
                - 再将检测结果与图片一起发送给 **阿里云通义千问 (Qwen)** 多模态大模型
                - Qwen 会综合判断猫咪行为、评估健康风险，并给出养护建议

                **2. 音频叫声分析**
                - 上传猫咪叫声音频（支持 WAV 等常见格式）
                - 系统直接通过 **Qwen Omni** 分析猫叫音频，不再把猫叫当成人类语音做 ASR 转写
                - Qwen 会按固定标签输出猫咪情绪、置信度、紧急程度和主人建议

                ---
                ### 重要提示

                - 本系统依赖 **阿里云 DashScope API**，需要有效的 API Key 才能使用 AI 分析功能
                - 如不填写 API Key，仅能查看基础的 YOLO 检测 / 音频元信息结果
                - API Key 仅保存在当前会话内存中，关闭页面后不会留存

                **获取 API Key：** 访问 [阿里云百炼平台](https://bailian.console.aliyun.com/) 开通 DashScope 服务即可获取

                > **安全提示：** API Key 仅保存在当前浏览器会话内存中，关闭页面后自动清除。建议单独申请专用子 Key 并设置额度限制，避免使用高权限主 Key。

                ---
                """,
            )
            with gr.Row():
                welcome_api_key = gr.Textbox(
                    label="请输入阿里云 DashScope API Key",
                    type="password",
                    placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    scale=4,
                )
            with gr.Row():
                welcome_btn = gr.Button("确认并进入系统", variant="primary", size="lg")
            welcome_msg = gr.Markdown("")

        # ======================== 主功能界面 ========================
        with gr.Column(visible=False, elem_classes=["main-container"]) as main_section:
            gr.Markdown(
                f"""
                # {APP_TITLE}
                支持两类演示流程：
                1. 上传猫咪图片，先做 YOLO 目标检测，再把图片和检测结果交给 Qwen 做语义分析。
                2. 上传预先准备好的猫叫声音频，提取元信息后直接交给 Qwen Omni 做猫叫情绪分类。

                后台终端会打印详细日志，页面中也会同步显示最近日志，方便调试。
                """
            )

            with gr.Accordion("全局接口配置", open=True):
                api_key_input = gr.Textbox(
                    label="Qwen / DashScope API Key",
                    type="password",
                    placeholder="优先读取这里；留空则读取 DASHSCOPE_API_KEY 或 QWEN_API_KEY",
                )
                api_base = gr.Textbox(
                    label="兼容接口地址",
                    value=DEFAULT_API_BASE,
                )
                model_overview = gr.Textbox(
                    label="当前默认模型总览",
                    value=(
                        f"图片分析默认模型：{DEFAULT_IMAGE_MODEL}\n"
                        f"音频识别默认模型：{DEFAULT_AUDIO_ASR_MODEL}\n"
                        f"猫叫情绪分析默认模型：{DEFAULT_AUDIO_REASONING_MODEL}"
                    ),
                    lines=3,
                    interactive=False,
                )
                gr.Markdown(
                    "建议先确认你的模型是否支持图片/音频多模态输入；若接口报错，可直接在页面日志与终端日志中查看请求阶段。"
                )

            with gr.Tabs():
                with gr.Tab("图片行为分析"), gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="filepath",
                            label="上传待分析图片",
                        )
                        conf_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.25,
                            step=0.05,
                            label="置信度阈值",
                        )
                        iou_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.45,
                            step=0.05,
                            label="NMS IoU 阈值",
                        )
                        max_det = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=20,
                            step=1,
                            label="最多保留目标数",
                        )
                        device = gr.Dropdown(
                            label="YOLO 推理设备",
                            choices=["", "cpu", "cuda:0", "cuda:1"],
                            value="",
                            allow_custom_value=True,
                            info="留空自动检测，可选 cpu / cuda:0 等",
                        )
                        image_model_name = gr.Textbox(
                            label="图片分析模型名",
                            value=DEFAULT_IMAGE_MODEL,
                        )
                        image_prompt = gr.Textbox(
                            label="图片分析提示词",
                            value=DEFAULT_IMAGE_PROMPT,
                            lines=4,
                        )
                        image_btn = gr.Button("开始图片分析", variant="primary")

                    with gr.Column(scale=1):
                        image_output = gr.Image(label="YOLO 标注结果")
                        detection_table = gr.Dataframe(
                            headers=["序号", "类别", "置信度", "边界框"],
                            datatype=["number", "str", "number", "str"],
                            row_count=1,
                            label="检测详情",
                        )
                        detection_summary = gr.Textbox(
                            label="检测摘要",
                            lines=2,
                        )
                        image_report = gr.Textbox(
                            label="Qwen 行为分析报告",
                            lines=12,
                        )
                        image_logs = gr.Textbox(
                            label="本会话日志",
                            lines=12,
                            value="暂无本会话日志。",
                        )

                with gr.Tab("音频叫声分析"), gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            type="filepath",
                            sources=["upload", "microphone"],
                            label="上传预先准备好的猫叫声音频",
                        )
                        audio_asr_model_name = gr.Textbox(
                            label="音频识别模型名（猫叫分析不使用ASR）",
                            value=DEFAULT_AUDIO_ASR_MODEL,
                        )
                        audio_reasoning_model_name = gr.Textbox(
                            label="猫叫情绪分析模型名",
                            value=DEFAULT_AUDIO_REASONING_MODEL,
                        )
                        audio_prompt = gr.Textbox(
                            label="音频分析提示词",
                            value=DEFAULT_AUDIO_PROMPT,
                            lines=4,
                        )
                        audio_btn = gr.Button("开始音频分析", variant="primary")

                    with gr.Column(scale=1):
                        audio_metadata = gr.Textbox(
                            label="音频元信息",
                            lines=8,
                        )
                        audio_report = gr.Textbox(
                            label="Qwen 猫叫情绪分析报告",
                            lines=12,
                        )
                        audio_records = gr.Dataframe(
                            headers=["序号", "上传时间", "文件名", "大小KB"],
                            datatype=["number", "str", "str", "number"],
                            row_count=1,
                            label="本会话音频记录",
                        )
                        audio_logs = gr.Textbox(
                            label="本会话日志",
                            lines=12,
                            value="暂无本会话日志。",
                        )

                with gr.Tab("CH340 硬件收发测试"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            serial_port = gr.Dropdown(
                                label="CH340 串口",
                                choices=[],
                                interactive=True,
                            )
                            serial_baudrate = gr.Number(
                                label="波特率",
                                value=115200,
                                precision=0,
                            )
                            serial_read_wait = gr.Slider(
                                minimum=0.1,
                                maximum=3.0,
                                value=0.5,
                                step=0.1,
                                label="读取等待秒数",
                            )
                            refresh_ports_btn = gr.Button("刷新串口")
                            serial_ports_detail = gr.Textbox(
                                label="串口详情",
                                lines=5,
                            )

                        with gr.Column(scale=1):
                            raw_packet = gr.Textbox(
                                label="原始发送包",
                                value='{"cmd":"ping"}',
                                lines=4,
                            )
                            append_newline = gr.Checkbox(
                                label="发送末尾追加换行",
                                value=True,
                            )
                            send_raw_btn = gr.Button("发送原始包", variant="primary")
                            hardware_result = gr.Textbox(
                                label="收发结果",
                                lines=8,
                            )
                            hardware_logs = gr.Textbox(
                                label="本会话硬件测试日志",
                                lines=8,
                                value="暂无本会话日志。",
                            )

                    with gr.Row():
                        with gr.Column(scale=1):
                            temperature_btn = gr.Button("读取温度")
                        with gr.Column(scale=1):
                            servo_id = gr.Number(label="舵机 ID", value=1, precision=0)
                            servo_angle = gr.Slider(
                                minimum=0,
                                maximum=180,
                                value=90,
                                step=1,
                                label="舵机角度",
                            )
                            servo_btn = gr.Button("发送舵机角度")

            image_btn.click(
                fn=analyze_image,
                inputs=[
                    image_input,
                    conf_threshold,
                    iou_threshold,
                    max_det,
                    device,
                    api_key_input,
                    api_base,
                    image_model_name,
                    image_prompt,
                    client_state,
                ],
                outputs=[
                    image_output,
                    detection_table,
                    detection_summary,
                    image_report,
                    image_logs,
                    client_state,
                ],
            )
            audio_btn.click(
                fn=analyze_audio,
                inputs=[
                    audio_input,
                    api_key_input,
                    api_base,
                    audio_asr_model_name,
                    audio_reasoning_model_name,
                    audio_prompt,
                    client_state,
                ],
                outputs=[
                    audio_metadata,
                    audio_report,
                    audio_records,
                    audio_logs,
                    client_state,
                ],
            )
            refresh_ports_btn.click(
                fn=refresh_serial_ports,
                inputs=[client_state],
                outputs=[serial_port, serial_ports_detail, hardware_logs, client_state],
            )
            send_raw_btn.click(
                fn=send_ch340_packet,
                inputs=[
                    serial_port,
                    serial_baudrate,
                    raw_packet,
                    append_newline,
                    serial_read_wait,
                    client_state,
                ],
                outputs=[hardware_result, hardware_logs, client_state],
            )
            temperature_btn.click(
                fn=send_temperature_query,
                inputs=[serial_port, serial_baudrate, serial_read_wait, client_state],
                outputs=[hardware_result, hardware_logs, client_state],
            )
            servo_btn.click(
                fn=send_servo_command,
                inputs=[
                    serial_port,
                    serial_baudrate,
                    servo_id,
                    servo_angle,
                    serial_read_wait,
                    client_state,
                ],
                outputs=[hardware_result, hardware_logs, client_state],
            )

            gr.on(
                triggers=[image_model_name.change, audio_asr_model_name.change, audio_reasoning_model_name.change],
                fn=lambda model_name, audio_asr, audio_reason: (
                    f"图片分析默认模型：{model_name}\n音频识别默认模型：{audio_asr}\n猫叫情绪分析默认模型：{audio_reason}"
                ),
                inputs=[image_model_name, audio_asr_model_name, audio_reasoning_model_name],
                outputs=[model_overview],
            )

        # ======================== 欢迎页 → 主界面切换 ========================
        def enter_system(api_key: str) -> tuple:
            key = (api_key or "").strip()
            if not key:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "",
                    "请输入有效的 API Key 后再进入系统。",
                )
            push_log("用户已填写 API Key，进入主界面")
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                key,
                "",
            )

        welcome_btn.click(
            fn=enter_system,
            inputs=[welcome_api_key],
            outputs=[welcome_section, main_section, shared_key, welcome_msg],
        )

        # 将共享的 key 同步到主界面的 api_key_input
        shared_key.change(
            fn=lambda k: k,
            inputs=[shared_key],
            outputs=[api_key_input],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--host", default="0.0.0.0", help="Gradio监听地址")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_LOCAL_PORT,
        help=f"Gradio本地端口 (默认 {DEFAULT_LOCAL_PORT})",
    )
    parser.add_argument(
        "--no-frp",
        action="store_true",
        default=False,
        help="禁用 frp 公网映射，仅启动本地 Gradio 服务",
    )
    parser.add_argument(
        "--weight",
        default=str(DEFAULT_WEIGHT),
        help="YOLO权重路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global MODEL_MANAGER
    MODEL_MANAGER = ModelManager(Path(args.weight))

    port = find_available_port(args.port)
    if port != args.port:
        push_log(f"端口 {args.port} 已被占用，自动切换为端口 {port}")

    frpc_proc = None
    if not args.no_frp:
        frpc_proc = _start_frpc_tunnel(port)
        if frpc_proc is not None:
            atexit.register(lambda: _stop_own_frpc())

    push_log(f"应用启动 | host={args.host} | port={port} | frp={'on' if frpc_proc else 'off'} | weight={args.weight}")

    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=port,
        share=False,
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
