#!/usr/bin/env python3
"""Start the local pet behavior application."""

from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
import webbrowser
from pathlib import Path


ENV_NAME = os.environ.get("PET_BEHAVIOR_ENV", "petbehavior")
ROOT_DIR = Path(__file__).resolve().parent
APP_FILE = ROOT_DIR / "pet_behavior_gradio_app.py"
DEFAULT_WEIGHT = ROOT_DIR / "runs" / "train" / "pet_behavior" / "weights" / "best.pt"


def environment_exists() -> bool:
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    return result.returncode == 0 and any(
        line.split() and line.split()[0] == ENV_NAME for line in result.stdout.splitlines()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动本地宠物行为分析应用")
    parser.add_argument("--port", type=int, default=7861, help="本地端口，默认 7861")
    parser.add_argument("--weight", default=str(DEFAULT_WEIGHT), help="行为模型权重路径")
    parser.add_argument("--enable-frp", action="store_true", help="显式启用 FRP 公网映射")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not APP_FILE.exists():
        raise SystemExit(f"未找到应用入口: {APP_FILE}")
    if not environment_exists():
        raise SystemExit(f"未找到 Conda 环境 {ENV_NAME}，请先运行 python setup_environment.py。")

    command = ["conda", "run", "-n", ENV_NAME, "python", str(APP_FILE), "--port", str(args.port)]
    command.extend(["--weight", args.weight])
    if args.enable_frp:
        command.append("--enable-frp")

    def open_browser() -> None:
        time.sleep(3)
        webbrowser.open(f"http://127.0.0.1:{args.port}")

    print("启动智能宠物行为识别系统")
    print(f"访问地址: http://127.0.0.1:{args.port}")
    print(f"公网映射: {'已显式启用' if args.enable_frp else '未启用'}")
    print("按 Ctrl+C 停止应用")
    threading.Thread(target=open_browser, daemon=True).start()
    subprocess.run(command, cwd=ROOT_DIR, check=True)


if __name__ == "__main__":
    main()
