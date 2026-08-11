#!/usr/bin/env python3
"""Start the local pet behavior application."""

from __future__ import annotations

import os
import subprocess
import sys
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
    return result.returncode == 0 and any(line.split() and line.split()[0] == ENV_NAME for line in result.stdout.splitlines())


def main() -> None:
    if not APP_FILE.exists():
        raise SystemExit(f"未找到应用入口: {APP_FILE}")
    if not environment_exists():
        raise SystemExit(f"未找到 Conda 环境 {ENV_NAME}，请先运行 python setup_environment.py。")

    command = ["conda", "run", "-n", ENV_NAME, "python", str(APP_FILE)]
    if DEFAULT_WEIGHT.exists():
        command.extend(["--weight", str(DEFAULT_WEIGHT)])

    def open_browser() -> None:
        time.sleep(3)
        webbrowser.open("http://127.0.0.1:7861")

    print("启动智能宠物行为识别系统")
    print("访问地址: http://127.0.0.1:7861")
    print("按 Ctrl+C 停止应用")
    threading.Thread(target=open_browser, daemon=True).start()
    subprocess.run(command, cwd=ROOT_DIR, check=True)


if __name__ == "__main__":
    main()
