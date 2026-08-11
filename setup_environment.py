#!/usr/bin/env python3
"""Create and verify a local environment for the pet behavior application."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ENV_NAME = os.environ.get("PET_BEHAVIOR_ENV", "petbehavior")
PYTHON_VERSION = "3.12"
ROOT_DIR = Path(__file__).resolve().parent


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, check=check, text=True)


def conda_exists() -> bool:
    result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
    return result.returncode == 0


def environment_exists() -> bool:
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
    return any(line.split() and line.split()[0] == ENV_NAME for line in result.stdout.splitlines())


def main() -> None:
    if not conda_exists():
        raise SystemExit("未检测到 Conda，请先安装 Miniconda 或 Anaconda。")

    if not environment_exists():
        run(["conda", "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"])
    else:
        print(f"复用已有环境: {ENV_NAME}")

    run(
        [
            "conda",
            "run",
            "-n",
            ENV_NAME,
            "python",
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
        ]
    )
    run(
        [
            "conda",
            "run",
            "-n",
            ENV_NAME,
            "python",
            "-m",
            "pip",
            "install",
            "-r",
            str(ROOT_DIR / "requirements.txt"),
        ]
    )
    run(
        [
            "conda",
            "run",
            "-n",
            ENV_NAME,
            "python",
            "-c",
            "import gradio, torch, serial, ultralytics; print('依赖验证通过')",
        ]
    )
    print(f"环境配置完成: {ENV_NAME}")
    print(f"启动命令: conda run -n {ENV_NAME} python start_app.py")


if __name__ == "__main__":
    main()
