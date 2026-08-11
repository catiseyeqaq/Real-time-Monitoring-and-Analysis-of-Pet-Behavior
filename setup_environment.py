#!/usr/bin/env python3
"""
智能宠物行为识别系统 - 一键环境配置脚本 (Python版)
使用方法：python setup_environment.py
"""

import os
import sys
import subprocess
import time

# 颜色代码 (Windows 需要特殊处理)
if os.name == "nt":
    os.system("color")  # 启用 ANSI 颜色

class Colors:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

def print_colored(text: str, color: str) -> None:
    """打印彩色文本"""
    print(f"{color}{text}{Colors.RESET}", flush=True)

def print_step(step: int, total: int, message: str) -> None:
    """打印步骤信息"""
    print()
    print_colored(f"[{step}/{total}] {message}...", Colors.YELLOW)

def run_quiet(cmd: list) -> subprocess.CompletedProcess:
    """静默运行命令（不显示输出，用于快速命令）"""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print_colored(f"命令执行失败: {' '.join(cmd)}", Colors.RED)
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        sys.exit(1)

def run_show(cmd: list, desc: str = "") -> bool:
    """运行命令并实时显示输出（用于耗时较长的命令）"""
    if desc:
        print(desc)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # 实时打印输出
        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(f"  {line}", flush=True)
        
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print_colored(f"命令执行失败: {' '.join(cmd)}", Colors.RED)
        print_colored(f"错误: {e}", Colors.RED)
        return False

def check_conda() -> None:
    """检查 Conda 是否安装"""
    print_step(1, 6, "检查 Conda 安装状态")
    
    try:
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_colored("错误：未检测到 Conda！请先安装 Miniconda 或 Anaconda。", Colors.RED)
        print_colored("下载地址：https://docs.conda.io/en/latest/miniconda.html", Colors.YELLOW)
        input("按 Enter 键退出...")
        sys.exit(1)
    
    print_colored(f"✓ Conda 已安装: {result.stdout.strip()}", Colors.GREEN)

def check_environment(env_name: str) -> bool:
    """检查 Conda 环境是否存在，返回 True 表示已存在且用户选择保留"""
    print_step(2, 6, f"检查 Conda 环境 [{env_name}]")
    
    result = run_quiet(["conda", "env", "list"])
    env_exists = any(
        line.startswith(env_name + " ") or line.startswith(env_name + "\t")
        for line in result.stdout.split("\n")
    )
    
    if env_exists:
        print_colored(f"环境 [{env_name}] 已存在", Colors.YELLOW)
        response = input("是否删除并重新创建？(y/N): ").strip().lower()
        if response == "y":
            print_colored("删除旧环境...", Colors.YELLOW)
            run_quiet(["conda", "remove", "-n", env_name, "--all", "-y"])
            return False
        else:
            print_colored("跳过环境创建，使用现有环境", Colors.GREEN)
            return True
    
    return False

def create_environment(env_name: str, python_version: str) -> None:
    """创建 Conda 环境（实时显示进度）"""
    print_step(3, 6, f"创建 Conda 环境 [{env_name}] (Python {python_version})")
    print_colored("正在下载和安装 Python 环境，请耐心等待...", Colors.YELLOW)
    
    success = run_show(
        ["conda", "create", "-n", env_name, f"python={python_version}", "-y"],
        desc=""
    )
    
    if success:
        print_colored(f"✓ 环境 [{env_name}] 创建成功", Colors.GREEN)
    else:
        print_colored("错误：环境创建失败！", Colors.RED)
        input("按 Enter 键退出...")
        sys.exit(1)

def install_dependencies(env_name: str, project_dir: str) -> None:
    """安装项目依赖（实时显示进度）"""
    print_step(4, 6, "安装项目依赖（这可能需要几分钟）")
    
    print_colored("依赖列表：", Colors.YELLOW)
    print("  - ultralytics >= 8.4.0 (YOLO)")
    print("  - torch >= 2.0.0 (CPU 版本)")
    print("  - torchvision >= 0.15.0")
    print("  - opencv-python >= 4.8.0")
    print("  - numpy >= 1.24.0")
    print("  - Pillow >= 10.0.0")
    print("  - gradio >= 4.0.0 (Web UI)")
    print("  - pyserial >= 3.5 (串口通信)")
    
    requirements_file = os.path.join(project_dir, "requirements.txt")
    
    if os.path.exists(requirements_file):
        print_colored("\n正在安装，实时进度如下：", Colors.YELLOW)
        print("-" * 48)
        cmd = [
            "conda", "run", "-n", env_name, "--no-capture-output",
            "pip", "install", "-r", requirements_file
        ]
        success = run_show(cmd)
        print("-" * 48)
    else:
        print_colored("警告：未找到 requirements.txt，将手动安装依赖...", Colors.YELLOW)
        dependencies = [
            "ultralytics>=8.4.0",
            "torch>=2.0.0", "torchvision>=0.15.0",
            "opencv-python>=4.8.0", "numpy>=1.24.0",
            "Pillow>=10.0.0", "gradio>=4.0.0", "pyserial>=3.5"
        ]
        cmd = ["conda", "run", "-n", env_name, "--no-capture-output", "pip", "install"] + dependencies
        success = run_show(cmd)
    
    if not success:
        print_colored("错误：依赖安装失败！", Colors.RED)
        print_colored("请检查网络连接后重试。", Colors.YELLOW)
        input("按 Enter 键退出...")
        sys.exit(1)
    
    print_colored("✓ 依赖安装成功", Colors.GREEN)

def verify_installation(env_name: str) -> bool:
    """验证安装"""
    print_step(5, 6, "验证安装")
    
    packages = {
        "ultralytics": "ultralytics",
        "torch": "torch", 
        "gradio": "gradio",
        "pyserial": "serial",
    }
    
    all_success = True
    for name, import_name in packages.items():
        cmd = ["conda", "run", "-n", env_name, "python", "-c", f"import {import_name}"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            print_colored(f"  ✓ {name}", Colors.GREEN)
        except subprocess.CalledProcessError:
            print_colored(f"  ✗ {name} (未安装)", Colors.RED)
            all_success = False
    
    return all_success

def check_weights(project_dir: str) -> None:
    """检查 YOLO 权重文件"""
    print_step(6, 6, "检查 YOLO 权重文件")
    
    weight_files = [
        os.path.join(project_dir, "runs", "train", "yolo-GDL", "weights", "best.pt"),
        os.path.join(project_dir, "yolo26n.pt")
    ]
    
    weight_found = False
    for wf in weight_files:
        if os.path.exists(wf):
            print_colored(f"  ✓ 找到权重文件：{wf}", Colors.GREEN)
            weight_found = True
    
    if not weight_found:
        print_colored("  警告：未找到训练好的权重文件", Colors.YELLOW)
        print_colored("  首次启动时会自动下载 yolo26n.pt 预训练模型", Colors.YELLOW)

def print_completion_message(env_name: str, project_dir: str, success: bool) -> None:
    """打印完成信息"""
    print()
    print_colored("=" * 48, Colors.CYAN)
    
    if success:
        print_colored("✓ 环境配置完成！", Colors.GREEN)
        print_colored("=" * 48, Colors.CYAN)
        print()
        print_colored("下一步操作：", Colors.YELLOW)
        print(f"  1. 激活环境：conda activate {env_name}")
        print(f"  2. 进入目录：cd {project_dir}")
        print(f"  3. 启动应用：python pet_behavior_gradio_app.py")
        print()
        print_colored(f"或者运行启动脚本：python start_app.py", Colors.CYAN)
    else:
        print_colored("⚠ 环境配置完成，但部分依赖可能未正确安装", Colors.YELLOW)
        print_colored("=" * 48, Colors.CYAN)
    
    print()

def main():
    """主函数"""
    print()
    print_colored("=" * 48, Colors.CYAN)
    print_colored("  智能宠物行为识别系统 - 环境配置脚本", Colors.CYAN)
    print_colored("=" * 48, Colors.CYAN)
    print()
    
    env_name = "yolocat"
    python_version = "3.12"
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    check_conda()
    
    env_exists = check_environment(env_name)
    if not env_exists:
        create_environment(env_name, python_version)
    
    install_dependencies(env_name, project_dir)
    
    success = verify_installation(env_name)
    
    check_weights(project_dir)
    
    print_completion_message(env_name, project_dir, success)
    
    input("按 Enter 键退出...")

if __name__ == "__main__":
    main()
