#!/usr/bin/env python3
"""
智能宠物行为识别系统 - 一键启动脚本 (Python版)
使用方法：python start_app.py
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

# 颜色代码 (Windows 需要特殊处理)
if os.name == "nt":
    os.system("color")  # 启用 ANSI 颜色

class Colors:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

def print_colored(text: str, color: str) -> None:
    """打印彩色文本"""
    print(f"{color}{text}{Colors.RESET}")

def print_step(step: int, total: int, message: str) -> None:
    """打印步骤信息"""
    print()
    print_colored(f"[{step}/{total}] {message}...", Colors.YELLOW)

def run_command(cmd: list, capture: bool = True, check: bool = True, shell: bool = False) -> subprocess.CompletedProcess:
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, capture_output=capture, text=True, check=check, shell=shell)
        return result
    except subprocess.CalledProcessError as e:
        print_colored(f"命令执行失败: {' '.join(cmd)}", Colors.RED)
        print_colored(f"错误信息: {e}", Colors.RED)
        if check:
            sys.exit(1)
        return e

def check_environment(env_name: str) -> bool:
    """检查 Conda 环境是否存在"""
    print_step(1, 4, f"检查 Conda 环境")
    
    result = run_command(["conda", "env", "list"], capture=True, check=True)
    env_exists = any(line.startswith(env_name + " ") or line.startswith(env_name + "\t") 
                     for line in result.stdout.split("\n"))
    
    if not env_exists:
        print_colored(f"错误：未找到 Conda 环境 [{env_name}]", Colors.RED)
        print_colored(f"请先运行 python setup_environment.py 配置环境", Colors.YELLOW)
        input("按 Enter 键退出...")
        sys.exit(1)
    
    print_colored(f"✓ 环境 [{env_name}] 已找到", Colors.GREEN)
    return True

def check_project_files(project_dir: str) -> None:
    """检查项目文件"""
    print_step(2, 4, "检查项目文件")
    
    app_script = os.path.join(project_dir, "pet_behavior_gradio_app.py")
    
    if not os.path.exists(app_script):
        print_colored(f"错误：未找到启动脚本 [pet_behavior_gradio_app.py]", Colors.RED)
        print_colored(f"当前目录：{project_dir}", Colors.YELLOW)
        input("按 Enter 键退出...")
        sys.exit(1)
    
    print_colored(f"✓ 找到启动脚本：pet_behavior_gradio_app.py", Colors.GREEN)

def check_weights(project_dir: str) -> None:
    """检查 YOLO 权重文件"""
    print()
    print_colored("检查 YOLO 权重文件...", Colors.YELLOW)
    
    weight_paths = [
        os.path.join(project_dir, "runs", "train", "yolo-GDL", "weights", "best.pt"),
        os.path.join(project_dir, "yolo26n.pt")
    ]
    
    weight_found = False
    for wp in weight_paths:
        if os.path.exists(wp):
            print_colored(f"✓ 找到权重文件：{wp}", Colors.GREEN)
            weight_found = True
    
    if not weight_found:
        print_colored("⚠ 未找到训练好的权重文件", Colors.YELLOW)
        print_colored("  将使用 YOLO26n 预训练模型（首次运行会自动下载）", Colors.YELLOW)

def get_conda_python(env_name: str) -> str:
    """获取 Conda 环境的 Python 路径"""
    result = run_command(["conda", "info", "--envs"], capture=True, check=True)
    
    lines = result.stdout.split("\n")
    for line in lines:
        if line.startswith(env_name + " ") or line.startswith(env_name + "\t"):
            env_path = line.split()[1].strip()
            if os.name == "nt":
                return os.path.join(env_path, "python.exe")
            else:
                return os.path.join(env_path, "bin", "python")
    
    # 默认路径
    if os.name == "nt":
        return f"D:\\miniconda3\\envs\\{env_name}\\python.exe"
    else:
        return f"~/miniconda3/envs/{env_name}/bin/python"

def start_application(env_name: str, project_dir: str) -> None:
    """启动应用"""
    print_step(3, 4, "启动应用")
    
    # 切换到项目目录
    os.chdir(project_dir)
    
    # 构建启动命令
    app_script = "pet_behavior_gradio_app.py"
    weight_path = os.path.join(project_dir, "runs", "train", "yolo-GDL", "weights", "best.pt")
    
    cmd = [sys.executable, app_script]
    
    # 如果有权重文件，指定权重路径
    if os.path.exists(weight_path):
        cmd.extend(["--weight", weight_path])
    
    print_colored("=" * 48, Colors.CYAN)
    print_colored(f"  启动命令：{' '.join(cmd)}", Colors.CYAN)
    print_colored(f"  访问地址：http://127.0.0.1:7861", Colors.CYAN)
    print_colored(f"  按 Ctrl+C 停止应用", Colors.CYAN)
    print_colored("=" * 48, Colors.CYAN)
    print()
    
    # 延迟打开浏览器
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://127.0.0.1:7861")
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # 启动应用
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print()
        print_colored("应用已停止", Colors.YELLOW)
    except Exception as e:
        print_colored(f"启动应用失败：{e}", Colors.RED)
        input("按 Enter 键退出...")

def main():
    """主函数"""
    # 打印标题
    print_colored("=" * 48, Colors.CYAN)
    print_colored("  智能宠物行为识别系统 - 启动中...", Colors.CYAN)
    print_colored("=" * 48, Colors.CYAN)
    print()
    
    # 设置变量
    env_name = "yolocat"
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 检查环境
    check_environment(env_name)
    
    # 检查项目文件
    check_project_files(project_dir)
    
    # 检查权重文件
    check_weights(project_dir)
    
    # 启动应用
    start_application(env_name, project_dir)
    
    print()
    input("按 Enter 键退出...")

if __name__ == "__main__":
    main()
