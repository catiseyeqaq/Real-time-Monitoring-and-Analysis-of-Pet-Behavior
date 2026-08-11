# 宠物行为实时监测与分析系统

面向宠物行为健康监测的 **视觉-音频多模态分析系统**：基于 **YOLO26** 完成猫咪行为目标检测，并结合 **通义千问 (Qwen) 视觉语言模型** 与 **Qwen-Omni 音频理解**，实现图像行为识别、叫声情绪/需求分析、风险提示与饲养建议生成。

> 项目定位：研究与工程演示版本。模型权重、数据集原始图片与音频不随代码仓库公开提供。

## 项目简介

系统提供两条完整的分析流水线，并整合在一个 Gradio Web 应用中：

1. **图片行为分析**：上传猫咪照片 → YOLO26 检测行为目标（猫喝水 / 猫进食 / 猫玩耍 / 猫睡觉 / 猫呕吐 / 猫如厕）→ 将检测结果与压缩后的图片一起发送给 Qwen 多模态大模型 → 输出行为判断、置信依据、健康风险与主人建议。
2. **音频叫声分析**：上传猫叫声音频 → 提取音频元信息 → 直接交给 Qwen-Omni 做猫叫情绪分类（不做人类语音 ASR 转写）→ 按固定标签输出情绪、置信度、紧急程度与建议（hungry / attention / fear / pain / warning / fighting / mating / relaxed / unknown）。

```
图片: 上传 → YOLO26 检测 → 检测摘要 + 压缩图 → Qwen 多模态 → 行为分析报告
音频: 上传 → 会话级存储 → 元信息提取 → Qwen-Omni 直接音频理解 → 情绪 JSON
```

## 主要功能

- **图片行为分析**：YOLO 检测 + Qwen 语义分析级联，输出标注图、检测明细表、统计摘要与行为分析报告。
- **音频叫声分析**：Qwen-Omni 直接理解猫叫音频，按标签体系输出情绪判定，内置 fighting / warning / mating 判别规则提示词，降低冲突叫声误判。
- **图片压缩管道**：发送 Qwen 前自动缩放（最长边 1024px）+ JPEG 压缩（质量 82），降低流量与 Token 开销，压缩前后指标在报告中标注。
- **会话隔离**：每个浏览器会话独立维护客户端 ID、日志与音频记录，音频文件落盘在按会话隔离的目录，7 天自动清理。
- **CH340 硬件收发测试**：可选 pyserial，支持串口刷新、原始 JSON 包收发、温度查询与舵机角度控制，便于对接喂食/环境监测下位机。
- **FRP 公网映射**：可选通过 frpc 将本地 Gradio 服务映射到公网（自动重试、PID 管理与退出清理），适合远程演示。
- **可观测性**：服务端环形日志缓冲 + 页面实时日志，请求各阶段（压缩、YOLO、Qwen 调用）均有日志。

## 技术方案

### 检测模型

- 基座：YOLO26n（基于仓库内 Ultralytics 源码训练，训练入口 `train.py`）。
- 类别：6 类猫咪行为（喝水、进食、玩耍、睡觉、呕吐、如厕）。
- 推理参数可在线调节：置信度阈值、NMS IoU 阈值、最大目标数、推理设备（cpu / cuda:0 / cuda:1）。

### 大模型级联

| 能力 | 默认模型 | 说明 |
|------|----------|------|
| 图片行为分析 | `qwen3.5-flash` | 接收 YOLO 检测摘要 + 压缩图片，输出行为分析 |
| 猫叫情绪分析 | `qwen3.5-omni-plus` | 直接音频理解，输出固定标签 JSON |
| ASR 转写 | 不使用 | 猫叫不做人类语音转写 |

接口为阿里云 DashScope OpenAI 兼容模式（`/compatible-mode/v1/chat/completions`），模型名、接口地址、提示词均可在页面中修改。

### 应用架构

- Web 框架：Gradio Blocks，Tab 页划分图片分析 / 音频分析 / 硬件测试。
- 模型管理：`ModelManager` 懒加载 + 线程锁，避免重复加载权重。
- API Key 管理：仅保存在当前会话内存，不落盘；支持页面输入或环境变量 `DASHSCOPE_API_KEY` / `QWEN_API_KEY`。

## 目录说明

```
.
├── pet_behavior_gradio_app.py   # Gradio 多模态应用入口
├── train.py                     # YOLO26n 行为检测训练入口
├── requirements.txt             # 项目依赖
├── ultralytics/                 # 仓库内 Ultralytics 框架源码（训练依赖）
└── runs/train/.../weights/      # 训练权重输出目录（.gitignore 排除）
```

## 环境要求与安装

建议 Windows 10/11、Python 3.10+；NVIDIA GPU 请按驱动安装匹配的 PyTorch。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

注意：`requirements.txt` 已将 Gradio 锁定在 `>=4.44.0,<6.0`。Gradio 6.0 移除了 `gr.update` 等接口，请勿升级。

## 快速开始

### 1. 训练行为检测模型

数据集（`猫咪行为数据集/cat_behavior_merged/data.yaml`）为私有数据，不随仓库提供。获得数据后：

```bash
python train.py
```

权重默认输出到 `runs/train/cat_behavior_yolo26n/weights/best.pt`，即应用默认加载路径。

### 2. 启动应用

```bash
python pet_behavior_gradio_app.py --weight <你的权重路径>
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--weight` | YOLO 权重路径（也可放在默认候选路径自动识别） |
| `--host` / `--port` | 监听地址与端口（默认 0.0.0.0:7861，端口占用自动顺延） |
| `--no-frp` | 禁用 frp 公网映射，仅本地访问 |

启动后浏览器自动打开。先在欢迎页填写阿里云 DashScope API Key 进入系统；不填写 Key 时只能使用 YOLO 检测与音频元信息功能。

### 3. 可选硬件与公网配置

- 串口功能需安装 `pyserial`；
- 公网映射需准备 `frp_windows/frp_0.68.1_windows_amd64/frpc.exe` 与环境变量 `FRP_SERVER_ADDR`、`FRP_SERVER_PORT`、`FRP_AUTH_TOKEN`、`FRP_REMOTE_PORT`、`FRP_PUBLIC_DOMAIN`。密钥只通过环境变量注入，禁止写入代码或提交到仓库。

## 商用与授权说明

- 本仓库根目录 LICENSE 为 AGPL-3.0 完整协议文本。对本项目代码进行复制、修改、再发布或通过网络提供服务时，应履行 AGPL-3.0 义务（保留版权与许可证声明、提供相应源代码与修改信息）。
- 仓库内 Ultralytics 源码遵循 AGPL-3.0；商业闭源集成需另行获得 Ultralytics 企业许可。
- 通义千问相关能力依赖阿里云 DashScope 服务，商用前请遵守阿里云百炼服务条款并完成费用与配额评估。
- 行为数据集与训练权重为私有资源，未经数据权利人书面授权不得公开分发原始数据或衍生模型。
- 如需闭源集成、SaaS 部署或设备销售等商业交付，请先与相关权利人签署单独的商业授权协议。

## 安全与免责声明

- API Key 建议单独申请专用子 Key 并设置额度限制，避免使用高权限主 Key。
- 音频与图片临时文件保存在本机 `.gradio/` 目录并按会话隔离，7 天自动清理；请勿把含隐私的现场素材提交到公开仓库。
- 本系统用于研究与演示，不构成兽医诊断建议；涉及宠物健康异常请以线下诊疗为准。

## 已知限制

- 音频分析依赖 Qwen-Omni 对猫叫的直接理解，短促或低信噪比音频的判定置信度有限。
- 串口硬件协议为项目自定义 JSON 格式，需要与下位机固件保持一致。

## 联系方式

项目仓库：[Real-time-Monitoring-and-Analysis-of-Pet-Behavior](https://github.com/catiseyeqaq/Real-time-Monitoring-and-Analysis-of-Pet-Behavior)
