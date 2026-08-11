# 智能宠物行为识别与环境联动系统

本文件与仓库首页 [README.md](README.md) 保持一致，面向中文用户说明项目的真实功能、运行方式、硬件协议和开源/商用边界。

## 项目定位

这是一个本地运行的宠物照护 Demo，不是通用视觉模型展示项目。它把以下能力整合到一个 Gradio Web 界面中：

- 图片或摄像头行为分析：进食、饮水、玩耍、睡眠、如厕、呕吐等行为线索。
- 可选的 Qwen 多模态分析：输出行为判断、健康风险和照护建议。
- 猫叫音频理解：输出情绪标签、置信度、紧急程度和主人建议。
- CH340/CH341 串口通信：温度、湿度、重量读取，原始 JSON 发包和舵机控制。
- 本地联动控制：根据重量变化触发投喂，根据温度阈值触发报警。

## 快速开始

```powershell
python -m pip install -r requirements.txt
python .\start_app.py
```

默认访问 `http://127.0.0.1:7861/`。首次进入时创建本地账号；需要云端语义分析时再填写 DashScope API Key。图片行为分析需要通过 `--weight` 提供与项目类别匹配的本地模型权重：

```powershell
python .\pet_behavior_gradio_app.py --weight .\path\to\your\best.pt
```

完整功能、目录、协议和故障排查请查看 [README.md](README.md) 与 [HARDWARE_SERIAL_PROTOCOL.md](HARDWARE_SERIAL_PROTOCOL.md)。

## 许可证与商用

项目代码按 [AGPL-3.0](LICENSE) 发布。商业使用并非自动禁止，但商业使用者必须履行 AGPL-3.0 的源代码、版权、许可证和网络服务交互义务，并分别确认第三方依赖、模型、数据集、API 服务和硬件协议的授权范围。闭源或白标部署前请取得必要的商业授权并完成法律审查，详见 [COMMERCIAL_USE.md](COMMERCIAL_USE.md)。
