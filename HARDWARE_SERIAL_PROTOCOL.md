# CH340 串口硬件对接说明

本文档给下位机硬件工程师使用，用于把开发板通过 CH340 / CH341 USB 串口模块接入当前本地宠物行为识别系统。

## 串口参数

| 参数         |                默认值 |
| ------------ | --------------------: |
| 串口芯片     |         CH340 / CH341 |
| Windows 端口 | 页面选择，例如 `COM3` |
| 波特率       |              `115200` |
| 数据位       |                   `8` |
| 校验位       |                `None` |
| 停止位       |                   `1` |
| 流控         |                    无 |
| 编码         |            UTF-8 文本 |
| 帧结束       |                  `\n` |
| 默认读取等待 |                `0.5s` |

开发板建议按“收到一行 JSON 后解析并返回一行 JSON”的方式实现。

## 上位机命令

### 1. Ping 测试

上位机发送：

```json
{ "cmd": "ping" }
```

下位机建议返回：

```json
{ "ok": true, "cmd": "ping", "message": "pong" }
```

### 2. 温度读取

上位机发送：

```json
{ "cmd": "temperature", "action": "read" }
```

下位机建议返回：

```json
{ "ok": true, "cmd": "temperature", "temperature_c": 22.0 }
```

上位机可解析字段：`temperature`、`temperature_c`、`temp`、`value`。

### 3. 湿度读取

上位机发送：

```json
{ "cmd": "humidity", "action": "read" }
```

下位机建议返回：

```json
{ "ok": true, "cmd": "humidity", "humidity_percent": 45.0 }
```

上位机可解析字段：`humidity`、`humidity_percent`、`humidity_rh`、`rh`、`value`。

单位建议为 `%RH`。

### 4. 重量读取

上位机发送：

```json
{ "cmd": "weight", "action": "read" }
```

下位机建议返回：

```json
{ "ok": true, "cmd": "weight", "weight_kg": 5.0 }
```

上位机可解析字段：`weight`、`weight_kg`、`kg`、`value`。

单位建议为 `kg`。

### 5. 舵机控制

上位机发送：

```json
{ "cmd": "servo", "id": 1, "angle": 90 }
```

字段说明：

| 字段    | 类型   | 说明                         |
| ------- | ------ | ---------------------------- |
| `cmd`   | string | 固定为 `servo`               |
| `id`    | int    | 舵机编号                     |
| `angle` | int    | 舵机角度，页面范围 `0 - 180` |

下位机建议返回：

```json
{ "ok": true, "cmd": "servo", "id": 1, "angle": 90 }
```

### 6. 温度报警

温度超过页面阈值时，上位机发送：

```json
{ "cmd": "alarm", "level": "high" }
```

下位机建议返回：

```json
{ "ok": true, "cmd": "alarm", "level": "high" }
```

## 图片识别后的环境显示

每次上传图片或使用摄像头识别后，网页会同步显示：

- 当前温度
- 当前湿度
- 喂食器食物重量
- 喂食器状态
- 舵机 ID
- 舵机角度参数
- 本轮累计投放估算

如果 CH340 未连接，系统会在后台静默使用默认室内环境值，前端只显示当前状态：

```text
当前温度: 22.0 °C
当前湿度: 45.0 %RH
喂食器食物重量: 5.000 kg
喂食器状态: 满载
舵机 ID: 1
舵机角度参数: 90°
```

这些模拟值按东北普通室内环境设置，用于无硬件时演示。

## 本地联动逻辑

默认参数：

| 参数             |                              默认值 |
| ---------------- | ----------------------------------: |
| 单轮投放上限     |                            `5.0 kg` |
| 吃食触发重量下降 |                           `0.02 kg` |
| 温度报警阈值     |                           `35.0 °C` |
| 自动投放控制码   | `{"cmd":"servo","id":1,"angle":90}` |
| 报警控制码       |    `{"cmd":"alarm","level":"high"}` |

联动顺序：

1. 读取重量。
2. 读取温度。
3. 读取湿度。
4. 温度过高则报警，不投放。
5. 第一次运行只记录初始重量。
6. 后续检测到重量下降超过阈值，认为猫吃了食物。
7. 未超过 `5kg` 上限则发送舵机投放控制码。
8. 超过 `5kg` 上限则阻止舵机并记录报警。

## Arduino / ESP32 伪代码

```cpp
void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line.indexOf("\"cmd\":\"ping\"") >= 0) {
    Serial.println("{\"ok\":true,\"cmd\":\"ping\",\"message\":\"pong\"}");
  } else if (line.indexOf("\"cmd\":\"temperature\"") >= 0) {
    Serial.println("{\"ok\":true,\"cmd\":\"temperature\",\"temperature_c\":22.0}");
  } else if (line.indexOf("\"cmd\":\"humidity\"") >= 0) {
    Serial.println("{\"ok\":true,\"cmd\":\"humidity\",\"humidity_percent\":45.0}");
  } else if (line.indexOf("\"cmd\":\"weight\"") >= 0) {
    Serial.println("{\"ok\":true,\"cmd\":\"weight\",\"weight_kg\":5.0}");
  } else if (line.indexOf("\"cmd\":\"servo\"") >= 0) {
    Serial.println("{\"ok\":true,\"cmd\":\"servo\",\"id\":1,\"angle\":90}");
  } else if (line.indexOf("\"cmd\":\"alarm\"") >= 0) {
    Serial.println("{\"ok\":true,\"cmd\":\"alarm\",\"level\":\"high\"}");
  } else {
    Serial.println("{\"ok\":false,\"error\":\"unknown_cmd\"}");
  }
}
```

实际工程中建议使用 ArduinoJson 等 JSON 库解析。

## 调试步骤

1. 插入 CH340。
2. 设备管理器确认端口，例如 `COM3`。
3. 打开本地网页 `http://127.0.0.1:7861/`。
4. 进入 `CH340 硬件收发测试`。
5. 点击 `刷新串口`。
6. 发送 `{"cmd":"ping"}`，确认返回 `pong`。
7. 分别点击 `读取温度`、`读取湿度`。
8. 进入 `本地联动控制`，清空手动温度/湿度/重量，让系统从开发板读取真实数据。
