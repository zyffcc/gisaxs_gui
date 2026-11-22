"""Disclosure before an assistant request, and where its API keys are saved."""

ASSISTANT_SERVICE_ZH = {
    "Allow requests": "允许发送请求",
    "Share data with the AI service?": "允许向 AI 服务发送数据吗？",
    "The AI assistant sends your instructions, frame information and analysis results to {service}. "
    "If you enable image previews, it also sends an image of the frame. Allow requests to this service?":
        "AI 助手会将你的指令、帧信息和分析结果发送给 {service}。如果启用图像预览，还会发送帧的预览图像。允许向该服务发送请求吗？",
    "GIMaP saves keys in the operating system’s credential store when available. Otherwise it saves them in plain text in the user data folder.":
        "GIMaP 优先将密钥保存在操作系统的凭据存储中。如果没有可用的凭据存储，密钥将以明文保存在用户数据文件夹中。",
}
