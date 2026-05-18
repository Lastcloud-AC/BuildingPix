"""
BuildingPix - API 配置模板
============================
复制此文件为 config.py，填入你自己的 API Key。
config.py 已被 .gitignore 排除，不会提交到 GitHub。

注意: URL 填写完整地址，包括路径，如 https://api.302.ai/v1/chat/completions
"""

# ═══ VLM 视觉模型（Phase 1/2 分析图片）═══
# 必须是支持图片输入的视觉模型!
VLM_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
VLM_API_KEY = "your-dashscope-api-key"
VLM_MODEL = "qwen3.5-omni-plus"

# ═══ 图像生成模型（Phase 1 生成正交图）═══
IMAGE_GEN_API_URL = "https://api.302.ai/v1/images/generations"
IMAGE_GEN_API_KEY = "your-302ai-api-key"
IMAGE_GEN_MODEL = "dall-e-3"

# ═══ 图像编辑模型（Phase 3 图生图，DashScope Qwen）═══
# 使用 DashScope 多模态生成接口（qwen-image-2.0-pro）
IMAGE_EDIT_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
IMAGE_EDIT_API_KEY = "your-dashscope-api-key"
IMAGE_EDIT_MODEL = "qwen-image-2.0-pro"

# ═══ 质检 VLM（Phase 2.5，可选单独配置）═══
# 如果留空，会自动复用上面的 VLM 配置
CHECKER_VLM_API_URL = "https://api.302.ai/v1/chat/completions"
CHECKER_VLM_API_KEY = "your-302ai-api-key"
CHECKER_VLM_MODEL = "gemini-3.1-flash-lite-preview"
