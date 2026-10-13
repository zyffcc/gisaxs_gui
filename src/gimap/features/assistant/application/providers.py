"""AI providers the assistant can use besides Claude: presets for OpenAI-compatible services.

Most mainstream model APIs speak the OpenAI chat-completions protocol with tool
calling: OpenAI itself, DeepSeek, Qwen (Alibaba Cloud Model Studio), Kimi,
Zhipu GLM, Google Gemini's compatible endpoint, OpenRouter, SiliconFlow, and
local servers (Ollama, LM Studio, vLLM).  One adapter serves them all; a preset
only fills in the address, the key's environment variable and a few model
suggestions.  Model names change often, so the suggestions are a starting
point and the settings can fetch the provider's own list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

KIND_OPENAI = "openai"
"""OpenAI chat completions at ``base_url``."""
KIND_AZURE = "azure"
"""Azure OpenAI: ``base_url`` is the resource endpoint, the model is the deployment name."""


@dataclass(frozen=True)
class ProviderPreset:
    key: str
    name: str
    """English, as the run record and the adapter's messages name the provider; the interface shows its
    translation (the Chinese table holds the names Chinese users know them by)."""
    base_url: str
    """Default address ("" when the person has to enter one)."""
    env_key: str = ""
    """Environment variable holding the key ("" when none is needed or known)."""
    needs_key: bool = True
    models: tuple[str, ...] = ()
    """Suggestions only; the provider's own list is fetched on demand."""
    kind: str = KIND_OPENAI
    group: str = "api"
    """``api`` (hosted), ``local`` (this computer or the lab's server)."""
    notes: str = ""


PROVIDERS: tuple[ProviderPreset, ...] = (
    ProviderPreset("openai", "OpenAI", "https://api.openai.com/v1", "OPENAI_API_KEY",
                   models=("gpt-5", "gpt-5-mini", "gpt-4.1")),
    ProviderPreset("deepseek", "DeepSeek", "https://api.deepseek.com", "DEEPSEEK_API_KEY",
                   models=("deepseek-chat", "deepseek-reasoner")),
    ProviderPreset("qwen", "Qwen (Alibaba Cloud Model Studio)", "https://dashscope.aliyuncs.com/compatible-mode/v1",
                   "DASHSCOPE_API_KEY", models=("qwen-plus", "qwen-max", "qwen-turbo")),
    ProviderPreset("moonshot", "Kimi (Moonshot AI)", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY",
                   models=("moonshot-v1-32k",)),
    ProviderPreset("zhipu", "Zhipu GLM", "https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY",
                   models=("glm-4.5", "glm-4-plus")),
    ProviderPreset("gemini", "Google Gemini (OpenAI-compatible endpoint)",
                   "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY",
                   models=("gemini-2.5-pro", "gemini-2.5-flash")),
    ProviderPreset("openrouter", "OpenRouter (many models)", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    ProviderPreset("siliconflow", "SiliconFlow", "https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY",
                   models=("deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct")),
    ProviderPreset("azure", "Azure OpenAI", "", "AZURE_OPENAI_API_KEY", kind=KIND_AZURE,
                   notes="Address: https://<resource>.openai.azure.com; model: the deployment name."),
    ProviderPreset("ollama", "Ollama (this computer)", "http://localhost:11434/v1", needs_key=False, group="local",
                   models=("qwen2.5:14b", "llama3.1:8b"),
                   notes="Runs on this computer; pick a model that supports tool calling."),
    ProviderPreset("lmstudio", "LM Studio (this computer)", "http://localhost:1234/v1", needs_key=False, group="local"),
    ProviderPreset("custom", "Custom OpenAI-compatible service", "", needs_key=False, group="local",
                   notes="vLLM, a lab server or any service that speaks the OpenAI chat API with tools."),
)
PROVIDER_KEYS = tuple(preset.key for preset in PROVIDERS)
AZURE_API_VERSION = "2024-10-21"


def provider(key: str) -> ProviderPreset:
    """The preset for ``key`` (the custom preset for anything unknown)."""
    return next((preset for preset in PROVIDERS if preset.key == key), PROVIDERS[-1])


def provider_url(preset: ProviderPreset, override: Optional[str]) -> str:
    """The address to use: the person's own address when given, else the preset's."""
    text = (override or "").strip()
    return text or preset.base_url


__all__ = [
    "AZURE_API_VERSION",
    "KIND_AZURE",
    "KIND_OPENAI",
    "PROVIDERS",
    "PROVIDER_KEYS",
    "ProviderPreset",
    "provider",
    "provider_url",
]
