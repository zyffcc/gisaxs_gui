"""Format Converter 的请求、结果和纯格式规则。"""

from .models import (
    ConversionJob,
    ConversionOptions,
    ConversionRequest,
    ConversionResult,
    InputSource,
)
from .rules import build_jobs, compact_frame_summary, parse_custom_frames

__all__ = [
    "ConversionJob",
    "ConversionOptions",
    "ConversionRequest",
    "ConversionResult",
    "InputSource",
    "build_jobs",
    "compact_frame_summary",
    "parse_custom_frames",
]
