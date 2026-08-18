"""Format Converter 的请求、结果和纯格式规则。"""

from .models import (
    ConversionJob,
    ConversionOptions,
    ConversionRequest,
    ConversionResult,
    InputSource,
)
from .rules import (
    build_jobs,
    compact_frame_summary,
    is_supported_input_path,
    output_may_lose_float_values,
    output_naming_summary,
    parse_custom_frames,
    render_output_example,
    select_frame_indices,
    select_source_frame_indices,
    visible_output_formats,
)

__all__ = [
    "ConversionJob",
    "ConversionOptions",
    "ConversionRequest",
    "ConversionResult",
    "InputSource",
    "build_jobs",
    "compact_frame_summary",
    "is_supported_input_path",
    "output_may_lose_float_values",
    "output_naming_summary",
    "parse_custom_frames",
    "render_output_example",
    "select_frame_indices",
    "select_source_frame_indices",
    "visible_output_formats",
]
