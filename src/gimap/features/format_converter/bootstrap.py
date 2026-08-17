"""Format Converter feature 的 composition root。"""

from src.gimap.app import AppContext

from .infrastructure.adapters import LocalConversionExecutor, LocalSourceRepository
from .presentation.view_model import FormatConverterViewModel


def create_format_converter_view_model(app_context: AppContext) -> FormatConverterViewModel:
    repository = LocalSourceRepository()
    executor = LocalConversionExecutor(repository)
    return FormatConverterViewModel(
        app_context=app_context,
        repository=repository,
        executor=executor,
    )
