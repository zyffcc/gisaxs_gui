"""Compose focused insitu setup bindings."""

from .insitu_dialog import InsituDialogMixin
from .insitu_watch_settings import InsituWatchSettingsMixin


class InsituSetupMixin(InsituDialogMixin, InsituWatchSettingsMixin):
    """Compatibility composition for focused insitu setup bindings."""

    pass
