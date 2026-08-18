from src.gimap.app.menu_manager import MenuManager
from src.gimap.app.presentation.menu_manager import MenuManager as PresentationMenuManager
from src.gimap.integrations.state import InMemorySettingsRepository
from ui.menu_manager import MenuManager as LegacyMenuManager


def test_legacy_menu_manager_path_reexports_application_owner() -> None:
    assert LegacyMenuManager is MenuManager
    assert issubclass(MenuManager, PresentationMenuManager)


def test_menu_manager_requires_explicit_settings_repository() -> None:
    settings = InMemorySettingsRepository({"beam": {"wavelength": 0.015}})
    manager = MenuManager(object(), settings=settings)

    assert manager.settings is settings
