"""Assistant composition root: Claude, the key store and the result files behind the panel."""

from __future__ import annotations

from typing import Callable

from .application import provider
from .infrastructure import (
    AnthropicAssistantLlm,
    ApiKeyStore,
    ClaudeCodeAgent,
    JsonResultStore,
    LocalFileExplorer,
    OpenAICompatibleLlm,
    ProviderKeyStore,
    estimated_cost,
    find_claude_cli,
)
from .presentation import AssistantController, AssistantServices, GuidedAnalysis
from .presentation.services import ProviderServices


def create_provider_services(data_dir) -> ProviderServices:
    """Other AI providers through the OpenAI-compatible adapter, keys kept per provider."""
    keys = ProviderKeyStore(data_dir)

    def create(key: str, model: str, url: str) -> OpenAICompatibleLlm:
        preset = provider(key)
        return OpenAICompatibleLlm(preset=preset, model=model, api_key=keys.load(key, preset.env_key), base_url=url or None)

    return ProviderServices(
        create=create,
        source=lambda key: keys.source(key, provider(key).env_key),
        has_saved_key=lambda key: keys.saved(key) is not None,
        save_key=keys.save,
        delete_key=keys.delete,
        list_models=lambda key, model, url: create(key, model or "list", url).list_models(),
        check=lambda key, model, url: create(key, model, url).check(),
    )


def create_assistant_services(data_dir, calibrator=None, fitter=None) -> AssistantServices:
    keys = ApiKeyStore(data_dir)
    store = JsonResultStore(data_dir)

    def create_llm(model: str, effort: str) -> AnthropicAssistantLlm:
        return AnthropicAssistantLlm(model=model, effort=effort, api_key=keys.load())

    return AssistantServices(
        create_llm=create_llm,
        credentials=keys.source,
        has_saved_key=lambda: keys.load() is not None,
        save_key=keys.save,
        delete_key=keys.delete,
        check=lambda model, effort: create_llm(model, effort).check(),
        store=store,
        save_text=store.save_text,
        save_run=store.save_run,
        cost=estimated_cost,
        create_agent=lambda cli, model, effort: ClaudeCodeAgent(cli=cli or None, model=model, effort=effort),
        find_cli=find_claude_cli,
        code_status=lambda cli: ClaudeCodeAgent(cli=cli or None).status(),
        code_login=lambda cli: ClaudeCodeAgent(cli=cli or None).open_login(),
        explorer=LocalFileExplorer(),
        calibrator=calibrator,
        fitter=fitter,
        providers=create_provider_services(data_dir),
    )


def create_guided_analysis(
    app_context, *, automation: Callable[[], object], calibrator=None, fitter=None, parent=None,
) -> GuidedAnalysis:
    """The automatic analysis (no model needed) on the frame in Analyze: its controls and results panel."""
    store = JsonResultStore(getattr(app_context, "data_dir", None))
    return GuidedAnalysis(
        automation, explorer=LocalFileExplorer(), calibrator=calibrator, fitter=fitter, save_text=store.save_text,
        parent=parent,
    )


def create_assistant_controller(
    window,
    app_context,
    *,
    automation: Callable[[], object],
    show_analyze: Callable[[], None],
    calibrator=None,
    fitter=None,
) -> AssistantController:
    """The controller of “Process with Claude”; ``calibrator`` fits standards, ``fitter`` GISAXS cuts (optional)."""
    return AssistantController(
        window,
        create_assistant_services(getattr(app_context, "data_dir", None), calibrator, fitter),
        settings=app_context.settings,
        automation=automation,
        show_analyze=show_analyze,
    )


__all__ = ["create_assistant_controller", "create_assistant_services", "create_guided_analysis", "create_provider_services"]
