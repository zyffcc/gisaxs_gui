"""Deprecated import path for the application runtime."""

from .runtime import ApplicationRuntime

MainController = ApplicationRuntime

__all__ = ["ApplicationRuntime", "MainController"]
