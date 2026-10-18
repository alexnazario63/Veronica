"""Pacote services.api da Veronica."""

from services.api.app import app, create_app
from services.api.services_hub import VeronicaPlatformHub

__all__ = ["app", "create_app", "VeronicaPlatformHub"]
