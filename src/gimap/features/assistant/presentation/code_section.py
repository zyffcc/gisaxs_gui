"""Settings ▸ Assistant ▸ Claude Code: which Claude Code, which model, and the sign-in."""

from __future__ import annotations

import re
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from ..application import BILLING_SUBSCRIPTION, CLAUDE_CODE_MODELS, LlmError
from src.gimap.app.presentation.i18n import tr, trf

from . import preferences
from .services import AssistantServices

SIGN_IN_HINT = (
    "Sign In… opens a window running “claude auth login”: choose your Claude account "
    "(Pro or Max). GIMaP starts Claude Code without any API key, so runs count against your "
    "plan's usage limits instead of being billed per token. Claude Code gets only the GIMaP "
    "tools — no shell, files or web. Meant for your own use: someone else needs their own "
    "sign-in or an API key."
)


ROW_SPACING = 6
"""Between the widgets of a row inside a form (a field and its buttons). A row layout nested in a form
inherits no spacing, which the style draws as 0 px: the buttons would touch the field."""
NOT_FOUND_AT = re.compile(r"Claude Code was not found at (?P<cli>.+)\.")
"""The adapter's sentence for a program path that does not exist (``infrastructure/claude_code_agent.py``)."""


def code_error_text(error) -> str:
    """Why Claude Code cannot run, in the interface language. The adapter writes English (the run record reads
    it); its two “not found” sentences are table entries — the install hint as a whole (``INSTALL_HINT`` of the
    adapter is the key), “… not found at <path>.” as a template with the path as it is. Any other error stays
    as the program gave it."""
    text = str(error)
    found = NOT_FOUND_AT.fullmatch(text)
    return trf("Claude Code was not found at {cli}.", cli=found["cli"]) if found else tr(text)


def describe_code_status(info: dict, model: str = "") -> tuple[bool, str]:
    """Whether Claude Code can run, and a sentence about its version and account."""
    if info.get("error"):
        return False, code_error_text(info["error"])
    version = str(info.get("version") or "").replace("(Claude Code)", "").strip()
    name = f"Claude Code {version}".strip()
    if not info.get("logged_in"):
        return False, tr("{name} is not signed in: use Sign In… in Settings ▸ Assistant with your Claude account.").format(
            name=name)
    shown_model = model or tr("Claude Code default")
    if info.get("billing") == BILLING_SUBSCRIPTION:
        plan = f" ({info['subscription']})" if info.get("subscription") else ""
        return True, tr("{name} · signed in with your Claude plan{plan} · model {model}.").format(
            name=name, plan=plan, model=shown_model)
    method = info.get("auth_method") or tr("an API account")
    return True, tr("{name} · signed in with {method}: billed per token, not your Claude plan · model {model}.").format(
        name=name, method=method, model=shown_model)


class ClaudeCodeSection(QGroupBox):
    def __init__(self, settings, services: AssistantServices, tasks, parent: QWidget = None):
        super().__init__(tr("Claude Code — your Claude plan"), parent)
        self.setObjectName("assistantCodeSection")
        self.settings = settings
        self.services = services
        self.tasks = tasks
        form = QFormLayout(self)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)
        cli_row = QHBoxLayout()
        cli_row.setSpacing(ROW_SPACING)
        self.cli_edit = QLineEdit(str(preferences.read(settings, "code_cli")), self)
        self.cli_edit.setObjectName("assistantCodeCliEdit")
        self.browse_button = QPushButton(tr("Browse…"), self)
        cli_row.addWidget(self.cli_edit, 1)
        cli_row.addWidget(self.browse_button)
        form.addRow(tr("Program"), cli_row)
        self.code_model_combo = QComboBox(self)
        self.code_model_combo.setObjectName("assistantCodeModelCombo")
        self.code_model_combo.setEditable(True)
        self.code_model_combo.addItems(CLAUDE_CODE_MODELS)
        self.code_model_combo.lineEdit().setPlaceholderText(tr("Claude Code default"))
        self.code_model_combo.setEditText(str(preferences.read(settings, "code_model")))
        form.addRow(tr("Model"), self.code_model_combo)
        account_row = QHBoxLayout()
        account_row.setSpacing(ROW_SPACING)
        self.account_label = QLabel(tr("Not checked yet."), self)
        self.account_label.setObjectName("assistantCodeAccount")
        self.account_label.setWordWrap(True)
        self.account_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # right under the buttons, whatever its row's height
        self.check_button = QPushButton(tr("Check"), self)
        self.check_button.setObjectName("assistantCodeCheckButton")
        self.login_button = QPushButton(tr("Sign In…"), self)
        self.login_button.setObjectName("assistantCodeLoginButton")
        account_row.addWidget(self.check_button)
        account_row.addWidget(self.login_button)
        account_row.addStretch(1)
        form.addRow(tr("Account"), account_row)
        # The answer below the buttons, in a row of its own: beside them (inside the row's layout) the form gave
        # it one line, so a longer answer — where to find Claude Code — was cut off in the middle. A row that
        # spans the form gets the height its wrapped text needs at any width.
        form.addRow(self.account_label)
        hint = QLabel(tr(SIGN_IN_HINT), self)
        hint.setWordWrap(True)
        hint.setProperty("gimapRole", "muted")
        form.addRow(hint)
        self._show_found()
        self.cli_edit.editingFinished.connect(self._remember_cli)
        self.browse_button.clicked.connect(self._browse)
        self.code_model_combo.activated.connect(lambda _index: self._remember_model())
        self.code_model_combo.lineEdit().editingFinished.connect(self._remember_model)
        self.check_button.clicked.connect(self.check)
        self.login_button.clicked.connect(self._login)

    def cli(self) -> str:
        return self.cli_edit.text().strip()

    def _show_found(self) -> None:
        found = self.services.find_cli("")
        self.cli_edit.setPlaceholderText(
            tr("found: {path}").format(path=found) if found else tr("not found — install Claude Code or browse to claude.exe")
        )

    def _remember_cli(self) -> None:
        preferences.write(self.settings, "code_cli", self.cli())

    def _remember_model(self) -> None:
        preferences.write(self.settings, "code_model", self.code_model_combo.currentText().strip())

    def _browse(self) -> None:
        start = self.cli() or self.services.find_cli("") or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Choose Claude Code"), start, "Claude Code (claude.exe claude claude.cmd);;All files (*)"
        )
        if path:
            self.cli_edit.setText(path)
            self._remember_cli()

    def check(self) -> None:
        self._remember_cli()
        self._remember_model()
        cli = self.cli()
        self.check_button.setEnabled(False)
        self.account_label.setText(tr("Checking…"))
        self.tasks.submit(
            "assistant-code-status",
            lambda: self.services.code_status(cli),
            on_done=self._checked,
            on_error=lambda message, _details: self._checked({"error": message}),
        )

    def _checked(self, info: dict) -> None:
        self.check_button.setEnabled(True)
        _ready, text = describe_code_status(info, self.code_model_combo.currentText().strip())
        self.account_label.setText(text)

    def _login(self) -> None:
        self._remember_cli()
        try:
            self.services.code_login(self.cli())
        except (LlmError, OSError) as exc:
            QMessageBox.warning(self, tr("Sign In to Claude Code"), code_error_text(getattr(exc, "message", None) or str(exc)))
            return
        self.account_label.setText(tr("Finish signing in in the window that opened, then press Check."))


__all__ = ["ClaudeCodeSection", "ROW_SPACING", "SIGN_IN_HINT", "code_error_text", "describe_code_status"]
