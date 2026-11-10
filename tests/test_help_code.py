"""Equations, code and scientific notation survive the offline help renderer verbatim."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest

from src.gimap.app.presentation.help_documents import render_markdown

ROOT = Path(__file__).resolve().parents[1]
_BLOCK = re.compile(r'<pre class="help-code"[^>]*><code>(.*?)</code></pre>', re.S)


def _code_blocks(page: str) -> list[str]:
    return [html.unescape(block) for block in _BLOCK.findall(page)]


@pytest.mark.parametrize("name", ["Physics_Manual.md", "Physics_Manual_zh.md", "User_Manual.md", "User_Manual_zh.md"])
def test_every_manual_equation_and_command_block_keeps_its_literal_text(name):
    source = ROOT / "docs" / name
    markdown = source.read_text(encoding="utf-8")
    expected = []
    for indent, body in re.findall(r"^([ \t]*)```[^\n]*\n(.*?)^\1```[ \t]*$", markdown, re.M | re.S):
        # CommonMark removes up to the opening fence's indentation from each content line.
        lines = [line[min(len(indent), len(line) - len(line.lstrip(' '))):] for line in body.splitlines(keepends=True)]
        expected.append("".join(lines).rstrip("\n") + "\n")
    assert expected
    assert _code_blocks(render_markdown(markdown, source.stem, source.parent)) == expected


def test_formula_alignment_and_literal_html_survive_without_turning_into_markup(tmp_path):
    equation = "q_x = k · X / L_p                 (2.5)\n  q_z = k · Z / L_p               (2.6)\n\nI < 0 & σ ≥ 0; <sample>\n"
    page = render_markdown(f"# Formula\n\n```text\n{equation}```\n\n## After\n\nStill here.", "Formula", tmp_path)
    assert _code_blocks(page) == [equation]
    assert "&lt;sample&gt;" in page and "Still here." in page
    assert '<h2 id="after"' in page
    assert "GIMaPCodeLiteral" not in page


def test_inline_symbols_paths_and_code_in_links_keep_code_semantics(tmp_path):
    page = render_markdown(
        "# Code `q_x`\n\nUse `α_i`, `λ²` and `array[i] < 0`.\n\n"
        "[Read `manual_name.md`](manual_name.md) and `` literal `backticks` ``.\n\n"
        "| Name | Meaning |\n|---|---|\n| `p_x` | pixel size |\n", "Code", tmp_path,
    )
    for literal in ("q_x", "α_i", "λ²", "array[i] < 0", "manual_name.md", "literal `backticks`", "p_x"):
        assert f"<code>{html.escape(literal)}</code>" in page, re.findall(
            r".{0,150}" + re.escape(html.escape(literal)) + r".{0,150}", page)
    assert '<h1 id="code-q_x"' in page


def test_indented_code_and_a_fenced_block_in_a_list_keep_indentation(tmp_path):
    page = render_markdown(
        "# Code\n\n    first\n      second\n\n1. Example\n\n   ```text\n   a = 1\n     b = 2\n   ```\n\nAfter.",
        "Code", tmp_path,
    )
    assert _code_blocks(page) == ["first\n  second\n", "a = 1\n  b = 2\n"]


def test_fenced_literals_keep_spaces_and_separate_adjacent_blocks(tmp_path):
    page = render_markdown("```\nfirst\n                  second\n```\n\n```\nthird\n```", "Code", tmp_path)
    assert _code_blocks(page) == ["first\n                  second\n", "third\n"]
    assert "GIMaPFence" not in page
    assert _code_blocks(render_markdown("```", "Code", tmp_path)) == ["\n"]


def test_old_qt_explicit_code_font_without_fixed_pitch_keeps_semantics():
    from PyQt5.QtGui import QFontDatabase, QTextCharFormat, QTextCursor, QTextDocument
    from src.gimap.app.presentation.help_code import document_html
    document = QTextDocument()
    document.setPlainText("q_x")
    cursor = QTextCursor(document)
    cursor.select(QTextCursor.Document)
    format = QTextCharFormat()
    format.setFontFamily(QFontDatabase.systemFont(QFontDatabase.FixedFont).family())
    format.setFontFixedPitch(False)
    cursor.mergeCharFormat(format)
    assert "<code>q_x</code>" in document_html(document)
