"""Offline documentation shell: a persistent outline around the complete rendered document.

The Markdown renderer owns heading IDs and file links. This module only adds navigation, styles
and local search; it never changes the scientific text. Assets are embedded so file URLs work offline.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path

from .i18n import translate

_HEADING = re.compile(r'<h([1-6])\b[^>]*\bid="([^"]*)"[^>]*>(.*?)</h\1>', re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_ASSETS = Path(__file__).parent


@dataclass
class _Entry:
    level: int
    anchor: str
    text: str
    children: list[_Entry] = field(default_factory=list)


def _plain(markup: str) -> str:
    return " ".join(html.unescape(_TAG.sub("", markup)).split())


def _outline(body: str, contents: str) -> str:
    root = _Entry(0, "", "")
    stack = [root]
    for index, match in enumerate(_HEADING.finditer(body)):
        level, anchor, inner = match.groups()
        label = _plain(inner)
        if (index == 0 and level == "1") or int(level) > 4 or label in {"Contents", contents}:
            continue
        entry = _Entry(int(level), html.unescape(anchor), label)
        while stack[-1].level >= entry.level:
            stack.pop()
        stack[-1].children.append(entry)
        stack.append(entry)

    def items(entries: list[_Entry]) -> str:
        result = []
        for entry in entries:
            link = f'<a href="#{html.escape(entry.anchor, quote=True)}">{html.escape(entry.text)}</a>'
            if entry.children:
                link = f'<details><summary>{link}</summary><ul>{items(entry.children)}</ul></details>'
            result.append(f"<li>{link}</li>")
        return "".join(result)

    return f'<ul class="outline">{items(root.children)}</ul>'


def _fold_contents(body: str, label: str) -> str:
    """Keep the original table of contents available without repeating it at full length on arrival."""
    headings = list(_HEADING.finditer(body))
    for index, match in enumerate(headings):
        if _plain(match.group(3)) not in {"Contents", label}:
            continue
        end = next((item.start() for item in headings[index + 1:] if int(item.group(1)) <= int(match.group(1))), len(body))
        section = body[match.start():end]
        return (body[:match.start()] + f'<details class="inline-contents"><summary>{html.escape(label)}</summary>'
                + section + "</details>" + body[end:])
    return body


def documentation_page(body: str, title: str, head: str = "") -> str:
    """Wrap already rendered HTML; navigation follows the document's language, including linked pages."""
    first = _HEADING.search(body)
    language = "zh" if re.search(r"[\u3400-\u9fff]", _plain(first.group(3)) if first else title) else "en"

    def text(english: str) -> str:
        return html.escape(translate(english, language) or english, quote=True)

    contents = translate("Contents", language) or "Contents"
    outline = _outline(body, contents)
    body = _fold_contents(body, contents)
    top = first.group(2) if first else "help-content"
    styles = (_ASSETS / "help_page.css").read_text(encoding="utf-8")
    script = (_ASSETS / "help_page.js").read_text(encoding="utf-8")
    return f'''<!doctype html>
<html lang="{language}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>{head}<style>{styles}</style></head>
<body>
<a class="skip-link" href="#help-content">{text("Skip to content")}</a>
<header class="help-header">
  <a class="brand" href="#{top}">GIMaP <span>{text("Documentation")}</span></a>
  <span class="document-name">{html.escape(title)}</span>
  <button id="contents-toggle" type="button" aria-controls="help-sidebar" aria-expanded="false">{text("Contents")}</button>
</header>
<div class="help-layout">
<aside id="help-sidebar" aria-label="{text("Document navigation")}">
  <div class="search-box" hidden>
    <label for="help-search">{text("Search this document")}</label>
    <input id="help-search" type="search" placeholder="{text("Search headings and text…")}" autocomplete="off">
    <p id="search-status" role="status" aria-live="polite" data-count="{text("{count} matching sections")}">
      {text("Search headings and text. Esc clears the search.")}</p>
  </div>
  <nav id="help-outline" aria-label="{text("Contents")}">
    <a class="overview-link" href="#{top}">{text("Overview")}</a>
    {outline}
  </nav>
  <nav id="help-results" aria-label="{text("Search results")}" hidden></nav>
</aside>
<main id="help-content" tabindex="-1">{body}
  <footer><a href="#{top}">{text("Back to top")}</a></footer>
</main>
</div>
<script>{script}</script></body></html>'''
