"""The documents of the Help menu, rendered as web pages and opened in the default browser.

Help ▸ User Manual, Physics and Methods and Third-party Notices read a Markdown file of the program
folder, let Qt turn it into HTML (``QTextDocument.setMarkdown`` → ``toHtml``), write the page to the
temporary folder and hand it to the browser (``QDesktopServices.openUrl``). Two things make Qt's HTML a
page that works:

* before ``setMarkdown``, every ``<`` outside code spans and fenced code blocks is escaped unless it
  starts an autolink (``<https://…>``): Qt reads a bare ``<placeholder>`` as raw HTML, and every
  paragraph and heading after it comes out empty;
* after ``toHtml``, every heading gets the ``id`` that GitHub gives it (the slug of its text), so that a
  Contents link ``[…](#3-the-window)`` jumps to it, and relative links and images point at the files
  beside the document (absolute ``file:`` URLs: a ``<base>`` would send ``#…`` links to the folder).

A link to another Markdown document (the User Manual to ``Physics_Manual.md#…``, a link to
``CHANGELOG.md`` or ``THIRD_PARTY_NOTICES.md``) points at that document's page instead of the raw ``.md``:
the documents linked are rendered too, each to a page of its own (``page_path``), and kept until their
``.md`` changes (``write_page``). ``help_page`` adds the offline chapter navigation and text search.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices, QTextDocument
from PyQt5.QtWidgets import QWidget

from .app_dialogs import warn
from .help_code import document_html
from .help_page import documentation_page
from .i18n import current_language, tr, trf

ROOT = Path(__file__).resolve().parents[4]
"""The program folder (a source checkout, or the folder of the Windows package)."""
DOCS = ROOT / "docs"
USER_MANUAL = DOCS / "User_Manual.md"
USER_MANUAL_ZH = DOCS / "User_Manual_zh.md"
"""The Chinese edition: Help ▸ User Manual opens it when the interface is in Chinese."""
PHYSICS_MANUAL = DOCS / "Physics_Manual.md"
PHYSICS_MANUAL_ZH = DOCS / "Physics_Manual_zh.md"
THIRD_PARTY_NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
PACKAGED_NOTICES = ROOT / "licenses" / "THIRD_PARTY_NOTICES.md"
"""Where the Windows package keeps a second copy, beside the licence texts it collects."""

# -- Markdown before Qt reads it -------------------------------------------------------------------

_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_AUTOLINK = re.compile(
    r"<(?:[A-Za-z][A-Za-z0-9+.\-]{1,31}:[^\s<>]*"  # <https://…>, <mailto:…>
    r"|[A-Za-z0-9.!#$%&'*+/=?^_`{|}~\-]+@[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)*)>")  # <name@example.org>
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_BLANK_LINE = re.compile(r"\n[ \t]*\n")


def _closing_ticks(text: str, start: int, run: int) -> int:
    """Where the code span opened by ``run`` backticks before ``start`` closes, or -1 (no closing run of the
    same length before a blank line: the backticks are then literal)."""
    position = start
    while True:
        found = text.find("`" * run, position)
        if found < 0 or _BLANK_LINE.search(text, start, found):
            return -1
        end = found
        while end < len(text) and text[end] == "`":
            end += 1
        if end - found == run:
            return found
        position = end


def _escape_prose(text: str) -> str:
    """``text`` (no fenced code in it) with every ``<`` escaped outside code spans, autolinks and comments."""
    out, index, size = [], 0, len(text)
    while index < size:
        char = text[index]
        if char == "\\":  # an escape (``\<``, ``\```): kept with the character it escapes
            out.append(text[index:index + 2])
            index += 2
        elif char == "`":
            end = index
            while end < size and text[end] == "`":
                end += 1
            closing = _closing_ticks(text, end, end - index)
            stop = end if closing < 0 else closing + (end - index)
            out.append(text[index:stop])
            index = stop
        elif char == "<":
            found = _AUTOLINK.match(text, index) or _COMMENT.match(text, index)
            out.append(found.group(0) if found else "\\<")
            index = found.end() if found else index + 1
        else:
            out.append(char)
            index += 1
    return "".join(out)


def escape_raw_html(markdown: str) -> str:
    """The Markdown with every ``<`` that could start raw HTML escaped (``\\<``).

    Code spans, fenced code blocks (````` ``` ````` or ``~~~``, also indented in a list), autolinks
    (``<https://…>``, ``<name@example.org>``), HTML comments and ``<`` already escaped stay as they are."""
    lines = markdown.splitlines(keepends=True)
    out: list[str] = []
    prose: list[str] = []
    fence: Optional[tuple[str, int]] = None
    for line in lines:
        opening = _FENCE.match(line)
        if fence is None:
            if opening and (opening.group(1)[0] == "~" or "`" not in line[opening.end():]):
                out.append(_escape_prose("".join(prose)))
                prose = []
                out.append(line)
                fence = (opening.group(1)[0], len(opening.group(1)))
            else:
                prose.append(line)
            continue
        out.append(line)
        if (opening and opening.group(1)[0] == fence[0] and len(opening.group(1)) >= fence[1]
                and not line[opening.end():].strip()):
            fence = None
    out.append(_escape_prose("".join(prose)))
    return "".join(out)


# -- Qt's HTML after it -------------------------------------------------------------------------------

_HEADING = re.compile(r"<h([1-6])((?:\s[^>]*)?)>(.*?)</h\1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_LINK = re.compile(r"""\b(href|src)="([^"]*)\"""")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")


def github_slug(text: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces as hyphens (``3. The
    window`` → ``3-the-window``; Chinese letters are kept)."""
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


def heading_texts(page: str) -> list[str]:
    """The text of every heading of an HTML page, in order (tags removed, entities read)."""
    return [" ".join(html.unescape(_TAG.sub("", inner)).split()) for _level, _attributes, inner in _HEADING.findall(page)]


def add_heading_anchors(page: str) -> str:
    """Every ``<h1>``…``<h6>`` with ``id`` = its GitHub slug; a slug seen before gets ``-1``, ``-2`` … as on
    GitHub. A heading that has an ``id`` already keeps it."""
    seen: dict[str, int] = {}

    def anchor(match: re.Match) -> str:
        level, attributes, inner = match.groups()
        if re.search(r"\bid=", attributes or ""):
            return match.group(0)
        slug = github_slug(" ".join(html.unescape(_TAG.sub("", inner)).split()))
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        if count:
            slug = f"{slug}-{count}"
        return f'<h{level} id="{html.escape(slug, quote=True)}"{attributes}>{inner}</h{level}>'

    return _HEADING.sub(anchor, page)


PageOf = Callable[[Path], Optional[Path]]
"""The page a linked file is shown as (``None``: the file itself)."""


def absolute_links(page: str, folder: Path, page_of: Optional[PageOf] = None) -> str:
    """Relative ``href`` and ``src`` (``manual_images/…``, ``LICENSE``) as ``file:`` URLs of ``folder``;
    ``#…`` links, ``https:``/``mailto:`` links and the like stay. A link (``href``) to a file that
    ``page_of`` gives a page for points at that page instead, at the same ``#anchor``."""

    def resolve(match: re.Match) -> str:
        name, target = match.groups()
        plain = html.unescape(target)
        if not plain or plain.startswith("#") or _SCHEME.match(plain) or plain.startswith("//"):
            return match.group(0)
        path, _hash, fragment = plain.partition("#")
        file = (folder / unquote(path)).resolve()
        shown = page_of(file) if page_of is not None and name == "href" else None
        url = QUrl.fromLocalFile(str(shown or file))
        if fragment:
            url.setFragment(fragment)
        return f'{name}="{html.escape(url.toString(), quote=True)}"'

    return _LINK.sub(resolve, page)


_BODY = re.compile(r"<body[^>]*>(.*)</body>", re.S | re.I)


def render_markdown(markdown: str, title: str, folder: Path, page_of: Optional[PageOf] = None, head: str = "") -> str:
    """A whole web page of ``markdown``: Qt's HTML of it, with anchors on the headings and the links and
    images of ``folder`` (where the document lies) as absolute ``file:`` URLs (``page_of``: see
    ``absolute_links``). An empty ``title`` takes the text of the first heading; ``head`` goes into ``<head>``."""
    document = QTextDocument()
    document.setMarkdown(escape_raw_html(markdown))
    rendered = document_html(document)
    found = _BODY.search(rendered)
    body = found.group(1) if found else rendered
    body = absolute_links(add_heading_anchors(body), folder, page_of)
    title = title or next(iter(heading_texts(body)), "") or "GIMaP"
    return documentation_page(body, title, head)


# -- the pages in the temporary folder ----------------------------------------------------------------

PAGE_FORMAT = 4
"""Goes up when the pages ``render_markdown`` writes change: a page kept from another format is rendered again."""
_STAMP = re.compile(r'<meta name="gimap-help-page" content="([^"]*)">')


def is_document(path: Path) -> bool:
    """A Markdown file that is there: a link to it opens its page (``write_page``)."""
    return path.suffix.lower() == ".md" and path.is_file()


def page_path(source: Path) -> Path:
    """Where the page of ``source`` is written: ``GIMaP_<stem>_<8 hex digits of its path>.html`` in the
    temporary folder (one page per file: two ``README.md``, or two installations, never share a page)."""
    key = os.path.normcase(str(source.resolve()))
    return Path(tempfile.gettempdir()) / f"GIMaP_{source.stem}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}.html"


def _stamp(source: Path) -> dict:
    """What a page records of the document it was rendered from."""
    stat = source.stat()
    return {"format": PAGE_FORMAT, "source": os.path.normcase(str(source.resolve())), "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size}


def _render_page(source: Path, title: str) -> list[Path]:
    """Render ``source`` to ``page_path(source)``; the Markdown documents it links to."""
    links: list[Path] = []

    def page_of(target: Path) -> Optional[Path]:
        if not is_document(target):
            return None
        if target not in links:
            links.append(target)
        return page_path(target)

    stamp = _stamp(source)
    markdown = source.read_text(encoding="utf-8")
    body = render_markdown(markdown, title, source.parent, page_of)
    stamp["links"] = [str(link) for link in links]
    head = f'<meta name="gimap-help-page" content="{html.escape(json.dumps(stamp), quote=True)}">'
    page_path(source).write_text(body.replace("</title>", "</title>" + head, 1), encoding="utf-8")
    return links


def _kept_links(source: Path) -> Optional[list[Path]]:
    """The documents linked by the page of ``source`` kept from before, or ``None`` when it has to be rendered
    again: no page, or the ``.md`` is newer than its page or not the one it was rendered from."""
    page = page_path(source)
    try:
        if page.stat().st_mtime_ns < source.stat().st_mtime_ns:
            return None
        found = _STAMP.search(page.read_text(encoding="utf-8"))
        stamp = json.loads(html.unescape(found.group(1))) if found else {}
        if not isinstance(stamp, dict) or any(stamp.get(key) != value for key, value in _stamp(source).items()):
            return None
        return [Path(link) for link in stamp.get("links", ())]
    except (OSError, UnicodeDecodeError, ValueError, TypeError):
        return None


def write_page(source: Path, title: str) -> Path:
    """``source`` rendered (``render_markdown``) to its page in the temporary folder (``page_path``), and the
    Markdown documents it links to — and those they link to — rendered to theirs, so that a link from one
    document to another opens the other's page at the section. ``source`` is always rendered again (its
    ``title`` follows the interface language); a linked page only when its ``.md`` is newer. A linked document
    that cannot be read is left out: the page of ``source`` opens all the same."""
    done = {source.resolve()}
    waiting = list(_render_page(source, title))
    while waiting:
        document = waiting.pop()
        if document.resolve() in done:
            continue
        done.add(document.resolve())
        links = _kept_links(document)
        if links is None:
            try:
                links = _render_page(document, "")
            except (OSError, UnicodeDecodeError):
                continue
        waiting.extend(link for link in links if is_document(link))
    return page_path(source)


def open_document(parent: Optional[QWidget], source: Path, title: str, missing: str) -> Optional[Path]:
    """Render ``source`` and open it in the default browser; the page written, or ``None`` when ``source``
    is not there (``missing``, a template with ``{path}``, then says so) or cannot be read."""
    if not source.is_file():
        warn(parent, title, missing.format(path=source))
        return None
    try:
        page = write_page(source, title)
    except (OSError, UnicodeDecodeError) as exc:
        warn(parent, title, trf("{name} could not be opened: {error}", name=source.name, error=exc))
        return None
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(page)))
    return page


# -- the documents of the Help menu -------------------------------------------------------------------


def edition(english: Path, chinese: Path, language: str) -> Path:
    """The edition of a document for an interface language: Chinese when there is one, else English."""
    return chinese if language == "zh" and chinese.is_file() else english


def manual_for_language(language: str) -> Path:
    return edition(USER_MANUAL, USER_MANUAL_ZH, language)


def physics_manual_for_language(language: str) -> Path:
    return edition(PHYSICS_MANUAL, PHYSICS_MANUAL_ZH, language)


def notices_path() -> Path:
    """THIRD_PARTY_NOTICES.md of the program folder (the package's copy in ``licenses/`` if that one is gone)."""
    return PACKAGED_NOTICES if not THIRD_PARTY_NOTICES.is_file() and PACKAGED_NOTICES.is_file() else THIRD_PARTY_NOTICES


def open_user_manual(parent: Optional[QWidget] = None) -> Optional[Path]:
    """Help ▸ User Manual: the manual in the interface language, as a web page in the default browser."""
    return open_document(parent, manual_for_language(current_language()), tr("User Manual"),
                         tr("The user manual was not found:\n{path}"))


def open_physics_manual(parent: Optional[QWidget] = None) -> Optional[Path]:
    """Help ▸ Physics and Methods: how GIMaP computes what it shows, in the default browser."""
    return open_document(parent, physics_manual_for_language(current_language()), tr("Physics and Methods"),
                         tr("The physics and methods manual was not found:\n{path}"))


def open_third_party_notices(parent: Optional[QWidget] = None) -> Optional[Path]:
    """Help ▸ Third-party Notices: what is not GIMaP's own work, and its licences, in the default browser."""
    return open_document(parent, notices_path(), tr("Third-party Notices"),
                         tr("The third-party notices were not found:\n{path}"))


__all__ = [
    "PHYSICS_MANUAL",
    "PHYSICS_MANUAL_ZH",
    "THIRD_PARTY_NOTICES",
    "USER_MANUAL",
    "USER_MANUAL_ZH",
    "absolute_links",
    "add_heading_anchors",
    "escape_raw_html",
    "github_slug",
    "heading_texts",
    "is_document",
    "manual_for_language",
    "open_document",
    "page_path",
    "open_physics_manual",
    "open_third_party_notices",
    "open_user_manual",
    "physics_manual_for_language",
    "render_markdown",
    "write_page",
]
