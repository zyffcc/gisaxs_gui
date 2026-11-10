"""Keep Markdown equations and code literal when Qt exports a document as HTML.

Qt records code fences and inline code in its text formats, but its HTML exporter writes ordinary
paragraphs and can lose the fixed-pitch font on macOS. Use those formats to restore semantic HTML.
No formula is parsed, evaluated or converted to another mathematical notation.
"""

from __future__ import annotations

import html
import re
import uuid

from PyQt5.QtGui import QFontDatabase, QTextCharFormat, QTextCursor, QTextDocument, QTextFormat

_CODE_FONT = "GIMaPCodeLiteral"
_INLINE = re.compile(r'<span\b[^>]*style="[^"]*GIMaPCodeLiteral[^"]*"[^>]*>(.*?)</span>', re.S)
_PRE = re.compile(r"<pre\b[^>]*>(.*?)</pre>", re.S)
_PRE_RUN = re.compile(r"(?:<pre\b[^>]*>.*?</pre>\s*)+", re.S)
_TAG = re.compile(r"<[^>]+>")
_FENCE = re.compile(r"^([ \t]*)(`{3,}|~{3,})([^\n]*)")


def protect_fences(markdown: str) -> tuple[str, dict[str, str]]:
    """Keep fenced source text outside Qt's version-dependent Markdown whitespace normalization."""
    lines = markdown.splitlines(keepends=True)
    output, literals = [], {}
    prefix = "GIMaPFence" + uuid.uuid4().hex
    index = 0
    while index < len(lines):
        opening = _FENCE.match(lines[index])
        if opening is None or (opening[2][0] == "`" and "`" in opening[3]):
            output.append(lines[index])
            index += 1
            continue
        indent, fence = opening[1], opening[2]
        first = index
        index += 1
        body = []
        while index < len(lines):
            closing = _FENCE.match(lines[index])
            if (closing and closing[2][0] == fence[0] and len(closing[2]) >= len(fence)
                    and not closing[3].strip()):
                break
            line = lines[index]
            body.append(line[min(len(indent), len(line) - len(line.lstrip(" "))):])
            index += 1
        token = prefix + str(len(literals))
        literals[token] = "".join(body).rstrip("\n") + "\n"
        output.extend((lines[first].rstrip("\n") + "\n", indent + token + "\n"))
        if index < len(lines):
            output.append(lines[index])
            index += 1
    return "".join(output), literals


def document_html(document: QTextDocument, literals: dict[str, str] | None = None) -> str:
    """Export Qt's parsed document, preserving grouped code lines and inline code as literal text."""
    block = document.begin()
    fixed_family = QFontDatabase.systemFont(QFontDatabase.FixedFont).family()
    code_spans: list[tuple[int, int]] = []
    while block.isValid():
        block_format = block.blockFormat()
        if block_format.hasProperty(QTextFormat.BlockCodeLanguage):
            # This property also covers indented code and fences without a language tag.
            block_format.setNonBreakableLines(True)
            QTextCursor(block).setBlockFormat(block_format)
        else:
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if fragment.isValid():
                    format = fragment.charFormat()
                    # Older Qt on Windows marks Markdown code by its explicit fixed-font family,
                    # without setting FontFixedPitch. Ordinary prose has no explicit font family.
                    if (format.fontFixedPitch() or
                            (format.hasProperty(QTextFormat.FontFamily) and format.fontFamily() == fixed_family)):
                        code_spans.append((fragment.position(), fragment.length()))
                iterator += 1
        block = block.next()

    # Collect first: changing a fragment's format while iterating can split/invalidate fragments.
    for start, length in code_spans:
        cursor = QTextCursor(document)
        cursor.setPosition(start)
        cursor.setPosition(start + length, QTextCursor.KeepAnchor)
        marker = QTextCharFormat()
        marker.setFontFamily(_CODE_FONT)
        cursor.mergeCharFormat(marker)

    page = _INLINE.sub(r"<code>\1</code>", document.toHtml())

    def code_block(match: re.Match) -> str:
        lines = []
        for inner in _PRE.findall(match.group(0)):
            # An empty QTextBlock is exported as <br />. Joining blocks already adds its newline.
            separator = "" if not _TAG.sub("", inner) else "\n"
            inner = re.sub(r"<br\s*/?>", separator, inner, flags=re.I)
            lines.append(html.unescape(_TAG.sub("", inner)))
        # Qt adds an empty block at a closing fence; it is not an extra line of the source formula.
        chunks, pending = [], []

        def append_literal(text: str) -> None:
            chunks.append(f'<pre class="help-code" tabindex="0"><code>{html.escape(text)}</code></pre>\n')

        for line in lines:
            original = (literals or {}).get(line.strip())
            if original is not None:
                if any(pending):
                    append_literal("\n".join(pending).rstrip("\n") + "\n")
                pending = []
                append_literal(original)
            else:
                pending.append(line)
        if any(pending) or not chunks:
            append_literal("\n".join(pending).rstrip("\n") + "\n")
        return "".join(chunks)

    return _PRE_RUN.sub(code_block, page)
