"""Keep Markdown equations and code literal when Qt exports a document as HTML.

Qt records code fences and inline code in its text formats, but its HTML exporter writes ordinary
paragraphs and can lose the fixed-pitch font on macOS. Use those formats to restore semantic HTML.
No formula is parsed, evaluated or converted to another mathematical notation.
"""

from __future__ import annotations

import html
import re

from PyQt5.QtGui import QTextCharFormat, QTextCursor, QTextDocument, QTextFormat

_CODE_FONT = "GIMaPCodeLiteral"
_INLINE = re.compile(r'<span\b[^>]*style="[^"]*GIMaPCodeLiteral[^"]*"[^>]*>(.*?)</span>', re.S)
_PRE = re.compile(r"<pre\b[^>]*>(.*?)</pre>", re.S)
_PRE_RUN = re.compile(r"(?:<pre\b[^>]*>.*?</pre>\s*)+", re.S)
_TAG = re.compile(r"<[^>]+>")


def document_html(document: QTextDocument) -> str:
    """Export Qt's parsed document, preserving grouped code lines and inline code as literal text."""
    block = document.begin()
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
                if fragment.isValid() and fragment.charFormat().fontFixedPitch():
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
        literal = "\n".join(lines).rstrip("\n") + "\n"
        return f'<pre class="help-code" tabindex="0"><code>{html.escape(literal)}</code></pre>\n'

    return _PRE_RUN.sub(code_block, page)
