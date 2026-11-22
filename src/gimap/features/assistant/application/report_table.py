"""Keep long explanations beside the Markdown table they belong to."""

from urllib.parse import quote


def table_row(cells, footnotes, *, move_columns):
    """If a row reaches 100 characters, move prose or a long parameter list below the table."""
    cells = [str(cell).replace("|", "\\|") for cell in cells]
    row = "| " + " | ".join(cells) + " |"
    for index in move_columns:
        if len(row) < 100:
            break
        content = cells[index]
        marker = f"[{len(footnotes) + 1}]"
        if len(content) > len(marker):
            footnotes.append(f"- {marker} {content}")
            cells[index] = marker
            row = "| " + " | ".join(cells) + " |"
    return row


def report_link(report: dict) -> str:
    """A relative Markdown link to the frame's report, or a dash for a frame without a folder."""
    folder = report.get("folder")
    return f"[report.md]({quote(str(folder))}/report.md)" if folder else "—"


def batch_table(headers, reports, rows, *, move_columns, link_ids):
    """Keep the frame name, report link and short observations together in each batch row."""
    linked = any(report.get("folder") for report in reports)
    headers = [*headers, "report"] if linked else headers
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    footnotes, references = [], []
    for report, cells in zip(reports, rows):
        cells = list(cells)
        movable = sorted(move_columns, key=lambda index: len(cells[index]), reverse=True)
        if linked:
            link = report_link(report)
            # Use a reference link if even a row with every long observation in the notes
            # would be too wide. The frame identity and the link stay in the table.
            minimum = list(cells)
            for offset, index in enumerate(movable, 1):
                marker = f"[{len(footnotes) + offset}]"
                if len(minimum[index]) > len(marker):
                    minimum[index] = marker
            if report.get("folder") and len("| " + " | ".join([*minimum, link]) + " |") >= 100:
                ref = f"r{link_ids[id(report)]}"
                link = f"[report.md][{ref}]"
                references.append(f"[{ref}]: {quote(str(report['folder']))}/report.md")
            cells.append(link)
        lines.append(table_row(cells, footnotes, move_columns=movable))
    return lines + (["", *footnotes] if footnotes else []) + (["", *references] if references else [])
