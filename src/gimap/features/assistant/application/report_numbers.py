"""Numerical precision and uncertainties in the automatic analysis report."""

def _number(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _fixed(value, decimals: int) -> str:
    """A number with a fixed number of decimals ("—" when there is none): one precision per table column."""
    return "—" if value is None else f"{float(value):.{decimals}f}"


def _pm(value, error, decimals: int) -> str:
    """``value ± error`` with the column's decimals; an error smaller than they show keeps one significant digit."""
    text = _fixed(value, decimals)
    if value is None or not isinstance(error, (int, float)) or not error > 0:
        return text
    shown = f"{float(error):.{decimals}f}"
    return f"{text} ± {shown if float(shown) > 0 else f'{float(error):.1g}'}"
