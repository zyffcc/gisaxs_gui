"""Plain spellings of the application's scientific symbols in UTF-8 text exports.

Source paths and file names must be inserted separately, unchanged. Other
characters (including names in another script) are kept; no encoding occurs here.
"""

_SPELLINGS = (
    ("Å⁻¹", "1/A"), ("⁻¹", "^-1"), ("⁻²", "^-2"), ("Å", "A"), ("°", "deg"), ("χ", "chi"), ("∥", "_par"),
    ("2θ", "2theta"), ("θ", "theta"), ("αi", "alpha_i"), ("αf", "alpha_f"), ("αc", "alpha_c"), ("α", "alpha"),
    ("σ", "sigma"), ("λ", "lambda"), ("η", "eta"), ("π", "pi"), ("µ", "u"), ("μ", "u"),
    ("₁₀", "10"), ("ᵣ", "_r"), ("²", "^2"), ("³", "^3"),
    ("×", "x"), ("±", "+/-"), ("−", "-"), ("·", "*"), ("≤", "<="), ("≥", ">="), ("→", "->"),
    ("–", "-"), ("—", "-"), ("…", "..."), ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
)
"""Applied in order: ``Å⁻¹`` before ``Å``, ``2θ`` before ``θ``."""


def export_text(text: str) -> str:
    """Spell the application's symbols plainly, leaving other characters alone."""
    for original, replacement in _SPELLINGS:
        text = text.replace(original, replacement)
    return text
