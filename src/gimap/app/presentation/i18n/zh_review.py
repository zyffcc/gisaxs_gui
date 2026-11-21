"""Chinese translations added by the first-user review fixes, merged into one table."""

from .zh_review_curves import REVIEW_CURVES_ZH
from .zh_review_display import REVIEW_DISPLAY_ZH
from .zh_review_records import REVIEW_RECORDS_ZH
from .zh_review_solutions import REVIEW_SOLUTIONS_ZH

REVIEW_ZH: dict[str, str] = {
    **REVIEW_RECORDS_ZH,
    **REVIEW_DISPLAY_ZH,
    **REVIEW_CURVES_ZH,
    **REVIEW_SOLUTIONS_ZH,
}
