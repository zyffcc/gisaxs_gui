"""One language per report: the section headings of the automatic analysis's reports.

The sentences of a report (decisions, reasons, open questions) are the application's and stay English: records
and agents read them. The headings follow the language the report is written for: the program's language in the
window, English on the command line unless ``--language zh`` is given. A heading is never written in both.
"""

from __future__ import annotations

ENGLISH, CHINESE = "en", "zh"
LANGUAGES = (ENGLISH, CHINESE)

HEADINGS_ZH = {
    "Result summary": "结果摘要",
    "Needs attention (answer, then run again)": "需要处理（回答后重新运行）",
    "Open questions": "需要处理",
    "Geometry": "几何",
    "Mask": "掩膜",
    "Record": "记录",
    "Frames": "帧",
    "Peaks (radial I(q))": "峰（径向 I(q)）",
    "Ring orientation": "取向",
    "Orientation checks": "取向核查",
    "Phase assignment": "物相指认",
    "Refined unit cells": "精修晶胞",
    "Crystal orientation": "晶体取向",
    "Decisions": "决策",
    "Steps": "步骤",
    "Cuts": "切线",
    "Length scale 2π/q*": "特征长度 2π/q*",
    "Fit of I(qy)": "I(qy) 的拟合",
    "Not analysed": "未分析",
    "Start versus end of the series": "序列的开始与结束",
}
"""The Chinese of every section heading (the key is the English heading)."""


def heading(text: str, language: str = ENGLISH) -> str:
    """``## text`` in ``language`` (English for any language without a table)."""
    return "## " + (HEADINGS_ZH.get(text, text) if language == CHINESE else text)


__all__ = ["CHINESE", "ENGLISH", "HEADINGS_ZH", "LANGUAGES", "heading"]
