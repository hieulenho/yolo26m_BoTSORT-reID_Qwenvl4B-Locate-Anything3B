"""Static research-summary text blocks for language benchmark reports."""

from __future__ import annotations


def limitations_section() -> str:
    return "\n".join(
        [
            "## Limitations",
            "",
            "- Small manually annotated language-query benchmark until more queries are added.",
            "- Football-focused smoke data; cross-domain claims require separate annotation.",
            "- Accuracy depends on raw tracker quality and available saved artifacts.",
            "- Same-kit players can confuse grounding and appearance verification.",
            "- No jersey OCR or action/event semantic reasoning is implemented.",
            "- Fusion is threshold/weight based, not learned.",
        ]
    )


def future_work_section() -> str:
    return "\n".join(
        [
            "## Future Work",
            "",
            "- Learned fusion model for candidate evidence.",
            "- OCR-assisted jersey number tracking.",
            "- Team classification and ball-aware relational queries.",
            "- Cross-camera semantic identity.",
            "- Online streaming architecture and global appearance gallery.",
        ]
    )
