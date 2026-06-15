"""Utility functions for generating before/after comparisons."""

from typing import List, Optional


def generate_anonymization_diff(
    original: str,
    anonymized: str,
    detected_entities: Optional[List[dict]] = None,
    anonymization_mode: str = "presidio"
) -> str:
    """Generate a markdown-formatted before/after comparison for anonymization.

    Creates a markdown display showing:
    - Summary table of detected PII entity types and counts (if available)
    - Original text section
    - Anonymized text section

    Args:
        original: Original text before anonymization
        anonymized: Anonymized text with PII masked
        detected_entities: Optional list of detected entity dicts. Each dict should have:
                          - entity_type: Type of entity (e.g., "PERSON", "EMAIL_ADDRESS")
                          - start: Start position in text
                          - end: End position in text
                          - score: Confidence score (optional)
                          - text: The actual text that was detected (optional)
        anonymization_mode: The anonymization method used ("presidio" or "kserve")

    Returns:
        Markdown-formatted string for display
    """
    try:
        diff_md = "### 🔍 Anonymization Summary\n\n"

        # Add mode indicator
        mode_label = {
            "presidio": "Presidio (Local PII Masking)",
            "kserve": "KServe (AI Model)"
        }.get(anonymization_mode, anonymization_mode)
        diff_md += f"**Method:** {mode_label}\n\n"

        # Entity detection summary (if available)
        if detected_entities:
            # Count entities by type
            entity_counts = {}
            for entity in detected_entities:
                entity_type = entity.get('entity_type', 'UNKNOWN')
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

            # Create summary table
            diff_md += "**PII Detected:**\n\n"
            diff_md += "| Entity Type | Count |\n"
            diff_md += "|------------|-------|\n"
            for entity_type, count in sorted(entity_counts.items()):
                diff_md += f"| {entity_type} | {count} |\n"
            diff_md += "\n"
        else:
            # KServe mode or no entities detected
            if anonymization_mode == "kserve":
                diff_md += "*AI model processed the text for anonymization.*\n\n"
            else:
                diff_md += "*No PII detected in the document.*\n\n"

        # Original text section
        diff_md += "### 📄 Original Text\n\n"
        diff_md += "```\n"
        diff_md += original[:2000]  # Limit to first 2000 chars for display
        if len(original) > 2000:
            diff_md += "\n... (truncated for display)"
        diff_md += "\n```\n\n"

        # Anonymized text section
        diff_md += "### 🔒 Anonymized Text\n\n"
        diff_md += "```\n"
        diff_md += anonymized[:2000]  # Limit to first 2000 chars for display
        if len(anonymized) > 2000:
            diff_md += "\n... (truncated for display)"
        diff_md += "\n```\n\n"

        return diff_md

    except Exception as e:
        return f"Error generating diff: {str(e)}"
