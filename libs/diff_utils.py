"""Utility functions for generating before/after comparisons."""

import difflib
from typing import List, Optional


def generate_anonymization_diff(
    original: str,
    anonymized: str,
    detected_entities: Optional[List[dict]] = None,
    anonymization_mode: str = "presidio"
) -> str:
    """Generate a unified diff-style comparison for anonymization.

    Creates a display showing:
    - Summary table of detected PII entity types and counts (if available)
    - Unified diff with red/green highlighting showing all changes
    - No truncation - displays complete text

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
        HTML-formatted string with styled unified diff for display
    """
    try:
        diff_html = ""

        # Header with summary
        diff_html += "<div style='margin-bottom: 20px;'>\n"
        diff_html += "<h3>🔍 Anonymization Summary</h3>\n"

        # Add mode indicator
        mode_label = {
            "presidio": "Presidio (Local PII Masking)",
            "kserve": "KServe (AI Model)"
        }.get(anonymization_mode, anonymization_mode)
        diff_html += f"<p><strong>Method:</strong> {mode_label}</p>\n"

        # Entity detection summary (if available)
        if detected_entities:
            # Count entities by type
            entity_counts = {}
            for entity in detected_entities:
                entity_type = entity.get('entity_type', 'UNKNOWN')
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

            # Create summary table
            diff_html += "<p><strong>PII Detected:</strong></p>\n"
            diff_html += "<table style='border-collapse: collapse; margin-bottom: 10px;'>\n"
            diff_html += "<tr><th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Entity Type</th>"
            diff_html += "<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Count</th></tr>\n"
            for entity_type, count in sorted(entity_counts.items()):
                diff_html += f"<tr><td style='border: 1px solid #ddd; padding: 8px;'>{entity_type}</td>"
                diff_html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{count}</td></tr>\n"
            diff_html += "</table>\n"
        else:
            # KServe mode or no entities detected
            if anonymization_mode == "kserve":
                diff_html += "<p><em>AI model processed the text for anonymization.</em></p>\n"
            else:
                diff_html += "<p><em>No PII detected in the document.</em></p>\n"

        diff_html += "</div>\n"

        # Generate unified diff
        diff_html += "<h3>📊 Unified Diff (Changes)</h3>\n"

        # Split texts into lines for diff
        original_lines = original.splitlines(keepends=True)
        anonymized_lines = anonymized.splitlines(keepends=True)

        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            original_lines,
            anonymized_lines,
            fromfile='Original',
            tofile='Anonymized',
            lineterm=''
        ))

        if not diff_lines:
            # No differences found
            diff_html += "<p><em>No differences detected between original and anonymized text.</em></p>\n"
        else:
            # Add CSS styling for the diff
            diff_html += """
<style>
.diff-container {
    font-family: 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.4;
    white-space: pre-wrap;
    word-wrap: break-word;
    background: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 10px;
    max-height: 600px;
    overflow-y: auto;
}
.diff-removed {
    background: #ffecec;
    color: #cc0000;
    display: block;
    margin: 0;
    padding: 2px 4px;
}
.diff-added {
    background: #eaffea;
    color: #008000;
    display: block;
    margin: 0;
    padding: 2px 4px;
}
.diff-context {
    color: #333;
    display: block;
    margin: 0;
    padding: 2px 4px;
}
.diff-header {
    color: #666;
    font-weight: bold;
    display: block;
    margin: 8px 0 4px 0;
    padding: 2px 4px;
    background: #e8e8e8;
}
.diff-file {
    color: #888;
    display: block;
    margin: 0;
    padding: 2px 4px;
}
</style>
"""

            # Process diff lines and apply styling
            diff_html += "<div class='diff-container'>"

            for line in diff_lines:
                # HTML escape special characters
                line_escaped = (line
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#39;'))

                if line.startswith('---') or line.startswith('+++'):
                    # File markers
                    diff_html += f"<span class='diff-file'>{line_escaped}</span>\n"
                elif line.startswith('@@'):
                    # Hunk header
                    diff_html += f"<span class='diff-header'>{line_escaped}</span>\n"
                elif line.startswith('-'):
                    # Removed line
                    diff_html += f"<span class='diff-removed'>{line_escaped}</span>\n"
                elif line.startswith('+'):
                    # Added line
                    diff_html += f"<span class='diff-added'>{line_escaped}</span>\n"
                else:
                    # Context line
                    diff_html += f"<span class='diff-context'>{line_escaped}</span>\n"

            diff_html += "</div>\n"

        return diff_html

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"<p style='color: red;'>Error generating diff: {str(e)}</p><pre>{error_detail}</pre>"
