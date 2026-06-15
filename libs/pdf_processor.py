"""PDF processing module for anonymizing PDFs using docling."""

import re
import io
from typing import List, Tuple, Optional, Callable
from pathlib import Path
import tempfile
import requests
from docling.document_converter import DocumentConverter
from pypdf import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


def pdf_to_markdown(pdf_bytes: bytes) -> str:
    """Convert PDF to markdown using docling.

    Args:
        pdf_bytes: PDF file content as bytes

    Returns:
        Markdown text extracted from PDF
    """
    # Docling works with file paths, so we need to create a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = Path(tmp_file.name)

    try:
        # Initialize docling converter
        converter = DocumentConverter()

        # Convert PDF to markdown
        result = converter.convert(tmp_path)

        # Export to markdown
        markdown_text = result.document.export_to_markdown()

        return markdown_text

    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex pattern.

    Handles common sentence boundaries: . ! ?
    """
    if not text or not text.strip():
        return []

    # Remove markdown headers, lists, and formatting
    # Keep the content but remove markdown syntax
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)  # Remove headers
    text = re.sub(r"^\*\s+", "", text, flags=re.MULTILINE)  # Remove bullet points
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)  # Remove numbered lists
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Remove bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Remove italic
    text = re.sub(r"`([^`]+)`", r"\1", text)  # Remove inline code

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    # Filter out empty sentences and clean up
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]


def _call_kserve_api(
    model_name: str, base_url: str, task_type: str, input_data: dict, timeout: int = 30
) -> Tuple[Optional[str], Optional[str]]:
    """Internal function to call KServe API endpoint.

    Args:
        model_name: Name of the model to call
        base_url: Base URL of the KServe endpoint
        task_type: Type of task (e.g., "anonymize", "summarize")
        input_data: Input data dict to send to the model
        timeout: Request timeout in seconds

    Returns:
        (result_text, error_message) - one will be None
    """
    from libs.apis import build_payload

    endpoint_url = f"{base_url.rstrip('/')}/v2/models/{model_name}/infer"

    try:
        payload = build_payload(task_type, input_data)
        response = requests.post(
            endpoint_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()

        result_data = response.json()

        if "outputs" in result_data and len(result_data["outputs"]) > 0:
            output_data = result_data["outputs"][0].get("data", [])
            if output_data:
                return output_data[0], None
            else:
                return None, "No output data in response"
        else:
            return None, "No outputs in response"

    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except requests.exceptions.RequestException as e:
        return None, str(e)
    except Exception as e:
        return None, f"Unexpected error - {str(e)}"


def anonymize_sentences(
    sentences: List[str],
    model_name: str,
    base_url: str,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[List[str], List[str]]:
    """Anonymize sentences using the KServe API.

    Returns (anonymized_sentences, errors)
    """
    anonymized = []
    errors = []

    for i, sentence in enumerate(sentences):
        result, error = _call_kserve_api(
            model_name, base_url, "anonymize", {"anonymize": sentence}, timeout=30
        )

        if result:
            anonymized.append(result)
        else:
            anonymized.append(sentence)
            errors.append(f"Sentence {i + 1}: {error}")

        if progress_callback:
            progress_callback((i + 1) / len(sentences))

    return anonymized, errors


def _create_pdf_from_text(
    text: str, pagesize: Tuple[float, float] = letter, handle_markdown: bool = False
) -> bytes:
    """Internal function to create PDF from text with optional markdown support.

    Args:
        text: Text content (plain or markdown)
        pagesize: Page dimensions (width, height)
        handle_markdown: Whether to process markdown formatting

    Returns:
        PDF bytes
    """
    output = io.BytesIO()
    can = canvas.Canvas(output, pagesize=pagesize)

    lines = text.split("\n")
    y_position = pagesize[1] - 50
    x_position = 50
    line_height = 14
    font_size = 10

    can.setFont("Helvetica", font_size)

    for line in lines:
        if not line.strip():
            y_position -= line_height / 2
            continue

        # Handle markdown headers if enabled
        if handle_markdown and line.startswith("#"):
            header_level = len(line) - len(line.lstrip("#"))
            text_content = line.lstrip("#").strip()
            can.setFont("Helvetica-Bold", font_size + (4 - header_level))
            y_position -= line_height * 1.5
        else:
            text_content = line
            can.setFont("Helvetica", font_size)

        # Word wrap long lines
        max_width = pagesize[0] - 100
        if can.stringWidth(text_content) > max_width:
            words = text_content.split()
            current_line = []
            for word in words:
                test_line = " ".join(current_line + [word])
                if can.stringWidth(test_line) <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        can.drawString(x_position, y_position, " ".join(current_line))
                        y_position -= line_height
                        if y_position < 50:
                            can.showPage()
                            y_position = pagesize[1] - 50
                            can.setFont("Helvetica", font_size)
                        current_line = [word]
                    else:
                        current_line = [word]
            if current_line:
                can.drawString(x_position, y_position, " ".join(current_line))
                y_position -= line_height
        else:
            can.drawString(x_position, y_position, text_content)
            y_position -= line_height

        if y_position < 50:
            can.showPage()
            y_position = pagesize[1] - 50
            can.setFont("Helvetica", font_size)

    can.save()
    output.seek(0)
    return output.getvalue()


def markdown_to_pdf(markdown_text: str, original_pdf_bytes: bytes) -> bytes:
    """Convert anonymized markdown back to PDF.

    Uses reportlab to create a simple text-based PDF with the anonymized content.
    Attempts to preserve basic structure from the original PDF.

    Args:
        markdown_text: Anonymized markdown text
        original_pdf_bytes: Original PDF to extract page size info

    Returns:
        PDF bytes
    """
    # Read original PDF to get page dimensions
    original_reader = PdfReader(io.BytesIO(original_pdf_bytes))
    if len(original_reader.pages) > 0:
        first_page = original_reader.pages[0]
        page_width = float(first_page.mediabox.width)
        page_height = float(first_page.mediabox.height)
        pagesize = (page_width, page_height)
    else:
        pagesize = letter

    return _create_pdf_from_text(markdown_text, pagesize, handle_markdown=True)


def summarize_pdf(
    pdf_file,
    model_name: str,
    base_url: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Optional[str], Optional[bytes], str, List[str]]:
    """Process PDF for summarization: extract to markdown, summarize via AI.

    Pipeline:
    1. PDF → Markdown (via docling)
    2. Markdown → AI Model for summarization (via KServe API)
    3. Summary → PDF (via reportlab)

    Returns (summary_text, summary_pdf_bytes, status_message, errors)
    """
    try:
        if progress_callback:
            progress_callback(0.05, "Reading PDF...")

        pdf_bytes = pdf_file.read()

        if len(pdf_bytes) > 10 * 1024 * 1024:
            return None, None, "Error: PDF file exceeds 10MB limit", []

        if progress_callback:
            progress_callback(0.1, "Converting PDF to markdown with docling...")

        # Step 1: PDF → Markdown
        markdown_text = pdf_to_markdown(pdf_bytes)

        if not markdown_text or not markdown_text.strip():
            return None, None, "Error: No text extracted from PDF", []

        if progress_callback:
            progress_callback(0.4, "Sending to AI model for summarization...")

        # Step 2: Summarize via AI model
        summary_text, error = _call_kserve_api(
            model_name, base_url, "summarize", {"summarize": markdown_text}, timeout=60
        )

        if not summary_text:
            return None, None, f"Error: Failed to generate summary - {error}", [error]

        if progress_callback:
            progress_callback(0.8, "Converting summary to PDF...")

        # Step 3: Summary → PDF
        summary_pdf = text_to_pdf(summary_text)

        if progress_callback:
            progress_callback(1.0, "Complete!")

        return summary_text, summary_pdf, "Summary generated successfully", []

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        return None, None, f"Error processing PDF: {str(e)}", [error_detail]


def text_to_pdf(text: str) -> bytes:
    """Convert plain text to PDF.

    Args:
        text: Text content to convert

    Returns:
        PDF bytes
    """
    return _create_pdf_from_text(text, letter, handle_markdown=False)


def process_pdf(
    pdf_file,
    model_name: str,
    base_url: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Optional[bytes], str, List[str]]:
    """Process PDF: extract to markdown, anonymize, rebuild PDF.

    Pipeline:
    1. PDF → Markdown (via docling)
    2. Markdown → Sentences (via regex)
    3. Anonymize sentences (via KServe API)
    4. Sentences → Markdown
    5. Markdown → PDF (via reportlab)

    Returns (pdf_bytes, summary_message, errors)
    """
    try:
        if progress_callback:
            progress_callback(0.05, "Reading PDF...")

        pdf_bytes = pdf_file.read()

        if len(pdf_bytes) > 10 * 1024 * 1024:
            return None, "Error: PDF file exceeds 10MB limit", []

        if progress_callback:
            progress_callback(0.1, "Converting PDF to markdown with docling...")

        # Step 1: PDF → Markdown
        markdown_text = pdf_to_markdown(pdf_bytes)

        if not markdown_text or not markdown_text.strip():
            return None, "Error: No text extracted from PDF", []

        if progress_callback:
            progress_callback(0.3, "Splitting into sentences...")

        # Step 2: Markdown → Sentences
        sentences = split_into_sentences(markdown_text)

        if not sentences:
            return None, "Error: No sentences extracted from markdown", []

        if progress_callback:
            progress_callback(0.35, f"Anonymizing {len(sentences)} sentences...")

        # Step 3: Anonymize sentences
        def anon_progress(pct):
            if progress_callback:
                progress_callback(
                    0.35 + (pct * 0.5),
                    f"Anonymizing sentence {int(pct * len(sentences))}/{len(sentences)}...",
                )

        anonymized_sentences, errors = anonymize_sentences(
            sentences, model_name, base_url, anon_progress
        )

        if progress_callback:
            progress_callback(0.9, "Rebuilding PDF from anonymized text...")

        # Step 4 & 5: Sentences → Markdown → PDF
        anonymized_markdown = " ".join(anonymized_sentences)
        result_pdf = markdown_to_pdf(anonymized_markdown, pdf_bytes)

        if progress_callback:
            progress_callback(1.0, "Complete!")

        success_count = len(sentences) - len(errors)
        summary = f"Anonymized {success_count}/{len(sentences)} sentences"
        if errors:
            summary += f" ({len(errors)} errors)"

        return result_pdf, summary, errors

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        return None, f"Error processing PDF: {str(e)}", [error_detail]
