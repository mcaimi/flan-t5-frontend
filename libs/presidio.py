"""Presidio integration module for PII detection and anonymization."""

try:
    import os
    from typing import Tuple, List, Optional, Callable
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from libs.pdf_processor import pdf_to_markdown, markdown_to_pdf
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    exit(1)


# Language code mapping for spaCy models
LANGUAGE_CODES = {
    'en': 'en',  # English
    'it': 'it',  # Italian
    'es': 'es',  # Spanish
    'de': 'de'   # German
}

# Model size to spaCy suffix mapping
MODEL_SIZES = {
    'small': 'sm',
    'medium': 'md',
    'large': 'lg'
}


def get_spacy_model_name(language: str, size: str) -> Tuple[str, str]:
    """Generate spaCy model name from language and size.

    Args:
        language: Language code ('en', 'it', 'es', 'de')
        size: Model size ('small', 'medium', 'large')

    Returns:
        Tuple of (model_name, lang_code)
        - model_name: Full spaCy model name (e.g., 'it_core_news_md')
        - lang_code: ISO language code for Presidio (e.g., 'it')
    """
    lang_code = LANGUAGE_CODES.get(language, 'en')
    size_suffix = MODEL_SIZES.get(size, 'lg')

    # English uses different naming pattern
    if language == 'en':
        model_name = f'en_core_web_{size_suffix}'
    else:
        model_name = f'{lang_code}_core_news_{size_suffix}'

    return model_name, lang_code


def detect_and_configure_gpu(gpu_config: dict) -> Tuple[str, str, str]:
    """Detect GPU/MPS availability and configure Presidio device.

    Supports NVIDIA GPUs (CUDA) and Apple Silicon (MPS).

    Args:
        gpu_config: Configuration dict with keys:
                   - gpu_enabled: 'auto', 'true'/'false', True/False
                   - device: 'auto', 'cuda', 'cuda:0', 'mps', 'cpu'

    Returns:
        Tuple of (device_name, device_type, status_message)
        - device_name: "cuda:0", "mps", "cpu", etc.
        - device_type: "nvidia", "apple", "cpu"
        - status_message: Human-readable status for logging/UI
    """
    try:
        import torch
    except ImportError:
        return 'cpu', 'cpu', 'PyTorch not available, using CPU'

    gpu_enabled = gpu_config.get('gpu_enabled', 'auto')
    device_preference = gpu_config.get('device', 'auto')

    # Check hardware availability
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

    # Helper function to get device info
    def get_device_info(device_str: str) -> Tuple[str, str, str]:
        """Returns (device, type, message)"""
        if device_str.startswith('cuda'):
            if cuda_available:
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    return device_str, 'nvidia', f'NVIDIA GPU: {gpu_name}'
                except Exception as e:
                    return device_str, 'nvidia', f'NVIDIA GPU (name unavailable: {e})'
            else:
                return 'cpu', 'cpu', 'WARNING: CUDA requested but not available, using CPU'
        elif device_str == 'mps':
            if mps_available:
                return 'mps', 'apple', 'Apple Silicon GPU (MPS)'
            else:
                return 'cpu', 'cpu', 'WARNING: MPS requested but not available, using CPU'
        elif device_str == 'cpu':
            return 'cpu', 'cpu', 'CPU (GPU disabled)'
        else:
            return 'cpu', 'cpu', f'WARNING: Unknown device "{device_str}", using CPU'

    # Force CPU mode
    if gpu_enabled in (False, 'false'):
        os.environ['PRESIDIO_DEVICE'] = 'cpu'
        return 'cpu', 'cpu', 'GPU disabled by configuration, using CPU'

    # Auto-detect best available device
    if device_preference == 'auto':
        if cuda_available:
            device = 'cuda:0'
            device_type = 'nvidia'
            try:
                gpu_name = torch.cuda.get_device_name(0)
                status = f'Auto-detected NVIDIA GPU: {gpu_name}'
            except Exception:
                status = 'Auto-detected NVIDIA GPU'
        elif mps_available:
            device = 'mps'
            device_type = 'apple'
            status = 'Auto-detected Apple Silicon GPU (MPS)'
        else:
            device = 'cpu'
            device_type = 'cpu'
            status = 'No GPU detected, using CPU'

        os.environ['PRESIDIO_DEVICE'] = device
        return device, device_type, status

    # Use specific device preference
    device, device_type, status = get_device_info(device_preference)

    # If GPU requested but not available, check gpu_enabled setting
    if device == 'cpu' and device_preference != 'cpu':
        if gpu_enabled in (True, 'true'):
            # Force mode - show warning but continue
            pass
        else:
            # Auto mode - try fallback to other GPU types
            if device_preference.startswith('cuda') and mps_available:
                device = 'mps'
                device_type = 'apple'
                status = 'CUDA not available, using Apple Silicon GPU (MPS)'
            elif device_preference == 'mps' and cuda_available:
                device = 'cuda:0'
                device_type = 'nvidia'
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    status = f'MPS not available, using NVIDIA GPU: {gpu_name}'
                except Exception:
                    status = 'MPS not available, using NVIDIA GPU'

    os.environ['PRESIDIO_DEVICE'] = device
    return device, device_type, status


def initialize_presidio(spacy_model: str = "en_core_web_lg", language_code: str = "en") -> Tuple[Optional[AnalyzerEngine], Optional[AnonymizerEngine], Optional[str]]:
    """Initialize Presidio analyzer and anonymizer engines.

    This function initializes the Presidio engines with caching support for Streamlit.
    If running in a Streamlit context, it will cache the engines in session state to
    avoid expensive reinitialization (2-5 seconds) on every rerun.

    Args:
        spacy_model: Name of the spaCy model to use for NER (default: en_core_web_lg).
                    Options: en_core_web_sm, en_core_web_md, en_core_web_lg, en_core_web_trf,
                    it_core_news_sm/md/lg, es_core_news_sm/md/lg, de_core_news_sm/md/lg
        language_code: ISO language code for Presidio analyzer (default: 'en').
                      Options: 'en', 'it', 'es', 'de'

    Returns:
        Tuple of (analyzer_engine, anonymizer_engine, error_message).
        If successful, error_message is None. If failed, engines are None.
    """
    try:
        # Check if we're in a Streamlit context and engines are already cached
        try:
            import streamlit as st
            if st.session_state.get("presidio_initialized", False):
                # Verify cached model and language match requested parameters
                if st.session_state.get("presidio_spacy_model") == spacy_model and \
                   st.session_state.get("presidio_language_code") == language_code:
                    return (
                        st.session_state.presidio_analyzer,
                        st.session_state.presidio_anonymizer,
                        None
                    )
                else:
                    # Model or language changed, reinitialize
                    print(f"Reinitializing Presidio with new model: {spacy_model}, language: {language_code}")
        except (ImportError, AttributeError, RuntimeError):
            # Not in Streamlit context or session state not available
            pass

        # Initialize engines with specified spaCy model
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language_code, "model_name": spacy_model}]
        }

        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        anonymizer = AnonymizerEngine()

        # Cache in session state if available
        try:
            import streamlit as st
            st.session_state.presidio_analyzer = analyzer
            st.session_state.presidio_anonymizer = anonymizer
            st.session_state.presidio_spacy_model = spacy_model
            st.session_state.presidio_language_code = language_code
            st.session_state.presidio_initialized = True
        except (ImportError, AttributeError, RuntimeError):
            # Not in Streamlit context, continue without caching
            pass

        return analyzer, anonymizer, None

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return None, None, f"Failed to initialize Presidio: {str(e)}\n{error_detail}"


def anonymize_text_with_presidio(
    text: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    entities: Optional[List[str]] = None,
    language: str = 'en'
) -> Tuple[Optional[str], List[dict], Optional[str]]:
    """Anonymize text using Presidio PII detection and anonymization.

    Args:
        text: Text to analyze and anonymize
        analyzer: Presidio AnalyzerEngine instance
        anonymizer: Presidio AnonymizerEngine instance
        entities: List of entity types to detect (e.g., ["PERSON", "EMAIL_ADDRESS"]).
                 If None, all default entities will be detected.
        language: ISO language code for text analysis (default: 'en').
                 Options: 'en', 'it', 'es', 'de'

    Returns:
        Tuple of (anonymized_text, detected_entities, error_message).
        detected_entities is a list of dicts with keys: entity_type, start, end, score, text.
        If successful, error_message is None. If failed, anonymized_text is None.
    """
    try:
        if not text or not text.strip():
            return None, [], "Empty text provided"

        if not analyzer or not anonymizer:
            return None, [], "Presidio engines not initialized"

        # Analyze text for PII
        results = analyzer.analyze(
            text=text,
            language=language,
            entities=entities if entities else None
        )

        if not results:
            # No PII found - return original text with empty entities list
            return text, [], None

        # Anonymize the text
        anonymized_result = anonymizer.anonymize(
            text=text,
            analyzer_results=results
        )

        # Convert results to serializable format for display
        detected_entities = [
            {
                'entity_type': result.entity_type,
                'start': result.start,
                'end': result.end,
                'score': result.score,
                'text': text[result.start:result.end]
            }
            for result in results
        ]

        return anonymized_result.text, detected_entities, None

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return None, [], f"Anonymization error: {str(e)}\n{error_detail}"


def generate_markdown_diff(
    original: str,
    anonymized: str,
    detected_entities: List[dict]
) -> str:
    """Generate a markdown-formatted before/after comparison.

    DEPRECATED: Use libs.diff_utils.generate_anonymization_diff() instead.
    This function is kept for backwards compatibility.

    Creates a markdown display showing:
    - Summary table of detected PII entity types and counts
    - Original text section
    - Anonymized text section

    Args:
        original: Original text before anonymization
        anonymized: Anonymized text with PII masked
        detected_entities: List of detected entity dicts from anonymize_text_with_presidio()

    Returns:
        Markdown-formatted string for display in Streamlit
    """
    from libs.diff_utils import generate_anonymization_diff
    return generate_anonymization_diff(
        original,
        anonymized,
        detected_entities,
        anonymization_mode="presidio"
    )


def anonymize_pdf_with_presidio(
    pdf_file,
    entities: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    spacy_model: str = "en_core_web_lg",
    language_code: str = "en"
) -> Tuple[Optional[bytes], Optional[str], Optional[str], List[dict], Optional[str]]:
    """Main orchestration function for Presidio-based PDF anonymization.

    This function implements the complete pipeline:
    1. PDF → Markdown (via docling)
    2. Analyze and anonymize markdown with Presidio
    3. Markdown → PDF (via reportlab)

    Args:
        pdf_file: PDF file object (must support .read())
        entities: List of entity types to detect (e.g., ["PERSON", "EMAIL_ADDRESS"]).
                 If None, all default entities will be detected.
        progress_callback: Optional callback function(percent: float, message: str)
                          to report progress updates
        spacy_model: Name of the spaCy model to use for NER (default: en_core_web_lg)
        language_code: ISO language code for Presidio analyzer (default: 'en').
                      Options: 'en', 'it', 'es', 'de'

    Returns:
        Tuple of (pdf_bytes, original_markdown, anonymized_markdown,
                  detected_entities, error_message).
        - pdf_bytes: Anonymized PDF as bytes, or None if failed
        - original_markdown: Original text extracted from PDF
        - anonymized_markdown: Anonymized version of the text
        - detected_entities: List of detected PII entities
        - error_message: Error description if failed, None if successful
    """
    try:
        if progress_callback:
            progress_callback(0.05, "Reading PDF...")

        # Read PDF bytes
        pdf_bytes = pdf_file.read()

        # Validate file size (max 10MB)
        if len(pdf_bytes) > 10 * 1024 * 1024:
            return None, None, None, [], "Error: PDF file exceeds 10MB limit"

        if progress_callback:
            progress_callback(0.1, "Converting PDF to markdown with docling...")

        # Step 1: Convert PDF to markdown using docling (reuse from pdf_processor)
        original_markdown = pdf_to_markdown(pdf_bytes)

        if not original_markdown or not original_markdown.strip():
            return None, None, None, [], "Error: No text extracted from PDF"

        if progress_callback:
            progress_callback(0.3, f"Initializing Presidio engines with {spacy_model}...")

        # Step 2: Initialize Presidio with specified spaCy model and language
        analyzer, anonymizer, init_error = initialize_presidio(spacy_model, language_code)
        if init_error:
            return None, None, None, [], f"Presidio initialization failed: {init_error}"

        if progress_callback:
            progress_callback(0.4, "Analyzing text for PII...")

        # Step 3: Anonymize the markdown text with Presidio
        try:
            anonymized_markdown, detected_entities, anon_error = anonymize_text_with_presidio(
                original_markdown,
                analyzer,
                anonymizer,
                entities,
                language_code
            )

            if anon_error:
                return None, None, None, [], f"Anonymization failed: {anon_error}"

        except (Exception,) as e:
            # Handle GPU/MPS out-of-memory errors
            import torch
            is_oom = False

            # Check for CUDA OOM
            if hasattr(torch.cuda, 'OutOfMemoryError') and isinstance(e, torch.cuda.OutOfMemoryError):
                is_oom = True
            # Check for MPS OOM (raises RuntimeError with "out of memory" message)
            elif isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                is_oom = True

            if is_oom:
                print(f"⚠ GPU/MPS out of memory - retrying with CPU (Error: {e})")

                # Clear GPU cache
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception as cache_err:
                    print(f"  Warning: Could not clear GPU cache: {cache_err}")

                # Fallback to CPU
                os.environ['PRESIDIO_DEVICE'] = 'cpu'
                print("  Reinitializing Presidio engines with CPU...")

                # Reinitialize engines with CPU
                analyzer, anonymizer, init_error = initialize_presidio(spacy_model, language_code)
                if init_error:
                    return None, None, None, [], f"CPU fallback initialization failed: {init_error}"

                # Retry anonymization with CPU
                if progress_callback:
                    progress_callback(0.4, "Retrying with CPU after GPU OOM...")

                anonymized_markdown, detected_entities, anon_error = anonymize_text_with_presidio(
                    original_markdown,
                    analyzer,
                    anonymizer,
                    entities,
                    language_code
                )

                if anon_error:
                    return None, None, None, [], f"Anonymization failed after CPU fallback: {anon_error}"
            else:
                # Not an OOM error, re-raise
                raise

        if progress_callback:
            progress_callback(0.8, "Rebuilding PDF from anonymized text...")

        # Step 4: Convert anonymized markdown back to PDF (reuse from pdf_processor)
        result_pdf = markdown_to_pdf(anonymized_markdown, pdf_bytes)

        if not result_pdf:
            return None, original_markdown, anonymized_markdown, detected_entities, "Error: Failed to generate PDF"

        if progress_callback:
            progress_callback(1.0, "Complete!")

        return result_pdf, original_markdown, anonymized_markdown, detected_entities, None

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return None, None, None, [], f"Error: {str(e)}\n{error_detail}"
