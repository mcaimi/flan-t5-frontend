try:
    import streamlit as st
    import requests
    import json
    import time
    import os
    import yaml
    from datetime import datetime
    from pathlib import Path
except ImportError:
    print("Error importing streamlit, requests, json, time, yaml, or datetime")
    exit(1)

# Import from libs modules
try:
    from libs.apis import build_payload, fetch_available_models, add_to_history
    from libs.ui import render_text_input, display_payload_preview, display_response
    from libs.pdf_processor import anonymize_pdf, summarize_pdf
    from libs.presidio import anonymize_pdf_with_presidio
    from libs.diff_utils import generate_anonymization_diff
except ImportError:
    print("Error importing apis, ui, pdf_processor, or presidio")
    exit(1)

# Ensure NLTK punkt tokenizer is available at startup
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab', quiet=False)
        print("NLTK punkt_tab tokenizer downloaded successfully")
except Exception as e:
    print(f"Warning: Could not verify NLTK punkt tokenizer: {e}")
    # Continue anyway - will be checked again in pdf_processor if needed

# Load configuration from config.yaml
def load_config():
    """Load configuration from config.yaml file."""
    config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print("❌ config.yaml file not found. Using default configuration.")
        return {'features': {}, 'presidio': {}}
    except Exception as e:
        print(f"❌ Error loading config.yaml: {str(e)}. Using default configuration.")
        return {'features': {}, 'presidio': {}}

# Load configuration
config = load_config()
features = config.get('features', {})
presidio_config = config.get('presidio', {})
ENABLE_TEXT_ANONYMIZATION = features.get('text_anonymization', False)

# Ensure spaCy language model is available for Presidio at startup
# Get configured spaCy model (default: en_core_web_lg)
SPACY_MODEL = presidio_config.get('spacy_model', 'en_core_web_lg')

try:
    import spacy
    try:
        # Try to load the configured model to verify it exists
        spacy.load(SPACY_MODEL)
        print(f"spaCy {SPACY_MODEL} model is available")
    except OSError:
        print(f"Downloading spaCy {SPACY_MODEL} language model (this may take a few minutes)...")
        import subprocess
        import sys
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", SPACY_MODEL
        ])
        print(f"spaCy {SPACY_MODEL} model downloaded successfully")
except Exception as e:
    print(f"Warning: Could not verify spaCy language model: {e}")
    print("Presidio PII anonymization may not work without the language model")
    # Continue anyway - will show error in UI if Presidio is used
ENABLE_PDF_ANONYMIZATION = features.get('pdf_anonymization', False)
ENABLE_PDF_SUMMARIZATION = features.get('pdf_summarization', False)

# Page configuration
st.set_page_config(
    page_title="Inference Pipeline Client",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# load endpoint address from environment variable
endpoint_address = os.getenv("KSERVE_ENDPOINT_ADDRESS", "http://localhost:8080")

# load model name from environment variable
model_name = os.getenv("MODEL_NAME", None)

# Initialize session state
if "request_history" not in st.session_state:
    st.session_state.request_history = []

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "available_models" not in st.session_state:
    st.session_state.available_models = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_name

if "base_url" not in st.session_state:
    st.session_state.base_url = endpoint_address

if "selected_task" not in st.session_state:
    st.session_state.selected_task = "all"

# Presidio session state
if "presidio_analyzer" not in st.session_state:
    st.session_state.presidio_analyzer = None

if "presidio_anonymizer" not in st.session_state:
    st.session_state.presidio_anonymizer = None

if "presidio_initialized" not in st.session_state:
    st.session_state.presidio_initialized = False

# Default texts
DEFAULT_TEXTS = {
    "anonymize": "Write your text here",
    "translate": "Translate this text to another language",
    "summarize": "Why KServe? KServe eliminates the complexity of productionizing AI models. Whether you're a data scientist wanting to deploy your latest LLM experiment, a DevOps engineer building scalable ML infrastructure, or a decision maker evaluating AI platforms, KServe provides a unified solution that works across clouds and scales with your needs. From Experiment to Production in Minutes - Deploy GenAI services and ML models with simple YAML configurations, no complex infrastructure setup required. Cloud-Agnostic by Design - Run anywhere: AWS, Azure, GCP, on-premises, or hybrid environments with consistent behavior. Enterprise-Scale Ready - Automatically handle traffic spikes, scale to zero when idle, and manage hundreds of models efficiently."
}

# Check if any features are enabled
if not ENABLE_TEXT_ANONYMIZATION and not ENABLE_PDF_ANONYMIZATION and not ENABLE_PDF_SUMMARIZATION:
    st.title("🤖 Inference Pipeline Client")
    st.info("ℹ️ There are no features enabled for this instance")
    st.markdown("""
    Please check your `config.yaml` file and enable at least one feature:
    - `text_anonymization`: Enable sentence-level text anonymization
    - `pdf_anonymization`: Enable PDF document anonymization
    - `pdf_summarization`: Enable PDF document summarization
    """)
    st.stop()

# App Header
st.title("🤖 Inference Pipeline Client")
st.markdown("Configure the HTTP endpoint and send requests to the inference pipeline")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Base URL input
base_url = st.sidebar.text_input(
    "Base Server URL",
    value=st.session_state.base_url,
    help="Base URL of the KServe inference server (environment variable: KSERVE_ENDPOINT_ADDRESS)",
    key="base_url_input"
)

# Update session state if base URL changes
if base_url != st.session_state.base_url:
    st.session_state.base_url = base_url
    st.session_state.available_models = []
    st.session_state.selected_model = None

# Discover Models button with model count
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    discover_button = st.button("🔍 Discover Models", use_container_width=True, key="discover_btn")
with col2:
    if st.session_state.available_models:
        st.success(f"✓ {len(st.session_state.available_models)}")

# Handle model discovery
if discover_button:
    with st.spinner("🔄 Discovering models..."):
        models, error = fetch_available_models(base_url)
        if error:
            st.sidebar.error(f"❌ {error}")
            st.session_state.available_models = []
        else:
            st.session_state.available_models = models
            if models:
                st.sidebar.success(f"✅ Found {len(models)} model(s)")
            else:
                st.sidebar.warning("⚠️ No models found")

# Model selection dropdown
if st.session_state.available_models:
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=st.session_state.available_models,
        index=0 if st.session_state.selected_model is None else 
              (st.session_state.available_models.index(st.session_state.selected_model) 
               if st.session_state.selected_model in st.session_state.available_models else 0),
        help="Choose a model for inference"
    )
    st.session_state.selected_model = selected_model
    
    # Build inference endpoint URL
    endpoint_url = f"{base_url.rstrip('/')}/v2/models/{selected_model}/infer"
    
    # Display constructed endpoint
    st.sidebar.text_input(
        "Inference Endpoint",
        value=endpoint_url,
        disabled=True,
        help="Auto-generated inference endpoint URL"
    )
    
    # Model information display
    with st.sidebar.expander("ℹ️ Model Info", expanded=False):
        st.markdown(f"""
        **Selected Model:** `{st.session_state.selected_model}`  
        **Endpoint:** `{endpoint_url}`  
        **Protocol:** KServe v2
        """)
else:
    if st.session_state.selected_model is not None:
        st.sidebar.info(f"Using {st.session_state.selected_model} as the default model. 👆 Click 'Discover Models' to fetch available models")
        print(f"Using model name from environment variable: {st.session_state.selected_model}")
        endpoint_url = f"{base_url.rstrip('/')}/v2/models/{st.session_state.selected_model}/infer"
    else:
        st.sidebar.info("👆 Click 'Discover Models' to fetch available models")
        endpoint_url = None

st.sidebar.divider()

# Request History in Sidebar
with st.sidebar.expander("📜 Request History", expanded=False):
    if st.session_state.request_history:
        for idx, entry in enumerate(st.session_state.request_history):
            status_icon = "✅" if entry["status"] == "success" else "❌"
            st.markdown(f"""
            **{status_icon} {entry['timestamp']}**  
            Task: `{entry['task']}`  
            Time: {entry['elapsed_time']:.3f}s
            """)
            if idx < len(st.session_state.request_history) - 1:
                st.markdown("---")
    else:
        st.info("No requests yet")

if st.session_state.request_history:
    if st.sidebar.button("🗑️ Clear History", use_container_width=True):
        st.session_state.request_history = []
        st.rerun()

# Main Content Area - Conditionally show tabs based on enabled features
enabled_features = []
if ENABLE_TEXT_ANONYMIZATION:
    enabled_features.append("text")
if ENABLE_PDF_ANONYMIZATION:
    enabled_features.append("pdf_anon")
if ENABLE_PDF_SUMMARIZATION:
    enabled_features.append("pdf_sum")

# Initialize containers
text_container = None
pdf_container = None
pdf_summary_container = None

if len(enabled_features) == 0:
    st.error("No features enabled")
elif len(enabled_features) == 1:
    # Single feature - no tabs needed
    if ENABLE_TEXT_ANONYMIZATION:
        text_container = st.container()
    elif ENABLE_PDF_ANONYMIZATION:
        pdf_container = st.container()
    elif ENABLE_PDF_SUMMARIZATION:
        pdf_summary_container = st.container()
elif len(enabled_features) == 2:
    # Two features - show two tabs
    tab_labels = []
    if ENABLE_TEXT_ANONYMIZATION:
        tab_labels.append("💬 Text Anonymization")
    if ENABLE_PDF_ANONYMIZATION:
        tab_labels.append("📄 PDF Anonymization")
    if ENABLE_PDF_SUMMARIZATION:
        tab_labels.append("📝 PDF Summarization")

    tab1, tab2 = st.tabs(tab_labels)

    # Assign containers based on which features are enabled
    tab_idx = 0
    if ENABLE_TEXT_ANONYMIZATION:
        text_container = tab1 if tab_idx == 0 else tab2
        tab_idx += 1
    if ENABLE_PDF_ANONYMIZATION:
        pdf_container = tab1 if tab_idx == 0 else tab2
        tab_idx += 1
    if ENABLE_PDF_SUMMARIZATION:
        pdf_summary_container = tab1 if tab_idx == 0 else tab2
        tab_idx += 1
else:
    # All three features enabled - show three tabs
    tab1, tab2, tab3 = st.tabs([
        "💬 Text Anonymization",
        "📄 PDF Anonymization",
        "📝 PDF Summarization"
    ])
    text_container = tab1
    pdf_container = tab2
    pdf_summary_container = tab3

# Tab 1: Text Anonymization (only if enabled)
if ENABLE_TEXT_ANONYMIZATION:
    with text_container:
        # Two Column Layout
        left_col, right_col = st.columns([3, 2], gap="large")

        # Left Column - Input Forms and Actions
        with left_col:
            st.header("📝 Input Configuration")

            # Task selector
            selected_task = st.radio(
                "Select Task",
                options=["all", "anonymize", "translate", "summarize"],
                format_func=lambda x: {
                    "all": "🔄 All Tasks",
                    "anonymize": "🔒 Anonymize",
                    "translate": "🌐 Translate",
                    "summarize": "📄 Summarize"
                }[x],
                horizontal=True,
                key="task_selector"
            )

            st.divider()

            texts_dict = {}

            # Display inputs based on selected task
            if selected_task == "all":
                st.markdown("### Configure all tasks at once")
                texts_dict["anonymize"] = render_text_input(
                    "Anonymize Text",
                    DEFAULT_TEXTS["anonymize"],
                    "anonymize_all",
                    height=100,
                    icon="🔒"
                )
                texts_dict["translate"] = render_text_input(
                    "Translate Text",
                    DEFAULT_TEXTS["translate"],
                    "translate_all",
                    height=100,
                    icon="🌐"
                )
                texts_dict["summarize"] = render_text_input(
                    "Summarize Text",
                    DEFAULT_TEXTS["summarize"],
                    "summarize_all",
                    height=150,
                    icon="📄"
                )

            elif selected_task == "anonymize":
                st.markdown("### 🔒 Anonymize Personal Information")
                st.info("This task replaces personal information (names, addresses) with placeholders")
                texts_dict["anonymize"] = render_text_input(
                    "Text to Anonymize",
                    DEFAULT_TEXTS["anonymize"],
                    "anonymize_single",
                    height=150
                )

            elif selected_task == "translate":
                st.markdown("### 🌐 Translate Text")
                st.info("This task translates text to another language")
                texts_dict["translate"] = render_text_input(
                    "Text to Translate",
                    DEFAULT_TEXTS["translate"],
                    "translate_single",
                    height=150
                )

            elif selected_task == "summarize":
                st.markdown("### 📄 Summarize Long Text")
                st.info("This task creates a concise summary of longer text")
                texts_dict["summarize"] = render_text_input(
                    "Text to Summarize",
                    DEFAULT_TEXTS["summarize"],
                    "summarize_single",
                    height=200
                )

            current_task = selected_task

            st.divider()

            # Send Request Button
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                send_button = st.button("🚀 Send Request", type="primary", use_container_width=True)
            with col_btn2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.rerun()

            # Handle Request
            if send_button:
                if not endpoint_url:
                    st.error("⚠️ Please discover models and select one before sending a request")
                    st.info("💡 Use the sidebar to enter your base URL and click 'Discover Models'")
                    st.stop()

                payload = build_payload(current_task, texts_dict)

                if not payload.get("inputs"):
                    st.warning("⚠️ Please enter text for at least one task")
                else:
                    with st.spinner("🔄 Sending request to inference pipeline..."):
                        start_time = time.time()
                        try:
                            headers = {"Content-Type": "application/json"}
                            response = requests.post(endpoint_url, headers=headers, json=payload, timeout=30)
                            elapsed_time = time.time() - start_time

                            response.raise_for_status()
                            response_data = response.json()

                            st.session_state.last_response = {
                                "data": response_data,
                                "elapsed_time": elapsed_time,
                                "status_code": response.status_code
                            }

                            add_to_history(endpoint_url, current_task, "success", elapsed_time)

                            st.success("✅ Request successful!")
                            display_response(response_data, elapsed_time, response.status_code)

                        except requests.exceptions.Timeout:
                            elapsed_time = time.time() - start_time
                            add_to_history(endpoint_url, current_task, "error", elapsed_time)
                            st.error("⏱️ Request timeout - the server took too long to respond")

                        except requests.exceptions.ConnectionError:
                            elapsed_time = time.time() - start_time
                            add_to_history(endpoint_url, current_task, "error", elapsed_time)
                            st.error(f"🔌 Connection error - cannot reach {endpoint_url}")
                            st.info("💡 Make sure the inference service is running and accessible")

                        except requests.exceptions.HTTPError as e:
                            elapsed_time = time.time() - start_time
                            add_to_history(endpoint_url, current_task, "error", elapsed_time)
                            st.error(f"❌ HTTP Error {response.status_code}: {str(e)}")
                            if response.text:
                                with st.expander("📄 Error Details"):
                                    st.code(response.text)

                        except Exception as e:
                            elapsed_time = time.time() - start_time
                            add_to_history(endpoint_url, current_task, "error", elapsed_time)
                            st.error(f"❌ Unexpected error: {str(e)}")

            # Display last successful response if exists
            if st.session_state.last_response and not send_button:
                with st.expander("📨 Last Response", expanded=False):
                    display_response(
                        st.session_state.last_response["data"],
                        st.session_state.last_response["elapsed_time"],
                        st.session_state.last_response["status_code"]
                    )

        # Right Column - Live Payload Preview
        with right_col:
            st.header("👁️ Live Preview")
            current_payload = build_payload(current_task, texts_dict)
            display_payload_preview(current_payload, show_metadata=True)

            st.divider()

            with st.container(border=True):
                st.subheader("ℹ️ About")
                st.markdown("""
                **Available Tasks:**
                - 🔒 **Anonymize**: Replaces personal info with placeholders
                - 🌐 **Translate**: Converts text to another language
                - 📄 **Summarize**: Creates concise summaries

                **Tips:**
                - Use the radio buttons to switch between tasks
                - Preview updates in real-time
                - View request history in sidebar
                """)

# Tab 2: PDF Anonymization (only if enabled)
if ENABLE_PDF_ANONYMIZATION:
    with pdf_container:
        st.header("📄 PDF Anonymization")
        st.markdown("Upload a PDF file to anonymize all PII while preserving the original layout")

        if not st.session_state.selected_model:
            st.warning("⚠️ Please discover and select a model from the sidebar first")
            st.stop()

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload PDF Document",
                type=["pdf"],
                accept_multiple_files=False,
                help="Upload a PDF file (max 10MB)"
            )

            if uploaded_file:
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.info(f"📄 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")

        with col2:
            st.markdown("### Settings")

            # Anonymization mode selector
            anonymization_mode = st.radio(
                "Anonymization Mode",
                options=["presidio", "kserve"],
                format_func=lambda x: {
                    "presidio": "🔒 Presidio (Local PII Masking)",
                    "kserve": "🤖 KServe (AI Model)"
                }[x],
                help="Choose anonymization method",
                key="anon_mode_selector"
            )

            # Show model selector only for KServe mode
            if anonymization_mode == "kserve":
                st.text_input(
                    "Model",
                    value=st.session_state.selected_model,
                    disabled=True,
                    help="Model selected in sidebar"
                )
            else:
                # Entity type selector for Presidio
                entity_options = st.multiselect(
                    "PII Entity Types",
                    options=[
                        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                        "CREDIT_CARD", "US_SSN", "LOCATION",
                        "DATE_TIME", "URL", "IP_ADDRESS"
                    ],
                    default=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
                    help="Select which PII types to detect and mask",
                    key="presidio_entities"
                )

        st.divider()

        if uploaded_file:
            process_button = st.button("🔒 Anonymize PDF", type="primary", use_container_width=True)

            if process_button:
                start_time = time.time()

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)

                try:
                    if anonymization_mode == "presidio":
                        # Presidio pipeline
                        pdf_bytes, original_md, anonymized_md, entities, error = anonymize_pdf_with_presidio(
                            uploaded_file,
                            entity_options if entity_options else None,
                            update_progress,
                            SPACY_MODEL
                        )

                        elapsed_time = time.time() - start_time

                        if pdf_bytes:
                            st.success(f"✅ Anonymized with Presidio - Found {len(entities)} PII instances")
                            add_to_history(
                                "presidio_local",
                                "pdf_anonymize_presidio",
                                "success",
                                elapsed_time
                            )

                            # Display diff
                            if original_md and anonymized_md:
                                diff_html = generate_anonymization_diff(
                                    original_md,
                                    anonymized_md,
                                    entities,
                                    anonymization_mode="presidio"
                                )
                                with st.expander("📊 Before/After Comparison", expanded=True):
                                    st.markdown(diff_html, unsafe_allow_html=True)

                            # Download button
                            st.download_button(
                                label="⬇️ Download Anonymized PDF",
                                data=pdf_bytes,
                                file_name=f"anonymized_{uploaded_file.name}",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        else:
                            st.error(f"❌ {error}")
                            add_to_history(
                                "presidio_local",
                                "pdf_anonymize_presidio",
                                "error",
                                elapsed_time
                            )

                    else:
                        # KServe pipeline
                        result_pdf, original_md, anonymized_md, summary, errors = anonymize_pdf(
                            uploaded_file,
                            st.session_state.selected_model,
                            st.session_state.base_url,
                            update_progress
                        )

                        elapsed_time = time.time() - start_time

                        if result_pdf:
                            st.success(f"✅ {summary}")
                            add_to_history(
                                f"{st.session_state.base_url}/v2/models/{st.session_state.selected_model}/infer",
                                "pdf_anonymize",
                                "success",
                                elapsed_time
                            )

                            # Display diff
                            if original_md and anonymized_md:
                                diff_html = generate_anonymization_diff(
                                    original_md,
                                    anonymized_md,
                                    detected_entities=None,
                                    anonymization_mode="kserve"
                                )
                                with st.expander("📊 Before/After Comparison", expanded=True):
                                    st.markdown(diff_html, unsafe_allow_html=True)

                            # Download button
                            st.download_button(
                                label="⬇️ Download Anonymized PDF",
                                data=result_pdf,
                                file_name=f"anonymized_{uploaded_file.name}",
                                mime="application/pdf",
                                use_container_width=True
                            )

                            # Show errors if any
                            if errors:
                                with st.expander(f"⚠️ {len(errors)} Errors During Processing", expanded=False):
                                    for error in errors:
                                        st.text(f"• {error}")
                        else:
                            st.error(f"❌ {summary}")
                            add_to_history(
                                f"{st.session_state.base_url}/v2/models/{st.session_state.selected_model}/infer",
                                "pdf_anonymize",
                                "error",
                                elapsed_time
                            )

                except Exception as e:
                    elapsed_time = time.time() - start_time
                    st.error(f"❌ Unexpected error: {str(e)}")
                    add_to_history(
                        "presidio_local" if anonymization_mode == "presidio" else f"{st.session_state.base_url}/v2/models/{st.session_state.selected_model}/infer",
                        "pdf_anonymize_presidio" if anonymization_mode == "presidio" else "pdf_anonymize",
                        "error",
                        elapsed_time
                    )
        else:
            st.info("👆 Upload a PDF file to get started")

            with st.container(border=True):
                st.subheader("ℹ️ How It Works")
                st.markdown("""
                **Presidio Mode (Local PII Masking)**:
                1. **Upload**: Select a PDF file (max 10MB)
                2. **Convert**: PDF is converted to markdown using docling
                3. **Analyze**: Presidio detects PII entities locally
                4. **Anonymize**: PII is replaced with entity type placeholders
                5. **Rebuild**: Anonymized text is converted back to PDF
                6. **Download**: Get your anonymized PDF with before/after diff

                **KServe Mode (AI Model)**:
                1. **Upload**: Select a PDF file (max 10MB)
                2. **Convert**: PDF is converted to markdown using docling
                3. **Split**: Markdown is split into individual sentences using NLTK
                4. **Anonymize**: Each sentence is processed through the AI model
                5. **Rebuild**: Anonymized text is converted back to PDF
                6. **Download**: Get your anonymized PDF

                **Note**: Presidio runs locally and provides privacy-preserving PII detection.
                KServe uses an external AI model for context-aware anonymization.
                """)

# Tab 3: PDF Summarization (only if enabled)
if ENABLE_PDF_SUMMARIZATION:
    with pdf_summary_container:
        st.header("📝 PDF Summarization")
        st.markdown("Upload a PDF file to generate an AI-powered summary")

        if not st.session_state.selected_model:
            st.warning("⚠️ Please discover and select a model from the sidebar first")
            st.stop()

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_summary_file = st.file_uploader(
                "Upload PDF Document",
                type=["pdf"],
                accept_multiple_files=False,
                help="Upload a PDF file to summarize (max 10MB)",
                key="pdf_summary_uploader"
            )

            if uploaded_summary_file:
                file_size_mb = len(uploaded_summary_file.getvalue()) / (1024 * 1024)
                st.info(f"📄 **{uploaded_summary_file.name}** ({file_size_mb:.2f} MB)")

        with col2:
            st.markdown("### Settings")
            st.text_input(
                "Model",
                value=st.session_state.selected_model,
                disabled=True,
                help="Model selected in sidebar",
                key="pdf_summary_model_display"
            )

        st.divider()

        if uploaded_summary_file:
            summarize_button = st.button("📝 Summarize PDF", type="primary", use_container_width=True, key="summarize_pdf_btn")

            if summarize_button:
                start_time = time.time()

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)

                try:
                    summary_text, summary_pdf, status_msg, errors = summarize_pdf(
                        uploaded_summary_file,
                        st.session_state.selected_model,
                        st.session_state.base_url,
                        update_progress
                    )

                    elapsed_time = time.time() - start_time

                    if summary_text and summary_pdf:
                        st.success(f"✅ {status_msg}")
                        add_to_history(
                            f"{st.session_state.base_url}/v2/models/{st.session_state.selected_model}/infer",
                            "pdf_summarize",
                            "success",
                            elapsed_time
                        )

                        # Display the summary
                        st.subheader("📄 Summary")
                        st.text_area(
                            "Generated Summary",
                            value=summary_text,
                            height=300,
                            disabled=True,
                            key="summary_display"
                        )

                        st.divider()

                        # Download buttons
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                label="⬇️ Download Summary (PDF)",
                                data=summary_pdf,
                                file_name=f"summary_{uploaded_summary_file.name}",
                                mime="application/pdf",
                                use_container_width=True,
                                key="download_summary_pdf"
                            )
                        with col_dl2:
                            st.download_button(
                                label="⬇️ Download Summary (TXT)",
                                data=summary_text,
                                file_name=f"summary_{uploaded_summary_file.name}.txt",
                                mime="text/plain",
                                use_container_width=True,
                                key="download_summary_txt"
                            )

                        # Show errors if any
                        if errors:
                            with st.expander(f"⚠️ {len(errors)} Issues During Processing", expanded=False):
                                for error in errors:
                                    st.text(f"• {error}")
                    else:
                        st.error(f"❌ {status_msg}")
                        add_to_history(
                            f"{st.session_state.base_url}/v2/models/{st.session_state.selected_model}/infer",
                            "pdf_summarize",
                            "error",
                            elapsed_time
                        )
                        if errors:
                            with st.expander("📄 Error Details", expanded=False):
                                for error in errors:
                                    st.text(f"• {error}")

                except Exception as e:
                    elapsed_time = time.time() - start_time
                    st.error(f"❌ Unexpected error: {str(e)}")
                    add_to_history(
                        f"{st.session_state.base_url}/v2/models/{st.session_state.selected_model}/infer",
                        "pdf_summarize",
                        "error",
                        elapsed_time
                    )
        else:
            st.info("👆 Upload a PDF file to get started")

            with st.container(border=True):
                st.subheader("ℹ️ How It Works")
                st.markdown("""
                1. **Upload**: Select a PDF file (max 10MB)
                2. **Convert**: PDF is converted to markdown using docling
                3. **Summarize**: The entire document is sent to the AI model for summarization
                4. **Display**: The generated summary is displayed on the page
                5. **Download**: Get your summary as PDF or plain text

                **Note**: Uses docling for high-quality PDF text extraction.
                The AI model generates a concise summary of the entire document.
                """)
