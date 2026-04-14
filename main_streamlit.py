try:
    import streamlit as st
    import requests
    import json
    import time
    import os
    from datetime import datetime
except ImportError:
    print("Error importing streamlit, requests, json, time, or datetime")
    exit(1)

# Import from libs modules
try:
    from libs.apis import build_payload, fetch_available_models, add_to_history
    from libs.ui import render_text_input, display_payload_preview, display_response
except ImportError:
    print("Error importing apis or ui")
    exit(1)

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

# Default texts
DEFAULT_TEXTS = {
    "anonymize": "Write your text here",
    "translate": "Translate this text to another language",
    "summarize": "Why KServe? KServe eliminates the complexity of productionizing AI models. Whether you're a data scientist wanting to deploy your latest LLM experiment, a DevOps engineer building scalable ML infrastructure, or a decision maker evaluating AI platforms, KServe provides a unified solution that works across clouds and scales with your needs. From Experiment to Production in Minutes - Deploy GenAI services and ML models with simple YAML configurations, no complex infrastructure setup required. Cloud-Agnostic by Design - Run anywhere: AWS, Azure, GCP, on-premises, or hybrid environments with consistent behavior. Enterprise-Scale Ready - Automatically handle traffic spikes, scale to zero when idle, and manage hundreds of models efficiently."
}

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

# Main Content Area - Two Column Layout
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
