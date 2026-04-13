"""UI widget components for the Streamlit application."""

try:
    import streamlit as st
    import json
    from datetime import datetime
except ImportError:
    print("Error importing streamlit, json, or datetime")
    exit(1)

def render_text_input(label, default, key, height=100, icon=""):
    """Render a text area with character count and optional icon."""
    display_label = f"{icon} {label}" if icon else label
    text = st.text_area(
        display_label,
        value=default,
        height=height,
        key=key,
        help=f"Enter text for {label.lower()}"
    )
    if text:
        char_count = len(text)
        st.caption(f"📊 {char_count:,} characters")
    return text


def display_payload_preview(payload, show_metadata=True):
    """Display payload preview with metadata and copy functionality."""
    if not payload.get("inputs"):
        st.info("No payload to display. Please enter text for the selected task(s).")
        return
    
    with st.container(border=True):
        st.subheader("📦 Payload Preview")
        
        if show_metadata:
            payload_str = json.dumps(payload, indent=2)
            payload_size = len(payload_str.encode('utf-8'))
            num_inputs = len(payload.get("inputs", []))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Inputs", num_inputs)
            with col2:
                st.metric("Size", f"{payload_size:,} bytes")
        
        st.json(payload, expanded=True)
        
        if st.button("📋 Copy to Clipboard", key="copy_payload", use_container_width=True):
            st.code(json.dumps(payload, indent=2), language="json")
            st.success("Payload displayed above - copy from the code block")


def display_response(response_data, elapsed_time, status_code):
    """Display API response with metrics and formatted output."""
    st.divider()
    st.subheader("📨 Response")
    
    # Extract and display data from outputs
    if "outputs" in response_data and isinstance(response_data["outputs"], list):
        st.subheader("🎯 Inference Results")
        for output in response_data["outputs"]:
            if "data" in output:
                task_name = output.get("name", "Output")
                data_values = output["data"]
                
                # Display as text area for better readability
                if isinstance(data_values, list) and len(data_values) > 0:
                    result_text = "\n\n".join(str(item) for item in data_values)
                else:
                    result_text = str(data_values)
                
                st.text_area(
                    f"{task_name.capitalize()} Result",
                    value=result_text,
                    height=150,
                    disabled=True,
                    key=f"result_{task_name}_{hash(result_text)}"
                )
    
    st.divider()
    
    # Metrics display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status Code", status_code, delta="Success" if status_code == 200 else "Error")
    with col2:
        st.metric("Response Time", f"{elapsed_time:.3f}s")
    with col3:
        response_size = len(json.dumps(response_data).encode('utf-8'))
        st.metric("Size", f"{response_size:,} bytes")
    
    with st.expander("📄 View Full Response Data", expanded=False):
        st.json(response_data)
    
    response_json = json.dumps(response_data, indent=2)
    st.download_button(
        label="💾 Download Response",
        data=response_json,
        file_name=f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )
