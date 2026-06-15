"""API-related functions for KServe v2 inference endpoints."""

try:
    import requests
    import json
    from datetime import datetime
except ImportError:
    print("Error importing requests, json, or datetime")
    exit(1)

def build_payload(task, texts_dict):
    """Build the inference payload based on task selection and input texts."""
    inputs = []
    
    if task == "all":
        for task_name in ["anonymize", "translate", "summarize"]:
            if task_name in texts_dict and texts_dict[task_name]:
                inputs.append({
                    "name": task_name,
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [texts_dict[task_name]]
                })
    else:
        if task in texts_dict and texts_dict[task]:
            inputs.append({
                "name": task,
                "shape": [1],
                "datatype": "BYTES",
                "data": [texts_dict[task]]
            })
    
    return {"inputs": inputs}


def fetch_available_models(base_url):
    """Fetch available models from KServe v2 discovery endpoint."""
    discovery_url = f"{base_url.rstrip('/')}/v2/models"
    try:
        response = requests.get(discovery_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("models", []), None
    except requests.exceptions.Timeout:
        return [], "Request timeout - server took too long to respond"
    except requests.exceptions.ConnectionError:
        return [], f"Cannot connect to {discovery_url}"
    except requests.exceptions.HTTPError as e:
        return [], f"HTTP Error {response.status_code}: {str(e)}"
    except json.JSONDecodeError:
        return [], "Invalid JSON response from server"
    except Exception as e:
        return [], f"Unexpected error: {str(e)}"


def add_to_history(endpoint, task, status, elapsed_time):
    """Add a request to the history stored in session state.

    Note: This function requires streamlit session state to be available.
    Import streamlit when calling this function.
    """
    import streamlit as st

    if "request_history" not in st.session_state:
        st.session_state.request_history = []

    history_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "task": task,
        "endpoint": endpoint,
        "status": status,
        "elapsed_time": elapsed_time
    }

    st.session_state.request_history.insert(0, history_entry)

    if len(st.session_state.request_history) > 10:
        st.session_state.request_history = st.session_state.request_history[:10]


def anonymize_text_batch(texts, model_name, base_url, timeout=30):
    """Anonymize multiple text strings using the KServe API.

    This is a helper that wraps the single-request API for batch processing.
    Returns list of (anonymized_text, error) tuples.
    """
    endpoint_url = f"{base_url.rstrip('/')}/v2/models/{model_name}/infer"
    results = []

    for text in texts:
        try:
            payload = build_payload("anonymize", {"anonymize": text})
            response = requests.post(
                endpoint_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            response.raise_for_status()

            result_data = response.json()

            if "outputs" in result_data and len(result_data["outputs"]) > 0:
                output_data = result_data["outputs"][0].get("data", [])
                if output_data:
                    results.append((output_data[0], None))
                else:
                    results.append((text, "No output data in response"))
            else:
                results.append((text, "No outputs in response"))

        except requests.exceptions.Timeout:
            results.append((text, "Request timeout"))
        except requests.exceptions.RequestException as e:
            results.append((text, str(e)))
        except Exception as e:
            results.append((text, f"Unexpected error: {str(e)}"))

    return results
