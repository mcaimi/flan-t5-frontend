# FLAN-T5 Frontend

This repository contains a Streamlit frontend implementation for interacting with an inference pipeline that processes text for anonymization, translation, and summarization tasks using FLAN-T5 models served through KServe.

## Features

- 🤖 Interactive web interface for FLAN-T5 model inference
- 🔒 Anonymization: Replaces personal information in text with placeholders
- 🌐 Translation: Translates text to another language
- 📄 Summarization: Creates concise summaries of long text
- 🔄 Support for individual or batch processing of all tasks
- 📊 Real-time payload preview and response visualization
- 📜 Request history tracking
- 🔍 Automatic model discovery from KServe endpoints
- 💾 Response download capability

## Files

- `main_streamlit.py`: Main Streamlit application entry point
- `libs/apis.py`: API-related functions for KServe v2 inference endpoints
- `libs/ui.py`: UI widget components for the Streamlit application
- `pyproject.toml`: Project dependencies and metadata (UV-managed)

## Installation

This is a UV-managed project. To install dependencies:

```bash
uv sync
```

## Usage

Start the Streamlit application:
```bash
uv run streamlit run main_streamlit.py
```

Or run directly with Python:
```bash
uv run python main_streamlit.py
```

## Application Interface

The Streamlit application provides a comprehensive web interface with:

### Configuration Panel (Sidebar)
- Base server URL configuration for KServe endpoint
- Model discovery and selection
- Request history tracking
- Clear history functionality

### Main Interface
- Task selection (Anonymize, Translate, Summarize, or All tasks)
- Dedicated text input areas for each task with character counting
- Live payload preview showing the exact JSON sent to the inference endpoint
- Send request button with real-time feedback
- Response visualization with metrics (status code, response time, size)
- Downloadable response data

## Configuration

Default server URL is `http://localhost:8080`. You can change this in the sidebar configuration panel.

The application automatically discovers available models from the KServe v2 discovery endpoint (`/v2/models`) and allows you to select which model to use for inference.

## Requirements

- Python 3.13+
- UV package manager
- Access to a running KServe inference server with FLAN-T5 models