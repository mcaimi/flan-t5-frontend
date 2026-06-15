# FLAN-T5 Frontend

This repository contains a Streamlit application for interacting with a KServe-based inference pipeline that processes text and PDF documents for anonymization, translation, and summarization tasks using configurable models.

## Features

- 🤖 Interactive web interface for FLAN-T5 model inference
- 📄 **PDF Anonymization**: Upload PDFs, extract text with docling, anonymize sentences, and download anonymized PDF (default mode)
- 📝 **PDF Summarization**: Upload PDFs, extract text with docling, generate AI-powered summaries, and download as PDF or TXT
- 🔒 **Sentence Anonymization** (optional): Single-sentence anonymization for quick text processing
- 🌐 Translation: Translates text to another language (optional mode)
- 📊 Summarization: Creates concise summaries of long text (optional mode)
- 🔄 Support for individual or batch processing of all tasks
- 📈 Real-time payload preview and response visualization
- 📜 Request history tracking
- 🔍 Automatic model discovery from KServe endpoints
- 💾 Response download capability (JSON and PDF)

## Files

- `main_streamlit.py`: Main Streamlit application entry point
- `config.yaml`: Feature flag configuration file
- `libs/apis.py`: API-related functions for KServe v2 inference endpoints
- `libs/ui.py`: UI widget components for the Streamlit application
- `libs/pdf_processor.py`: PDF processing module using docling for text extraction and anonymization
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

The available features are controlled by the `config.yaml` file in the project root.

## Configuration

The application uses `config.yaml` to enable/disable features:

```yaml
features:
  text_anonymization: false  # Enable sentence-level text anonymization tab
  pdf_anonymization: true    # Enable PDF document anonymization tab
  pdf_summarization: true    # Enable PDF document summarization tab
```

### Feature Flags

- **`text_anonymization`**: When `true`, shows the Text Anonymization interface with support for single-sentence anonymization, translation, and summarization tasks.
- **`pdf_anonymization`**: When `true`, shows the PDF Anonymization interface for uploading and processing PDF documents.
- **`pdf_summarization`**: When `true`, shows the PDF Summarization interface for uploading PDFs and generating AI-powered summaries.

### Feature Combinations

The application dynamically creates tabs based on enabled features. You can enable any combination:

1. **Single Feature** (shows single container, no tabs):
   ```yaml
   features:
     text_anonymization: false
     pdf_anonymization: true
     pdf_summarization: false
   ```

2. **Two Features** (shows two tabs):
   ```yaml
   features:
     text_anonymization: false
     pdf_anonymization: true
     pdf_summarization: true
   ```

3. **All Features** (shows three tabs):
   ```yaml
   features:
     text_anonymization: true
     pdf_anonymization: true
     pdf_summarization: true
   ```

4. **No Features**:
   ```yaml
   features:
     text_anonymization: false
     pdf_anonymization: false
     pdf_summarization: false
   ```
   Displays a courtesy message: "There are no features enabled for this instance"

## Application Interface

The Streamlit application provides a comprehensive web interface with:

### Configuration Panel (Sidebar)
- Base server URL configuration for KServe endpoint
- Model discovery and selection
- Request history tracking
- Clear history functionality

### Main Interface

**Text Anonymization Tab** (when `text_anonymization: true`):
- Task selection (Anonymize, Translate, Summarize, or All tasks)
- Dedicated text input areas for each task with character counting
- Live payload preview showing the exact JSON sent to the inference endpoint
- Send request button with real-time feedback
- Response visualization with metrics (status code, response time, size)
- Downloadable response data

**PDF Anonymization Tab** (when `pdf_anonymization: true`):
- PDF file upload (max 10MB)
- Automatic text extraction using docling
- Sentence-level anonymization via AI model
- Progress tracking with real-time status
- Download anonymized PDF
- Error reporting for failed sentences

**PDF Summarization Tab** (when `pdf_summarization: true`):
- PDF file upload (max 10MB)
- Automatic text extraction using docling
- AI-powered document summarization
- Progress tracking with real-time status
- Summary display in text area
- Download summary as PDF or TXT format
- Error reporting and detailed status messages

## Runtime Configuration

### KServe Endpoint Configuration

Default server URL is `http://localhost:8080`. You can change this in the sidebar configuration panel.

The application automatically discovers available models from the KServe v2 discovery endpoint (`/v2/models`) and allows you to select which model to use for inference.

### Deployment in Containerized Environments

The default behaviour can be also modified via environment variables: this is useful when deploying in containerized environments (e.g. k8s)

#### Available config variables:

- KSERVE_ENDPOINT_ADDRESS: The address of the KServe v2 inference endpoint. Defaults to "http://localhost:8080"
- MODEL_NAME: The selected model name. This must match the name of the deployed model in the KServe Engine. Defaults to "None" (e.g. you need to perform discovery before inference)

## Requirements

- Python 3.13+
- UV package manager
- Access to a running KServe inference server with FLAN-T5 anonymization models
- Dependencies:
  - streamlit>=1.56.0
  - requests>=2.33.1
  - docling>=2.0.0 (for PDF text extraction)
  - pypdf>=5.1.0 (for PDF manipulation)
  - reportlab>=4.2.0 (for PDF generation)

## Building the image on Openshift using OCP BuildConfigs

A Containerfile is provided for automated builds on OCP.

1. Generate a token to access the target container registry (Optional) (e.g. a Robot Account for Quay.io)

2. Create a push secret to access the target registry (Optional)

```bash
$ oc create secret docker-registry push-secret --docker-server=<SERVER URL> --docker-username=<USERNAME> --docker-password=<PASSWORD> -n <NAMESPACE>
```

3. Create a build config

```yaml
#
# Private Git: create a Secret (basic-auth or ssh) and set spec.source.sourceSecret.
#
apiVersion: build.openshift.io/v1
kind: BuildConfig
metadata:
  name: flan-t5-frontend
spec:
  runPolicy: Serial
  source:
    type: Git
    git:
      uri: https://codeberg.org/mcaimi/flan-t5-frontend
      ref: main
    # sourceSecret:
    #   name: git-credentials
  strategy:
    type: Docker
    dockerStrategy:
      dockerfilePath: Containerfile
  output:
    to:
      kind: DockerImage
      name: <TARGET CONTAINER REGISTRY IMAGE URL+TAG> # fill in the required information
    pushSecret:
      name: push-secret # optional secret created at step 2
```

4. Run the build on OCP

```bash
$ oc apply -f buildconfig.yaml
$ oc start-build flan-t5-frontend
```

## Deploy on OCP

```bash
$ oc new-app -e KSERVE_ENDPOINT_ADDRESS=<KSERVE_ENDPOINT> --image=<IMAGE:TAG> --name=flan-t5-frontend
$ oc create route edge flan-t5-frontend --service=flan-t5-frontend --port=8501-tcp
```


