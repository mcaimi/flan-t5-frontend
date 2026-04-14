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

### Deployment in containerized environments

The default behaviour can be also modified via environment variables: this is useful when deploying in containerized environments (e.g. k8s)

#### Available config variables:

- KSERVE_ENDPOINT_ADDRESS: The address of the KServe v2 inference endpoint. Defaults to "http://localhost:8080"
- MODEL_NAME: The selected model name. This must match the name of the deployed model in the KServe Engine. Defaults to "None" (e.g. you need to perform discovery before inference)

## Requirements

- Python 3.13+
- UV package manager
- Access to a running KServe inference server with FLAN-T5 models

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

## Depoloy on OCP

```bash
$ oc new-app -e KSERVE_ENDPOINT_ADDRESS=<KSERVE_ENDPOINT> --image=<IMAGE:TAG> --name=flan-t5-frontend
$ oc create route edge flan-t5-frontend --service=flan-t5-frontend --port=8501-tcp
```


