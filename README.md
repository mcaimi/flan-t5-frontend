# FLAN-T5 Frontend

This repository contains a Streamlit application for interacting with a KServe-based inference pipeline that processes text and PDF documents for anonymization, translation, and summarization tasks using configurable models.

## Features

- 🤖 Interactive web interface for FLAN-T5 model inference
- 📄 **PDF Anonymization**: Upload PDFs, extract text with docling, split into sentences using NLTK, anonymize each sentence, and download anonymized PDF with two modes:
  - **Presidio Mode** (Local PII Masking): Privacy-preserving local PII detection and masking using Microsoft Presidio
  - **KServe Mode** (AI Model): Context-aware anonymization using external AI model
- 📝 **PDF Summarization**: Upload PDFs, extract text with docling, split into sentences using NLTK, summarize each sentence, and download as PDF or TXT
- 🔒 **Text Anonymization** (optional): Single-sentence anonymization, translation, and summarization for quick text processing
- 🔄 Support for individual or batch processing of all tasks
- 📈 Real-time payload preview and response visualization
- 📜 Request history tracking
- 🔍 Automatic model discovery from KServe endpoints
- 💾 Response download capability (JSON and PDF)
- 🧠 **NLTK-based sentence splitting** for improved sentence boundary detection
- 🎨 **Unified diff display**: Red/green Git-style diff view showing before/after changes without truncation

## Files

- `main_streamlit.py`: Main Streamlit application entry point
- `config.yaml`: Feature flag and Presidio configuration file
- `libs/apis.py`: API-related functions for KServe v2 inference endpoints
- `libs/ui.py`: UI widget components for the Streamlit application
- `libs/pdf_processor.py`: PDF processing module using docling for text extraction, NLTK for sentence splitting, and KServe API for anonymization and summarization
- `libs/presidio.py`: Presidio integration for local PII detection and anonymization
- `libs/diff_utils.py`: Unified diff generation with red/green highlighting for before/after comparisons
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

The application uses `config.yaml` to enable/disable features and configure Presidio:

```yaml
features:
  text_anonymization: false  # Enable sentence-level text anonymization tab
  pdf_anonymization: true    # Enable PDF document anonymization tab
  pdf_summarization: true    # Enable PDF document summarization tab

presidio:
  spacy_model: en_core_web_lg  # spaCy model for Presidio NER (sm/md/lg/trf)
  gpu_enabled: auto            # GPU acceleration: auto, true, false
  device: auto                 # Device selection: auto, cuda, cuda:0, mps, cpu
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

### GPU Acceleration for Presidio

Presidio supports GPU acceleration on both NVIDIA GPUs (CUDA) and Apple Silicon (MPS) for faster PII detection:

**Configuration options:**

- **`gpu_enabled`**: Controls GPU usage
  - `auto` (default): Automatically use GPU if available, fallback to CPU
  - `true`: Force GPU usage, show warning if not available
  - `false`: Always use CPU even if GPU is available

- **`device`**: Specific device to use
  - `auto` (default): Auto-detect best device (CUDA > MPS > CPU)
  - `cuda` or `cuda:0`: NVIDIA GPU (first GPU)
  - `cuda:1`, `cuda:2`, etc.: Specific NVIDIA GPU (multi-GPU systems)
  - `mps`: Apple Silicon GPU (M1/M2/M3/M4 Macs)
  - `cpu`: CPU only

**Performance expectations:**

| Hardware | Speedup vs CPU | Best for |
|----------|----------------|----------|
| NVIDIA GPU | 2-3x (standard models)<br>5-10x (transformer models) | Large PDFs, transformer models (`en_core_web_trf`) |
| Apple Silicon (MPS) | 1.5-2.5x (standard models)<br>3-5x (transformer models) | M1/M2/M3/M4 Macs with macOS 12.3+ |
| CPU | Baseline | Small documents, development |

**Requirements:**

- **NVIDIA GPU**: CUDA-capable GPU with drivers installed (`nvidia-smi` works)
- **Apple Silicon**: M1/M2/M3/M4 Mac with macOS 12.3 or later
- PyTorch with CUDA/MPS support (already included in dependencies)

**GPU memory usage:**

| spaCy Model | NVIDIA GPU | Apple Silicon (MPS) |
|-------------|------------|---------------------|
| `en_core_web_sm` | ~0.5GB | Shares unified memory |
| `en_core_web_md` | ~1.0GB | Shares unified memory |
| `en_core_web_lg` | ~2.0GB | Shares unified memory |
| `en_core_web_trf` | ~4-6GB | Shares unified memory |

**Recommended configurations:**

```yaml
# NVIDIA GPU system
presidio:
  spacy_model: en_core_web_trf
  gpu_enabled: auto
  device: cuda:0

# Apple Silicon Mac
presidio:
  spacy_model: en_core_web_lg
  gpu_enabled: auto
  device: mps

# CPU-only (e.g., cloud deployment without GPU)
presidio:
  spacy_model: en_core_web_lg
  gpu_enabled: false
  device: cpu
```

**Monitoring GPU usage:**

The application displays device status in the sidebar under "🔧 Presidio Status":
- NVIDIA GPU: Shows GPU name and memory usage
- Apple Silicon: Shows "Apple Silicon GPU (MPS)"
- CPU: Shows "Device: CPU"

**Error handling:**

If GPU runs out of memory, the application automatically:
1. Clears GPU cache
2. Falls back to CPU
3. Retries the operation
4. Logs the fallback for debugging

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
- Anonymization mode selector:
  - **Presidio Mode**: Local privacy-preserving PII detection with configurable entity types (PERSON, EMAIL_ADDRESS, PHONE_NUMBER, etc.)
  - **KServe Mode**: AI model-based context-aware anonymization
- Automatic text extraction using docling
- Intelligent sentence splitting using NLTK (KServe mode only)
- Sentence-by-sentence anonymization via AI model (KServe mode) or full-text PII masking (Presidio mode)
- Progress tracking with real-time status
- **Unified diff view**: Red/green Git-style comparison showing all changes without truncation
- PII detection summary table (Presidio mode only)
- Download anonymized PDF
- Error reporting for failed sentences

**PDF Summarization Tab** (when `pdf_summarization: true`):
- PDF file upload (max 10MB)
- Automatic text extraction using docling
- Intelligent sentence splitting using NLTK
- Sentence-by-sentence summarization via AI model
- Progress tracking with real-time status
- Summary display in text area
- Download summary as PDF or TXT format
- Error reporting and detailed status messages

## PDF Processing Pipelines

### KServe Mode (AI Model) Pipeline

For PDF anonymization and summarization using the AI model:

1. **PDF → Markdown**: Extract text content using docling
2. **Markdown → Sentences**: Split text into sentences using NLTK's punkt tokenizer
3. **Process Sentences**: Each sentence is sent to the KServe AI model (anonymize or summarize)
4. **Sentences → Markdown**: Rebuild the document from processed sentences
5. **Markdown → PDF**: Convert back to PDF using reportlab

### Presidio Mode (Local PII Masking) Pipeline

For privacy-preserving local anonymization:

1. **PDF → Markdown**: Extract text content using docling
2. **Analyze Text**: Presidio analyzer detects PII entities (PERSON, EMAIL_ADDRESS, etc.) using spaCy NER
3. **Anonymize Text**: Presidio anonymizer replaces detected entities with placeholders (e.g., `<PERSON>`, `<EMAIL_ADDRESS>`)
4. **Markdown → PDF**: Convert anonymized text back to PDF using reportlab
5. **Generate Diff**: Create unified red/green diff showing all changes

### Key Technologies

- **docling**: High-quality PDF text extraction that preserves document structure
- **NLTK**: Advanced sentence tokenization that handles edge cases (abbreviations, decimals, etc.) better than regex-based approaches
- **KServe v2 API**: Scalable inference endpoint for processing individual sentences (KServe mode)
- **Presidio**: Microsoft's open-source PII detection and anonymization framework (Presidio mode)
  - **presidio-analyzer**: Detects PII entities using spaCy NER models
  - **presidio-anonymizer**: Masks detected entities with configurable operators
- **spaCy**: Industrial-strength NLP for named entity recognition (powers Presidio)
- **reportlab**: PDF generation from processed text
- **difflib**: Python standard library for generating unified diffs with red/green highlighting

### Benefits of Different Anonymization Modes

**Presidio Mode (Local PII Masking)**:
- **Privacy-preserving**: All processing happens locally, no data sent to external APIs
- **Fast**: No network latency, instant PII detection
- **Configurable**: Choose which entity types to detect and mask
- **Transparent**: See exactly what was detected with entity type and confidence scores
- **Deterministic**: Same input always produces the same output

**KServe Mode (AI Model)**:
- **Context-aware**: AI model understands context for more nuanced anonymization
- **Sentence-level processing**: Each sentence processed independently for better quality
- **Progress tracking**: Real-time feedback on which sentence is being processed
- **Error resilience**: Failed sentences don't block the entire document
- **Flexible**: Can be fine-tuned for specific domains or use cases

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
- Access to a running KServe inference server with FLAN-T5 models (for KServe mode)
- Dependencies:
  - streamlit>=1.56.0
  - requests>=2.33.1
  - docling>=2.0.0 (for PDF text extraction)
  - pypdf>=5.1.0 (for PDF manipulation)
  - reportlab>=4.2.0 (for PDF generation)
  - nltk>=3.8.0 (for sentence tokenization)
  - presidio-analyzer>=2.2.354 (for local PII detection)
  - presidio-anonymizer>=2.2.354 (for local PII masking)

### First-Time Setup

The application automatically downloads required models on first run:
- **NLTK punkt tokenizer**: Downloaded automatically for sentence splitting
- **spaCy language model**: Downloaded automatically for Presidio NER (default: `en_core_web_lg`)

You can configure which spaCy model to use in `config.yaml`:
```yaml
presidio:
  spacy_model: en_core_web_lg  # Options: en_core_web_sm, en_core_web_md, en_core_web_lg, en_core_web_trf
```

Model sizes and accuracy:
- `en_core_web_sm` (~12 MB): Small, fast, less accurate
- `en_core_web_md` (~40 MB): Medium, balanced
- `en_core_web_lg` (~560 MB): Large, most accurate (default)
- `en_core_web_trf` (~440 MB): Transformer-based, highest accuracy, slower

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


