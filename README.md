# TikTok Skincare Product Sentiment Analysis (CeraVe Focus)

A comprehensive multimodal pipeline designed to collect, process, extract features from, and perform sentiment analysis on TikTok review videos focused on **CeraVe** skincare products. The project leverages speech-to-text, facial expression analysis, audio classification, language identification, and LLM-based (Qwen 7B) few-shot classification to categorize video product reviews into **Positive**, **Negative**, or **Neutral** sentiment classes.

---

## Overview

Social media platforms like TikTok have become primary channels for consumer product reviews, especially within the skincare and beauty industry. However, analyzing sentiment in short-form video content presents unique challenges compared to traditional text reviews, as sentiment is expressed through multiple modalities—spoken language, facial expressions, and acoustic cues.

This project focuses specifically on analyzing TikTok user reviews and feedback for **CeraVe** skincare products, providing an end-to-end pipeline that:
1. **Scrapes** TikTok videos and metadata related to CeraVe skincare products.
2. **Preprocesses and cleans** raw metadata by removing duplicates, correcting mislabeled instances, refining transcripts to fix spelling and domain-specific product/brand names (e.g., *CeraVe*, *Cleanser*), filtering out non-English/Arabic videos, and removing music-only videos that lack spoken speech.
3. **Extracts Multimodal Features**, including speech transcripts (via Speech-to-Text), and facial expression feature.
4. **Performs LLM-based Sentiment Analysis** using a few-shot prompted **Qwen 7B** model running in a GPU-accelerated environment.
5. **Evaluates** classification performance using standard evaluation metrics and visual confusion matrices.

---

## Project Pipeline

```
  ┌───────────────────────────────┐
  │    TikTok Data Collection     │  (data_collection/tiktok_scraper.py)
  └───────────────┬───────────────┘
                  │
                  ▼
  ┌───────────────────────────────┐
  │   Data Preprocessing & Fix    │  (process_data/dedup.py, fix_transcript.py, etc.)
  └───────────────┬───────────────┘
                  │
                  ▼
  ┌───────────────────────────────┐
  │ Multimodal Feature Extraction │  (extract/ speech_to_text, facial_expression.)
  └───────────────┬───────────────┘
                  │
                  ▼
  ┌───────────────────────────────┐
  │  Prompt Engineering (Few-Shot)│  (prompt/prompt_fewshot.txt)
  └───────────────┬───────────────┘
                  │
                  ▼
  ┌───────────────────────────────┐
  │     LLM Sentiment Analysis    │  (sentiment_analysis/ [Qwen 7B on GPU / Colab])
  └───────────────┬───────────────┘
                  │
                  ▼
  ┌───────────────────────────────┐
  │  Evaluation & Visualizations  │  (results/ classification_report.csv, confusion_matrix.png)
  └───────────────┬───────────────┘
```

---

## Features

- **Automated TikTok Scraping:** Collects CeraVe skincare product review videos and associated metadata.
- **Multimodal Feature Extraction (`extract/`):**
  - **Speech-to-Text:** Transcribes spoken audio from TikTok videos into text.
  - **Facial Expression Analysis:** Extracts visual emotion/expression cues from video frames.
- **Data Cleaning & Quality Control (`process_data/ and extract/`):**
  - Automated deduplication of video entries.
  - Identification and correction or removal of mislabeled neutral data.
  - Transcript enhancement including domain-specific  (e.g., brand names like *CeraVe* and product terms like *Cleanser*).
  - spelling correction for transcript using **spelling_correction**
  - Filtering out non-English/Arabic content using **language_id_model** and music-only videos without speech using **audio_classification** 
- **LLM-based Few-Shot Sentiment Classification:** Utilizes open-source LLMs (Qwen 7B) with tailored prompt templates for accurate 3-class sentiment prediction (**Positive**, **Negative**, **Neutral**).
- **Automated Evaluation Pipeline:** Generates detailed classification reports and visual confusion matrices saved automatically to the `results/` directory.

---

## Project Structure

```
.
├── data/
│   ├── final_dataset.csv              # Processed dataset ready for inference/evaluation
│   └── tiktok_results_all.csv         # Raw scraped dataset from TikTok scraper
├── data_collection/
│   └── tiktok_scraper.py              # Script for scraping TikTok videos and metadata
├── extract/
│   ├── audio_classifier/              # Audio extraction & classification modules
│   ├── facial_expression_model/       # Facial expression detection & analysis
│   ├── language_id_model/             # Language identification filtering
│   ├── speech_to_text/                # Audio transcription module
│   ├── spelling_correction/           # Text transcript spelling correction module
│   ├── __init__.py
│   └── bais.py                        # Orchestrates video downloading, audio extraction, and frame extraction for multimodal processing.
├── process_data/
│   ├── __init__.py
│   ├── dedup.py                       # Data deduplication script
│   ├── fix_or_remove_mislabeled_neutral.py # Handlers for mislabeled neutral sentiment instances
│   └── fix_transcript.py              # Transcript cleaning, spelling correction, and normalization
├── prompt/
│   └── prompt_fewshot.txt             # Few-shot prompt template for LLM inference
├── results/
│   ├── classification_report.csv     # Metrics per sentiment class (Precision, Recall, F1)
│   ├── confusion_matrix.csv          # Confusion matrix raw count values
│   ├── confusion_matrix.png          # Visual confusion matrix heatmap plot
│   └── predictions_df.csv            # Detailed model predictions against ground truth
├── sentiment_analysis/
│   ├── notebook/
│   │   └── qwen_7b_sentiment_analysis.ipynb # Google Colab workflow notebook
│   ├── qwen_7b/
│   │   ├── __init__.py
│   │   ├── inference.py               # Qwen 7B inference utilities
│   │   └── model.py                   # Model initialization and loader
│   ├── evaluate.py                    # Evaluation metric calculation and plot generation
│   └── run_sentiment_analysis.py      # Entry point for GPU execution of sentiment analysis
├── .env                               # Environment variables configuration (API keys, path overrides)
├── .gitignore                         # Git ignore configurations
├── config.py                          # Global configuration settings and path declarations
├── main.py                            # Main application entry point / pipeline orchestration
└── requirements.txt                   # Project dependencies
```

---

## Data Processing

Before passing data into the sentiment classification model, raw collected data undergoes extensive preprocessing:

| Processing Module | File                                                              | Purpose                                                                                                                     |
| :--- |:------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|
| **Deduplication** | `process_data/dedup.py`                                           | Removes duplicate video entries harvested during scraping.                                                                  |
| **Neutral Label Fix** | `process_data/fix_or_remove_mislabeled_neutral.py`                | Identifies ambiguously labeled neutral rows and cleans/removes them to improve dataset quality.                             |
| **Transcript Fix & Spelling Correction** | `process_data/fix_transcript.py` & `extract/spelling_correction/` | Corrects speech-to-text spelling errors, specifically domain-specific brand/product terms (e.g., *CeraVe*, *Cleanser*).     |
| **Data Filtering** | `extract/language_id_model/` & `extract/audio_classifier/`        | Filters out non-English/Arabic videos as well as music-only videos that lack spoken verbal content.                         |
| **Multimodal Feature Extraction** | `extract/facial_expression_model/` & `extract/speech-to-text/`    | Extracts signals (facial expressions, Speech-to-Text)                                                                       |

---

## Sentiment Analysis

The core sentiment analysis engine relies on large language modeling using the **Qwen 7B** architecture.

- **Classes:** Positive, Negative, Neutral
- **Inference Strategy:** Few-shot in-context learning guided by predefined examples in `prompt/prompt_fewshot.txt`.
- **Core Modules:**
  - `sentiment_analysis/qwen_7b/model.py`: Handles model initialization, tokenizer loading, and precision settings.
  - `sentiment_analysis/qwen_7b/inference.py`: Executes batched inference on cleaned transcripts and extracted metadata.
  - `sentiment_analysis/run_sentiment_analysis.py`: Main driver script executing inference on GPU.

---

## Google Colab / GPU Requirement

> **⚠️ CRITICAL EXECUTION NOTE:**
> The sentiment analysis stage relies on **Qwen 7B**, which requires substantial VRAM (NVIDIA GPU with at least 16GB+ VRAM recommended, e.g., T4, V100, A100).

- **Local Execution vs. GPU/Colab:**
  - TikTok Data Collection (`data_collection/tiktok_scraper.py`),data preprocessing (`process_data/`), feature extractionc(`extract/facial_expression_model/` & `extract/speech-to-text/` ) and evaluation scripts (`sentiment_analysis/evaluate.py`) can be executed on standard CPU devices locally.
  - The inference engine (`sentiment_analysis/run_sentiment_analysis.py`) and notebook (`sentiment_analysis/notebook/qwen_7b_sentiment_analysis.ipynb`) **MUST** be run on a GPU-enabled environment (e.g., Google Colab Pro, AWS EC2 GPU instance, or local CUDA workstation).
- **Interactive Workflow:** Use `sentiment_analysis/notebook/qwen_7b_sentiment_analysis.ipynb` in Google Colab to run the sentiment analysis interactively and reproduce results.

---

## Prompt Engineering

Prompt design plays a fundamental role in directing the Qwen 7B model to output structured, high-accuracy sentiment classifications.

- **Template Location:** `prompt/prompt_fewshot.txt`
- **Design Strategy:** Uses **few-shot learning**, providing explicit examples of **CeraVe** skincare product feedback along with multimodal context cues.
- **Output Constraint:** Directs the LLM to output clear classifications into one of the three target categories: `Positive`, `Negative`, or `Neutral`.

---

## Evaluation

Model predictions are saved and evaluated using `sentiment_analysis/evaluate.py`. Outputs are compiled inside the `results/` folder:

- **`results/predictions_df.csv`**: Contains raw transcripts, ground truth labels, extracted features, and predicted labels generated by the model.
- **`results/classification_report.csv`**: Detailed metrics including Precision, Recall, and F1-score breakdown per sentiment class.
- **`results/confusion_matrix.csv`**: Numerical tabular confusion matrix detailing true vs. predicted counts.
- **`results/confusion_matrix.png`**: High-resolution rendered heatmap visualizing model confusion patterns across classes.

---

## Local File Paths Warning

> **⚠️ IMPORTANT NOTICE ON DATASET PATHS:**
>
> The columns `audio_path` and `video_path` within `data/final_dataset.csv`, `data/tiktok_results_all.csv`, and `results/predictions_df.csv` contain absolute/relative local paths native to the developer's original machine.
>
> **These paths will NOT resolve automatically on another system.** When setting up the project on a new environment, you must:
> 1. Download/re-extract audio and video files locally.
> 2. Update the path references in your local dataset copies or configure `config.py` / `.env` to reflect your local directory structure before running feature extraction scripts.

---

## Installation

### Prerequisites
- Python 3.9+
- Git
- NVIDIA GPU with CUDA support (required for LLM inference stage)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create or edit the `.env` file in the project root to set up local environment variables:
```bash
# Example .env configuration
DATA_DIR=./data
RESULTS_DIR=./results
# Ensure secret credentials or local API keys are defined here and NEVER committed to Git.
```

---
