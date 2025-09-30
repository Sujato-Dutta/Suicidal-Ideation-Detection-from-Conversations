# 🧠 Suicidal Ideation Detection from Conversations

A production-ready Streamlit application that analyzes conversational text for potential suicidal ideation using a fine‑tuned TinyBERT sequence classification model. The app focuses on clear results, high‑contrast UI, and responsible messaging.

> Important: This tool is for informational purposes only and must not replace professional assessment. If someone is at risk, contact local emergency services or crisis hotlines immediately.


## 🔍 Demo

App link: https://suicidal-ideation-detection-from-conversations-vubkqsyggfcgrov.streamlit.app/


## ✨ Features
- Fast text classification using fine‑tuned TinyBERT (for viewing the colab notebook -> https://dagshub.com/Sujato-Dutta/Suicidal-Ideation-Detection-from-Conversations/src/main/research/Suicide_Detection.ipynb)
- Single and batch text analysis
- Confidence visualization (matplotlib)
- Lightweight, responsive UI in Streamlit
- Simple logging and error handling
- Monitoring using mlflow
- Containerized using Docker

## 🧪 Model & Training
- Base: TinyBERT (Hugging Face Transformers)
- Fine‑tuning: Performed on Google Colab with GPU for faster training
- Artifacts exported and bundled with the repo for inference
- Default artifact path resolved via <mcfile name="model_loader.py" path="src/model_loader.py"></mcfile>

### Colab (GPU) Fine‑Tuning Summary
- Environment: Google Colab, GPU runtime
- Libraries: `transformers`, `datasets`, `torch`/`tensorflow`
- Steps:
  - Load dataset of labeled conversational texts
  - Tokenize with TinyBERT tokenizer
  - Fine‑tune TinyBERT for sequence classification
  - Save artifacts (config, tokenizer, weights) to project structure below

## 📦 Artifacts
Model files are stored.
<mcfolder name="fine_tuned_tinybert_suicide_detection" path="artifacts/fine_tuned_tinybert_suicide_detection/fine_tuned_tinybert_suicide_detection"></mcfolder>

Included files (typical):
- `config.json`
- `model.safetensors`
- `special_tokens_map.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_args.bin`
- `vocab.txt`

## 📂 Project Structure

```bash
├── .gitignore                                   # Ignore unnecessary files for Git
├── README.md                                    # Project documentation and overview
├── requirements.txt                             # Python dependencies
├── setup.py                                     # Package setup configuration
│
├── app/                                         # Streamlit app interface
│   └── streamlit_app.py                         # Main Streamlit script
│
├── artifacts/                                   # Saved model artifacts
│   └── fine_tuned_tinybert_suicide_detection/   # Fine-tuned TinyBERT model directory
│       └── fine_tuned_tinybert_suicide_detection/  # Model files and tokenizer
│
├── research/                                    # Jupyter research notebooks
│   └── Suicide_Detection.ipynb                  # Model experimentation and analysis
│
├── src/                                         # Source code for the pipeline
│   ├── __init__.py                              # Initialize source package
│   ├── model_loader.py                          # Load fine-tuned TinyBERT model
│   ├── model_predict.py                         # Inference and prediction logic
│
│   ├── monitoring/                              # Monitoring and tracking utilities
│   │   ├── __init__.py
│   │   └── mlflow_helper.py                     # MLflow tracking helper functions
│
│   ├── suicidal_ideation_pipeline.egg-info/     # Distribution metadata
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   ├── dependency_links.txt
│   │   ├── not-zip-safe
│   │   └── top_level.txt
│
│   └── utils/                                   # Utility modules
│       ├── __init__.py
│       ├── exceptions.py                        # Custom exception handling
│       └── logger.py                            # Logging configuration
```

## Author
Sujato Dutta | LinkedIn [https://www.linkedin.com/in/sujato-dutta/]
