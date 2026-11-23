# 🦷 Oral Disease Classifier (ResNet50V2)

An end-to-end deep learning web application for classifying oral disease images using a fine-tuned **ResNet50V2** model, built with **Streamlit** and integrated with **Hugging Face Hub** for automatic model weight retrieval.

## 📌 Overview

This application classifies clinical/oral images into **6 categories**:
- **Calculus**
- **Data caries**
- **Gingivitis**
- **Mouth Ulcer**
- **Tooth Discoloration**
- **Hypodontia**

## ✨ Key Features

- **🔍 Diagnosis Tab:** Upload oral images (`PNG`, `JPG`, `JPEG`, `WEBP`), view confidence-aware predictions, and inspect full class probability distributions.
- **📊 Model Results Tab:** Displays saved evaluation metrics (`results/metrics.json`), training curves, confusion matrix, model comparison table (`results/comparison.csv`), and persistent prediction history (`results/history.json`).
- **☁️ Automatic Hub Model Loading:** Checks for a local `ResNet50V2.keras` file first and automatically downloads it from [`mo123essam/oral-disease-resnet50v2`](https://huggingface.co/mo123essam/oral-disease-resnet50v2) if not found locally.
- **🔧 Cross-Version Keras Compatibility:** Automatically strips unsupported `quantization_config` metadata when loading `.keras` archives across differing TensorFlow/Keras environments.

## 🗂️ Project Structure

```text
├── app.py              # Streamlit web application & inference pipeline
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── results/            # Optional metrics, plots, comparison table & history
