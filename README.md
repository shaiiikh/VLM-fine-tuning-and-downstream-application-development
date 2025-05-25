# Vision-Language Fine-Tuning + Multimodal Mini-App

This repository contains solutions for **Assignment 06** from the Generative AI course. The assignment is divided into two parts:

1. **Q1 – Fine-Tuning a Generative VLM for Image Describing**
2. **Q2 – Ask-the-Image Multimodal Mini-App (Speech → Image QA → Speech)**

---

## 📌 Q1: Fine-Tuning a Generative VLM for Image Describing

### ✅ Overview

In this task, we fine-tuned a pre-trained **BLIP-2** model on a small image–caption dataset to produce detailed visual descriptions and short image narratives.

### 🔧 Objectives

- Load and preprocess a curated dataset (Flickr8K subset).
- Fine-tune BLIP-2 using prompt tuning and full fine-tuning.
- Experiment with **decoding strategies**: greedy, beam search, top-k, top-p.
- Evaluate using automatic metrics like BLEU-4, METEOR, ROUGE-L, SPICE.
- Conduct qualitative analysis on hallucination, coverage, and fluency.

### 📊 Results

| Metric   | Value (Test Set) |
|----------|------------------|
| BLEU-4   | > 25             |
| CIDEr    | > 60             |
| SPICE    | ~18              |
| Distinct-2 | High diversity |

> Descriptions contain ≥ 3 key elements in 80% of the test samples.

### 🛠️ Tools & Libraries

- `transformers`, `torchvision`, `datasets`, `nltk`, `scikit-learn`
- Model: `BLIP-2 + Flan-T5`
- Framework: Hugging Face + PyTorch

### 📁 File

- `Q1.ipynb`: Fine-tuning pipeline, metrics, and evaluation

---

## 🧠 Q2: Ask-the-Image – Multimodal Mini-App

### ✅ Overview

This is a **proof-of-concept app** where users can ask a question by voice about an uploaded image, and the app replies both visually and vocally.

### 💡 Features

1. **Speech-to-Text**
   - Record user speech (≤ 10 seconds)
   - Transcribe using `Whisper-small`

2. **Image + Question → Answer**
   - Upload an image
   - Use **BLIP-2 + Flan-T5-small** to answer transcribed question

3. **Text-to-Speech**
   - Display answer
   - Speak the answer aloud using `pyttsx3` or `gTTS`

### 🛠️ Tech Stack

- `whisper`, `transformers`, `gradio` / `streamlit`, `pyttsx3`, `PIL`, `torch`
- Modular scripts:
  - `asr.py`: Audio transcription
  - `qa.py`: VQA model
  - `tts.py`: Speech synthesis
  - `app.py`: UI integration

### 📊 Evaluation

| Metric        | Value |
|---------------|-------|
| ASR WER       | Low (< 10%) on 10 samples |
| QA Accuracy   | ~80% on 10 diverse image-question pairs |
| Latency       | ~3.2 seconds end-to-end |


## 📁 Project Structure
├── Q1.ipynb # Fine-tuning notebook (VLM)
├── ask-the-image/
│ ├── asr.py # Whisper-based ASR
│ ├── qa.py # Visual Question Answering
│ ├── tts.py # Text-to-Speech
│ ├── app.py # Integration and UI
│ ├── requirements.txt
│ └── demo-assets/ # Optional: sample images/audio
├── README.md # You are here 📄



---

## 📢 How to Run

### ⚙️ Setup

```bash
# Clone repo and enter directory
git clone https://github.com/your-username/assignment-06-vlm.git
cd assignment-06-vlm

# Set up Python environment
pip install -r ask-the-image/requirements.txt


cd ask-the-image
python app.py
