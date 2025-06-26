# 🎭 Flexible Emotion Detector

Detect human emotions from **video 🎦**, **audio 🎧**, or **text 💬** — all in one flexible and interactive Streamlit app.

<p align="center">
  <img src="docs/front_view.png" alt="App screenshot" width="640">
</p>

---

## 🧩 Project Overview

Flexible Emotion Detector is a **multi-modal emotion recognition system** that adapts to different input types — video, audio, or text. It extracts emotional cues from:

- **Facial expressions** in videos
- **Speech signals** in audio
- **Words and phrases** in spoken or written text

This tool integrates deep learning models for vision, audio, and NLP to give a unified emotion prediction experience. It is ideal for applications like affective computing, mood analysis, and emotional insight extraction in human–AI interaction.

---

## 💡 Problem It Solves

- ✅ Users often don’t have the same input format — some only provide audio, some video, others just typed or spoken text.
- ✅ Most emotion models work on only **one modality**.
- ✅ This system adapts to **whatever input is available** — making it useful in real-world settings like:
  - Virtual therapy & mood tracking
  - Voice assistants
  - Social media content analysis
  - Human behavior research

---

## ✨ Key Features
| Modality | What Happens | Model |
|----------|--------------|-------|
| **Face** | Picks the emotion that appears most often across all faces in the video. | CNN (FER-2013) |
| **Speech** | Extracts MFCC features → LSTM → emotion from vocal tone. | Audio model (`emotion_audio_model.h5`) |
| **Text** | Speech-to-text → EmoRoBERTa sentiment classifier. | Hugging Face pipeline |

Switch modalities on/off in the sidebar. Upload **video (mp4)**, **audio (wav/mp3)**, or just paste text.

---

## 🚀 Quick Start (local)

```bash
# clone
git clone https://github.com/Uvais5/Flexible-Emotion-Detector.git
cd Flexible-Emotion-Detector

# create env & install deps
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# launch app
streamlit run new_main.py         # open browser at http://localhost:8501

