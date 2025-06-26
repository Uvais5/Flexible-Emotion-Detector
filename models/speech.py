# models/speech.py
# ------------------------------------------------------------
# Audio-emotion inference (MFCC ➜ NN ➜ label, probability)
# ------------------------------------------------------------
from pathlib import Path
import numpy as np
import librosa
from keras.models import load_model

# ---------- 1. one-time setup --------------------------------
MODEL_PATH = Path("models/emotion_audio_model.h5")          # adapt if needed
LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "pleasure",   # lower-case for consistency
    6: "sad",
}

# cache the network so we don’t hit disk every call
_audio_model = load_model(MODEL_PATH)


# ---------- 2. feature extractor (kept as you wrote it) -------
def extract_mfcc(filename, n_mfcc=40):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    return mfcc


# ---------- 3. main API --------------------------------------
def sound_detection(filename):
    """
    Parameters
    ----------
    filename : str | Path
        Path to an audio file (wav/ogg/etc.).

    Returns
    -------
    label : str
        Predicted emotion label.
    prob  : float
        Soft-max probability (0-1) of that label.
    """
    # ➊ Extract features and shape for the model
    feats = extract_mfcc(filename).reshape(1, -1).astype("float32")

    # ➋ Predict
    softmax = _audio_model.predict(feats, verbose=0)[0]   # shape (7,)
    best_idx = int(np.argmax(softmax))
    label = LABELS[best_idx]
    prob = float(softmax[best_idx])

    return label, prob
