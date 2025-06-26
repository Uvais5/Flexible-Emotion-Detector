# backend.py  –  core pipeline with zero side-effects on import
from pathlib import Path
import moviepy.editor as mp
import speech_recognition as sr

from models import face, speech, text

AUDIO_DIR = Path("models/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)       # once at import

# ------------------------------------------------------------------
def video_to_audio(video_path: Path) -> Path:
    """Return path to WAV extracted from *video_path*."""
    wav_path = AUDIO_DIR / f"{video_path.stem}.wav"
    mp.VideoFileClip(str(video_path)).audio.write_audiofile(
        wav_path, logger=None
    )
    return wav_path

# ------------------------------------------------------------------
import speech_recognition as sr
from pathlib import Path

def audio_to_text(wav_path: Path, chunk_duration=10) -> str:
    """
    Convert long audio to text using chunking for better accuracy.
    chunk_duration is in seconds (default = 10).
    """
    r = sr.Recognizer()
    full_text = []

    try:
        with sr.AudioFile(str(wav_path)) as source:
            audio_length = int(source.DURATION)  # in seconds
            for i in range(0, audio_length, chunk_duration):
                audio_chunk = r.record(source, duration=chunk_duration)
                try:
                    chunk_text = r.recognize_google(audio_chunk)
                    full_text.append(chunk_text)
                except sr.UnknownValueError:
                    continue  # skip chunk if not understood
                except sr.RequestError as e:
                    raise RuntimeError(f"Google Speech Recognition failed: {e}") from e
    except Exception as e:
        print(f"[ERROR] audio_to_text(): {e}")
        return ""

    return " ".join(full_text)


# ------------------------------------------------------------------

# ───────── Video ─────────
def detect_from_video(
    video_path: Path,
    do_face=True, do_speech=True, do_text=True,
) -> dict:
    """Full pipeline when the input is a *video* file."""
    video_path = Path(video_path).expanduser().resolve()

    # Face ───────────────────────────────
    face_label, face_prob = ("N/A", 0.0)
    if do_face:
        face_label, face_prob = face.face_detection_novideo(str(video_path))

    # Audio track ────────────────────────
    wav_path = video_to_audio(video_path)

    # Speech ─────────────────────────────
    speech_label, speech_prob = ("N/A", 0.0)
    if do_speech:
        speech_label, speech_prob = speech.sound_detection(str(wav_path))

    # Text  ──────────────────────────────
    text_label, text_prob, transcript = "N/A", 0.0, ""
    if do_text:
        transcript = audio_to_text(wav_path)
        if transcript:
            text_label, text_prob = _label_and_prob_from_text(transcript)

    return _assemble(face_label, face_prob,
                     speech_label, speech_prob,
                     text_label, text_prob,
                     transcript)

# ───────── Audio ─────────
def detect_from_audio(
    audio_path: Path,
    do_speech=True, do_text=True,
) -> dict:
    """Pipeline when the input is an *audio* file."""
    audio_path = Path(audio_path).expanduser().resolve()

    face_label, face_prob = "N/A", 0.0                     # no face

    speech_label, speech_prob = ("N/A", 0.0)
    if do_speech:
        speech_label, speech_prob = speech.sound_detection(str(audio_path))

    text_label, text_prob, transcript = "N/A", 0.0, ""
    if do_text:
        transcript = audio_to_text(audio_path)
        if transcript:
            text_label, text_prob = _label_and_prob_from_text(transcript)

    return _assemble(face_label, face_prob,
                     speech_label, speech_prob,
                     text_label, text_prob,
                     transcript)

# ───────── Raw text ───────
def detect_from_text(raw_text: str) -> dict:
    """Pipeline when the user directly supplies text."""
    face_label, face_prob = "N/A", 0.0
    speech_label, speech_prob = "N/A", 0.0
    text_label, text_prob = _label_and_prob_from_text(raw_text)
    return _assemble(face_label, face_prob,
                     speech_label, speech_prob,
                     text_label, text_prob,
                     raw_text)

# ===== helpers ====================================================
def _label_and_prob_from_text(txt: str):
    preds = text.text_detection(txt)
    if isinstance(preds, list) and isinstance(preds[0], list):
        preds = preds[0]                                   # [[{…}]] → [{…}]
    best = max(preds, key=lambda d: d["score"])
    return best["label"], best["score"]

def _assemble(
    face_lbl, face_p,
    speech_lbl, speech_p,
    text_lbl, text_p,
    transcript,
):
    return {
        "face":   {"label": face_lbl,   "score": face_p},
        "speech": {"label": speech_lbl, "score": speech_p},
        "text":   {"label": text_lbl,   "score": text_p},
        "transcript": transcript,
    }
