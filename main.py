# new_main.py  – pick input type + modalities
import tempfile, io
from pathlib import Path
import streamlit as st
import backend

# ───── Sidebar controls ───────────────────────────────────────────
st.sidebar.header("⚙️  Choose input & modalities")



input_type = st.sidebar.radio(
    "Input type", ["Video", "Audio", "Raw text"], index=0
)

modalities = st.sidebar.multiselect(
    "Run which modalities?",
    ["Face", "Speech", "Text"],
    default=["Face", "Speech", "Text"]
)

run_face   = "Face"   in modalities and input_type == "Video"
run_speech = "Speech" in modalities and input_type != "Raw text"
run_text   = "Text"   in modalities

st.title("🎬 Flexible Emotion Detector")

st.markdown(
    """
**How these numbers are produced**

* **Face** – multiple faces can appear, but we look at every frame and report
  the *emotion of the face that appears most often* across the whole video.

* **Speech** – emotion is predicted **directly from the sound-wave**  
  (MFCC features → neural-network). No text is used here.

* **Text** – emotion is predicted **from text only**.  
  For video / audio inputs we first transcribe the speech, then feed the
  transcript into a language-model classifier.
""",
    unsafe_allow_html=False,
)
# ───── Input widgets based on choice ──────────────────────────────
uploaded_file = None
raw_text = ""

if input_type == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=("mp4", "mov", "avi"))
elif input_type == "Audio":
    uploaded_file = st.file_uploader("Upload an audio file", type=("wav", "ogg", "mp3"))
else:  # Raw text
    raw_text = st.text_area("Paste text here", height=150)

# ───── Run button & processing ────────────────────────────────────
def pct(x): return f"{x*100:5.1f} %" if x else "—"

if (uploaded_file or raw_text) and st.button("🔍 Analyse"):
    with st.spinner("Running models…"):
        if input_type == "Video":
            # save temp video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.getbuffer())
                path = Path(tmp.name)
            res = backend.detect_from_video(
                path, do_face=run_face, do_speech=run_speech, do_text=run_text)
            path.unlink(missing_ok=True)

        elif input_type == "Audio":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.getbuffer())
                path = Path(tmp.name)
            res = backend.detect_from_audio(
                path, do_speech=run_speech, do_text=run_text)
            path.unlink(missing_ok=True)

        else:  # Raw text
            res = backend.detect_from_text(raw_text)

    # ───── Show results ───────────────────────────────────────────
    cols = st.columns(3)
    if run_face:
        cols[0].metric("Face"  , res["face"]["label"]  , pct(res["face"]["score"]))
    if run_speech:
        cols[1].metric("Speech", res["speech"]["label"], pct(res["speech"]["score"]))
    if run_text:
        cols[2].metric("Text"  , res["text"]["label"]  , pct(res["text"]["score"]))

    if run_text and res["transcript"]:
        with st.expander("📝 Transcript"):
            st.write(res["transcript"])
    # NEW ↓↓↓ ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Uploaded media")

    if input_type == "Video":
        st.video(uploaded_file)           # replay the video
    elif input_type == "Audio":
        st.audio(uploaded_file)           # audio player
    else:  # Raw text
        st.code(raw_text or "_No text provided_")