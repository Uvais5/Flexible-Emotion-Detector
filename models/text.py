from transformers import pipeline
def text_detection(text):
    try:
        # emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
        emo = pipeline("text-classification",model="j-hartmann/emotion-english-distilroberta-base",top_k=None)
        emotion_labels = emo(text)[0]
    except:
        print("wrong with text model")
        emotion_labels = "somthing wrong please try again"
    return emotion_labels