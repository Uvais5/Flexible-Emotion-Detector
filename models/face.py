import cv2
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
import pandas as pd

from pathlib import Path


# -------------------------------------------------------------------
# One-time helpers (change these paths to suit your project layout)
MODEL_JSON = Path("models/saved_model/face_emotion_model_js.json")
MODEL_WEIGHTS = Path("models/saved_model/face_emotion_model_js.h5")
CASCADE_XML = Path("models/saved_model/haarcascades/haarcascade_frontalface_default.xml")

# Map index → readable label
EMOTION_DICT = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# -------------------------------------------------------------------
def _load_face_model():
    """Load model architecture + weights just once per call."""
    with open(MODEL_JSON, "r") as f:
        model = model_from_json(f.read())
    model.load_weights(MODEL_WEIGHTS)
    return model

def face_detection(filename):
    emo = []
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",3: "Happy",4: "Neutral", 5: "Sad",6: "Surprised"}
    #load json and create model
    json_file = open("C:/Machine Learning/Emotion_app/models/saved_model/face_emotion_model_js.json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #laod weights into new model
    model.load_weights("C:/Machine Learning/Emotion_app/models/saved_model/face_emotion_model_js.h5")
    print("loaded model from disk")
        #start webcam feed
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(filename)
    while True:
        rat, frame = cap.read()
        
        
        if not rat:
            break
        frame = cv2.resize(frame,(400,580))
        face_dectector = cv2.CascadeClassifier("C:/Machine Learning/face_emotion_detection/haarcascades/haarcascade_frontalface_default.xml")
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #datect the face available on camera
        num_face = face_dectector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        #taken each face available on the camera and preprocess it 
        for(x, y, w, h) in num_face:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10),(0,255,0),4)
            roi_gray_frame = gray_frame[y:y + h,x:x + w]
            croping_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)),-1),0)
            #predict  the emotion 
            emotion_prediction = model.predict(croping_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex],(x+5,y-20),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0 ),2,cv2.LINE_AA)
            
            emo.append(emotion_dict[maxindex])
         
        cv2.imshow('Emotion Dectection',frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame(emo,columns=["emo"])
    
    d = df.value_counts()

    dd = d.to_string()
    return dd[10:26].replace("\n","")
def face_detection_novideo(filename):
    """
    Analyse a **video file** and return (best_label, best_probability).

    * best_probability is the average soft-max score of the winning class
      across all detected faces (0-1 float).
    * If no faces are found, returns ("N/A", 0.0).
    """

    model = _load_face_model()
    face_detector = cv2.CascadeClassifier(str(CASCADE_XML))

    # Accumulate soft-max scores
    total_scores = np.zeros(len(EMOTION_DICT), dtype=np.float64)
    sample_count = 0

    cap = cv2.VideoCapture(str(filename))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float32") / 255.0          # normalise 0-1
            roi = np.expand_dims(np.expand_dims(roi, -1), 0)

            softmax = model.predict(roi, verbose=0)[0]   # shape (7,)
            total_scores += softmax
            sample_count += 1

    cap.release()

    if sample_count == 0:
        return "N/A", 0.0

    avg_scores = total_scores / sample_count
    best_idx = int(np.argmax(avg_scores))
    best_label = EMOTION_DICT[best_idx]
    best_prob = float(avg_scores[best_idx])             # already 0-1

    return best_label, best_prob
