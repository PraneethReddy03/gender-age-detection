import cv2
import numpy as np
import tensorflow as tf
import argparse

def infer(model_path):
    model = tf.keras.models.load_model(model_path)
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    age_ranges = {i: f"{i*10}-{i*10+9}" for i in range(12)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (64, 64))
            inp = np.expand_dims(face_resized / 255.0, 0)
            gender_pred, age_pred = model.predict(inp)
            gender = "M" if gender_pred[0][0] > 0.5 else "F"
            age = np.argmax(age_pred, axis=-1)[0]
            age_label = age_ranges.get(age, "Unknown")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{gender}, {age_label}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )
        cv2.imshow("Gender & Age", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to your trained model file (either .h5 or .keras)")
    args = p.parse_args()
    infer(args.model)
