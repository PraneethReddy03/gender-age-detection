cat << 'EOF' > src/infer.py
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    instr_pos = (10, 20)
    instr_color = (0, 255, 0)
    instr_thick = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        cv2.putText(frame,
                    "Press 'q' or Esc to quit",
                    instr_pos,
                    font,
                    0.6,
                    instr_color,
                    instr_thick)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (64, 64))
            inp = np.expand_dims(face_resized / 255.0, 0)
            
            # Run inference
            gender_pred, age_pred = model.predict(inp)
            prob_female = float(gender_pred[0][0])
            gender = "F" if prob_female > 0.5 else "M"

            # Age bracket
            age_idx = np.argmax(age_pred, axis=-1)[0]
            age_label = age_ranges.get(age_idx, "Unknown")

            # Draw box + text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{gender} ({prob_female:.2f}), {age_label}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )
            # Draw confidence bar at bottom
            h_frame, w_frame = frame.shape[:2]
            bar_base = (10, h_frame - 10)
            bar_length = int((w_frame - 20) * prob_female)
            bar_height = 10
            cv2.rectangle(frame,
                          bar_base,
                          (10 + bar_length, bar_base[1] - bar_height),
                          (255, 0, 255), -1)
            cv2.rectangle(frame,
                          (10, bar_base[1]),
                          (w_frame - 10, bar_base[1] - bar_height),
                          (255, 0, 255), 1)
            cv2.putText(frame,
                        f"Female conf: {prob_female:.2f}",
                        (10, bar_base[1] - bar_height - 5),
                        font,
                        0.5,
                        (255, 0, 255),
                        1)
        cv2.imshow("Gender & Age", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if cv2.getWindowProperty("Gender & Age", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to your trained model file (either .h5 or .keras)")
    args = p.parse_args()
    infer(args.model)
EOF
