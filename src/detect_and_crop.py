import os
import cv2
import argparse

def detect_and_crop(input_dir, output_dir):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        path = os.path.join(input_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for i, (x, y, w, h) in enumerate(faces):
            crop = img[y:y+h, x:x+w]
            out_path = os.path.join(
                output_dir,
                f"{os.path.splitext(fname)[0]}_face{i}.jpg"
            )
            cv2.imwrite(out_path, crop)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Raw images folder")
    p.add_argument("--output", required=True, help="Cropped output folder")
    args = p.parse_args()
    detect_and_crop(args.input, args.output)
