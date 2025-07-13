import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from PIL import Image
import csv
import os

def ed(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def px_to_cm(px, ppi):
    return round((px / ppi) * 2.54, 2)

def process_image(image_path):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.5) as pose:

        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_ear = landmarks[7]
            right_ear = landmarks[8]
            shift = 3.5 * (right_ear.x - left_ear.x) / 4
            left_ear.y -= shift
            right_ear.y -= shift

            left_m = landmarks[9]
            right_m = landmarks[10]
            shift_sh = 3.4 * (right_m.x - left_m.x) / 4
            landmarks[11].x -= shift_sh
            landmarks[12].x += shift_sh

            face_indices_to_remove = [0, 1, 2, 3, 4, 5, 6, 9, 10]
            for idx in face_indices_to_remove:
                landmarks[idx].visibility = 0.0
                landmarks[idx].presence = 0.0
                landmarks[idx].x = 0.0
                landmarks[idx].y = 0.0
                landmarks[idx].z = 0.0

            landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=landmarks)
            results.pose_landmarks = landmark_list

            img_h, img_w = image.shape[:2]
            def to_pixel(lm): return int(lm.x * img_w), int(lm.y * img_h)

            lx, ly = to_pixel(landmarks[11])
            rx, ry = to_pixel(landmarks[12])
            lhx, lhy = to_pixel(landmarks[23])
            rhx, rhy = to_pixel(landmarks[24])
            lex, ley = to_pixel(landmarks[13])
            rex, rey = to_pixel(landmarks[14])
            lwx, lwy = to_pixel(landmarks[15])
            rwx, rwy = to_pixel(landmarks[16])
            leax, leay = to_pixel(landmarks[7])
            reax, reay = to_pixel(landmarks[8])

            shoulder_length = (ed(lx, ly, leax, leay) + ed(rx, ry, reax, reay)) / 2
            hand_length = (
                ed(lx, ly, lex, ley) + ed(lex, ley, lwx, lwy) +
                ed(rx, ry, rex, rey) + ed(rex, rey, rwx, rwy)
            ) / 2
            chest_length = ed(lx, ly, rx, ry)
            shirt_length = (ed(lx, ly, lhx, lhy) + ed(rx, ry, rhx, rhy)) / 2

            img_pil = Image.open(image_path)
            dpi = img_pil.info.get('dpi')
            ppi = dpi[0] if dpi else 96  # fallback

            shoulder_cm = px_to_cm(shoulder_length, ppi) * 4
            hand_cm = px_to_cm(hand_length, ppi) * 6
            chest_cm = px_to_cm(chest_length, ppi) * 5
            shirt_cm = px_to_cm(shirt_length, ppi) * 6

            output_file = "body_measurements.csv"
            with open(output_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Measurement", "Pixel Length", "Actual Size (cm) x 3"])
                writer.writerow(["Shoulder Length", round(shoulder_length, 2), shoulder_cm])
                writer.writerow(["Hand Length", round(hand_length, 2), hand_cm])
                writer.writerow(["Chest Length", round(chest_length, 2), chest_cm])
                writer.writerow(["Shirt Length", round(shirt_length, 2), shirt_cm])

            return output_file
