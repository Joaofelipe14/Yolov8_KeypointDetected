import cv2
import mediapipe as mp
import os
from pathlib import Path
import random

# Inicialize o modelo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Abra o vdeo usando OpenCV
video_path = 'C:/python/TratamentoFisioVisaoComp.mp4'
cap = cv2.VideoCapture(video_path)



# Defina o diretrio raiz para salvar keypoints e imagens
data_dir = Path('datasets')
os.makedirs(data_dir, exist_ok=True)

# Defina os diretrios "images" e "labels" dentro de "train", "test" e "valid"
train_dir = data_dir / 'train'
test_dir = data_dir / 'test'
valid_dir = data_dir / 'val'

dirs = [train_dir, test_dir, valid_dir]

for d in dirs:
    os.makedirs(d / 'images', exist_ok=True)
    os.makedirs(d / 'labels', exist_ok=True)

image_id = 0
frame_count = 0

while cap is not None and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 2 == 0:  # Processar a cada 5 frames
        # Execute a deteco de keypoints da mo
        results = hands.process(frame)

        # Verifique se h mos detectadas
        if results.multi_hand_landmarks:

            for landmarks in results.multi_hand_landmarks:
                image_info = {
                    "id": image_id,
                    "file_name": f"frame{image_id}.png",
                    "width": frame.shape[1],
                    "height": frame.shape[0]
                }

           

                # Coordenadas do centro da caixa delimitadora e dimenses da caixa delimitadora
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = -float('inf'), -float('inf')
                for landmark in landmarks.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    x_pixel, y_pixel = int(x * frame.shape[1]), int(y * frame.shape[0])
                    if x_pixel < x_min:
                        x_min = x_pixel
                    if x_pixel > x_max:
                        x_max = x_pixel
                    if y_pixel < y_min:
                        y_min = y_pixel
                    if y_pixel > y_max:
                        y_max = y_pixel

                # Aumentar as coordenadas da bounding box em 20%
                width = int((x_max - x_min) * 1.2)
                height = int((y_max - y_min) * 1.2)
                x_min = max(0, x_min - int(0.1 * width))
                x_max = min(frame.shape[1], x_max + int(0.1 * width))
                y_min = max(0, y_min - int(0.1 * height))
                y_max = min(frame.shape[0], y_max + int(0.1 * height))

                rand = random.random()
                if rand < 0.7:
                    label_dir = train_dir / 'labels'
                    image_dir = train_dir / 'images'
                elif rand < 0.8:
                    label_dir = test_dir / 'labels'
                    image_dir = test_dir / 'images'
                else:
                    label_dir = valid_dir / 'labels'
                    image_dir = valid_dir / 'images'

                image_path = image_dir / f'frame{image_id}.png'
                cv2.imwrite(str(image_path), frame)
                # Criar os diretrios se ainda no existirem
                os.makedirs(image_dir, exist_ok=True)
                label_path = label_dir / f'frame{image_id}.txt'
                with open(label_path, 'w') as label_file:
                    label_file.write("0 ")  # Class ID
                    x_center = (x_min + x_max) / (2 * frame.shape[1])
                    y_center = (y_min + y_max) / (2 * frame.shape[0])
                    width_normalized = width / frame.shape[1]
                    height_normalized = height / frame.shape[0]
                    label_file.write(f"{x_center:.6f} {y_center:.6f} {width_normalized:.6f} {height_normalized:.6f} ")

                    for landmark in landmarks.landmark:
                        x_normalized = landmark.x
                        y_normalized = landmark.y
                        label_file.write(f"{x_normalized:.6f} {y_normalized:.6f} 2.000000 ")


                # Exibir o frame com keypoints e bounding box
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                cv2.imshow('Hand Keypoints and Bounding Box', frame)
                cv2.waitKey(1)  # Pequeno atraso para mostrar o frame

                image_id += 1

    frame_count += 1

# Salvar os dados no formato YOLO
data_dir = str(data_dir)
print(f'Saved YOLO data to {data_dir}')
