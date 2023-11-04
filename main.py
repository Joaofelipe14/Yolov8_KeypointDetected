from ultralytics import YOLO
import cv2
# Descomente para fazer o treinamento

model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
results = model.train(data='data1.yaml', epochs=1, imgsz=640)





# Descomente para fazer a previsao baseado no treinamento 
#model = YOLO('C:/python/runs/pose/train21/weights/best.pt')  # load a pretrained model (recommended for training)
#model.predict('C:/python/TratamentoFisioVisaoComp.mp4',show=True,imgsz=320,conf=0.9)
#model.predict('C:/python/datasets/val/images/frame4.png',save=True,imgsz=320,conf=0.9)

