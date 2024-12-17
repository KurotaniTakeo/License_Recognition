from ultralytics import YOLO
model = YOLO("/root/yolov11/train11_29/runs/detect/train14/weights/best.pt")
results = model.train(data="wider_face.yaml", epochs=32, imgsz=640, batch=32, workers=12)