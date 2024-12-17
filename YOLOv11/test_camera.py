import cv2
from ultralytics import YOLO
import time
model = YOLO(model="12_15_best.pt")
# 摄像头编号
camera_nu = 0
# 打开摄像头
cap = cv2.VideoCapture(camera_nu)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        result = model.predict(frame)
        result_plot = result[0].plot()
        cv2.imshow("best1", result_plot)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()