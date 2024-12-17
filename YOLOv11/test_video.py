import cv2
from ultralytics import YOLO
import time

# 创建YOLO模型的实例
model = YOLO(model="12_15_best.pt")

# 指定视频文件路径
video_file_path = ""  # 文件路径

# 打开视频文件
cap = cv2.VideoCapture(video_file_path)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        result = model.predict(frame)
        result_plot = result[0].plot()
        cv2.imshow("best1", result_plot)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break  # 如果没有读取到帧，则跳出循环

    time.sleep(1)
cap.release()
cv2.destroyAllWindows()
