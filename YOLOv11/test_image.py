import cv2
from ultralytics import YOLO

# 创建YOLO模型的实例
model = YOLO(model="../12_15_best.pt")

# 指定图片文件路径
image_file_path = "D:/CCPD2019/CCPD2019/ccpd_base/02875-92_88-272&445_557&552-566&577_281&563_291&450_576&464-0_13_33_26_24_29_31-129-40.jpg"  # 请替换为你的实际文件路径

# 读取图片文件
frame = cv2.imread(image_file_path)

if frame is not None:
    result = model.predict(frame)
    print(result[0])
    result_plot = result[0].plot()

    # 显示结果
    cv2.imshow("Result", result_plot)
    k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
    if k == 27:  # 键盘上Esc键的键值
        cv2.destroyAllWindows()

else:
    print("Error: Could not read the image file.")

cv2.destroyAllWindows()
