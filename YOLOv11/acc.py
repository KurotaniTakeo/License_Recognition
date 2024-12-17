import cv2
import os
from ultralytics import YOLO


# 定义一个函数用于将归一化坐标转换为绝对坐标
def denormalize_bbox(width, height, bbox):
    x_center, y_center, w, h = bbox
    x_center_abs = x_center * width
    y_center_abs = y_center * height
    w_abs = w * width
    h_abs = h * height
    return [x_center_abs, y_center_abs, w_abs, h_abs]


# 定义一个函数用于计算精度
def calculate_accuracy(predictions, ground_truths, threshold):
    true_count = 0
    false_count = 0

    for pred, gt in zip(predictions, ground_truths):
        if pred is None or gt is None:
            # 未检测到坐标或无对应真值文件
            false_count += 1
        else:
            # 比较预测坐标和真实坐标
            matches = [
                abs(p - g) <= threshold
                for p, g in zip(pred, gt)
            ]
            if all(matches):
                true_count += 1
            else:
                false_count += 1

    accuracy = true_count / (true_count + false_count)
    return accuracy


# 设置路径、模型和阈值
image_dir = "D:/ccpd_new/train/images"
label_dir = "D:/ccpd_new/train/labels"  # 用于存放标注文件的目录
model = YOLO(model="12_15_best.pt")
threshold = 10  # 用于判断预测是否在正确范围内

# 初始化统计列表
predictions_list = []
ground_truths_list = []

# 遍历目录中的所有图片文件
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

        # 读取图片
        frame = cv2.imread(image_path)

        # 确保图片被正确读取
        if frame is not None:
            img_height, img_width = frame.shape[:2]

            # 获取模型预测的坐标
            result = model.predict(frame)
            if len(result[0].boxes) > 0:
                # 假设只考虑第一个检测对象
                pred_coords_xyxy = result[0].boxes.xyxy[0].tolist()
                # 转换为 [x_center, y_center, width, height]
                x1, y1, x2, y2 = pred_coords_xyxy
                pred_coords = [
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                    x2 - x1,
                    y2 - y1
                ]
            else:
                pred_coords = None
        else:
            print(f"Error: Could not read image {filename}")
            pred_coords = None

        # 读取并解析对应的标注文件
        try:
            with open(label_path, 'r') as f:
                line = f.readline().strip().split()
                gt_normalized = list(map(float, line[1:]))  # 忽略类别
                gt_coords = denormalize_bbox(img_width, img_height, gt_normalized)
        except FileNotFoundError:
            print(f"Warning: No corresponding label file for {filename}")
            gt_coords = None

        # 添加结果到列表中
        predictions_list.append(pred_coords)
        ground_truths_list.append(gt_coords)

# 计算准确率
accuracy = calculate_accuracy(predictions_list, ground_truths_list, threshold)
print(f"Model Accuracy: {accuracy:.2%}")
