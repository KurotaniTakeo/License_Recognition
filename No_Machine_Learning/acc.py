import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged

def find_license_plate_contour(edged):
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx
    return None

# 提取并裁剪车牌区域
def extract_license_plate(image, contour):
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(image, mask)
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = masked_image[y:y + h, x:x + w]
    return cropped_image

def process_images(directory, num_images):
    files = os.listdir(directory)
    selected_files = random.sample(files, min(num_images, len(files)))
    success_count = 0

    for file in selected_files:
        image_path = os.path.join(directory, file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        # 预处理图像
        preprocessed_image = preprocess_image(image)

        # 查找车牌轮廓，并提取车牌
        license_plate_contour = find_license_plate_contour(preprocessed_image)

        if license_plate_contour is not None:
            license_plate_image = extract_license_plate(image, license_plate_contour)

            # 将车牌转换为灰度图后进行二值化
            img_gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
            _, img_thre = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

            # 如果需要保存或进一步处理，可以在这里继续

            success_count += 1
        else:
            print(f"No license plate detected in image: {image_path}")

    total_images = len(selected_files)
    success_rate = (success_count / total_images) * 100 if total_images > 0 else 0

    print(f"Processed {total_images} images with a success rate of {success_rate:.2f}%.")

if __name__ == "__main__":
    directory_path = "D:/CCPD2019/CCPD2019/ccpd_base/"
    num_images_to_process = 1000  # 指定要处理的图像数量
    process_images(directory_path, num_images_to_process)
