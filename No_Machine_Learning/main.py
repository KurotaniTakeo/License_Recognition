import cv2
import numpy as np
import matplotlib.pyplot as plt


# 显示图像的辅助函数
def show_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


# 预处理图像：灰度转换，高斯模糊，Canny边缘检测
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged


# 找到潜在车牌轮廓
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


# 分割字符
def segment_characters(img_thre):
    height, width = img_thre.shape
    white = []
    black = []
    white_max = 0
    black_max = 0

    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0
        t = 0
        for j in range(height):
            if img_thre[j][i] == 255:
                s += 1
            if img_thre[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)

    arg = False
    if black_max > white_max:
        arg = True

    def find_end(start_):
        end_ = start_ + 1
        for m in range(start_ + 1, width - 1):
            if (black[m] if arg else white[m]) > (0.95 * black_max if arg else 0.95 * white_max):
                end_ = m
                break
        return end_

    n = 1
    start = 1
    end = 2
    while n < width - 2:
        n += 1
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
            start = n
            end = find_end(start)
            n = end
            if end - start > 5:
                cj = img_thre[1:height, start:end]
                cv2.imshow('Segmented Character', cj)
                cv2.waitKey(0)


if __name__ == "__main__":
    # 读取图片
    image_path = ("D:/CCPD2019/CCPD2019/ccpd_base/03-98_78-227&406_477&562-490&577_223&505_207&386_474&458-0_0_6_6_29_25_26-157-48.jpg")
    image = cv2.imread(image_path)
    show_image(image, "Original Image")

    # 预处理图像
    preprocessed_image = preprocess_image(image)
    show_image(preprocessed_image, "Edged Image")

    # 查找车牌轮廓，并提取车牌
    license_plate_contour = find_license_plate_contour(preprocessed_image)

    if license_plate_contour is not None:
        license_plate_image = extract_license_plate(image, license_plate_contour)
        show_image(license_plate_image, "License Plate")

        # 将车牌转换为灰度图后进行二值化
        img_gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
        _, img_thre = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

        # 保存二值化结果并显示
        show_image(img_thre, "Thresholded License Plate")

    else:
        print("No license plate detected.")
