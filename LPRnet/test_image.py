import argparse
import random
import cv2
import os
import numpy as np

from imutils import paths
from torch.utils.data import Dataset
from ultralytics import YOLO
from LPRNet import *

from PIL import Image, ImageDraw, ImageFont

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        # Image = cv2.imread(filename)
        Image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            # one_hot_base = np.zeros(len(CHARS))
            # one_hot_base[CHARS_DICT[c]] = 1
            label.append(CHARS_DICT[c])

        if len(label) == 8:
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True

def get_args():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_dirs', default=r"D:\CCPD2019\lpr", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=100, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model',
                        default=r'C:\Users\1\PycharmProjects\YOLOv5-LPRNet-Licence-Recognition\weights\runslprnet-pretrain.pth')
    # parser.add_argument('--pretrained_model', default=r'E:\code\pycode\locaklicnese\local_yolo_lpr_licnese\best.pth', help='pretrained base model')
    return parser.parse_args()

# def recognize_license_plate(image_path, pretrained_model_path):
#     img_size = [94, 24]
#     lpr_max_len = 8
#     dropout_rate = 0
#     use_cuda = torch.cuda.is_available()
#
#     # Initialize LPRNet
#     lprnet = LPRNet(lpr_max_len=lpr_max_len, phase=False, class_num=len(CHARS), dropout_rate=dropout_rate)
#     device = torch.device("cuda:0" if use_cuda else "cpu")
#     lprnet.to(device)
#
#     # Load pretrained model
#     if pretrained_model_path:
#         lprnet.load_state_dict(torch.load(pretrained_model_path))
#     else:
#         raise FileNotFoundError("Pretrained model not found.")
#
#     # Preprocess the image
#     image = extract_and_resize_license_plate(image_path, output_size=(94, 24))
#     image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
#
#     if use_cuda:
#         image = image.to(device)
#
#     # Forward pass
#     with torch.no_grad():
#         prebs = lprnet(image)
#
#     # Decode the prediction
#     prebs = prebs.cpu().numpy()
#     preb = prebs[0, :, :]  # Assuming single image
#     preb_label = [np.argmax(preb[:, j], axis=0) for j in range(preb.shape[1])]
#
#     # Remove repeated characters and blanks
#     no_repeat_blank_label = []
#     pre_c = preb_label[0]
#     if pre_c != len(CHARS) - 1:
#         no_repeat_blank_label.append(pre_c)
#     for c in preb_label:
#         if (pre_c == c) or (c == len(CHARS) - 1):
#             if c == len(CHARS) - 1:
#                 pre_c = c
#             continue
#         no_repeat_blank_label.append(c)
#         pre_c = c
#
#     # Convert to string
#     license_plate = ''.join([CHARS[i] for i in no_repeat_blank_label])
#
#     return license_plate

# 裁切yolo
def extract_and_resize_license_plate(image_path, model_path="../weights/12_15_best.pt", output_size=(94, 24)):
    # 创建YOLO模型的实例
    model = YOLO(model=model_path)

    # 读取图片文件
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not read the image file.")
        return None

    # 使用模型进行预测
    results = model.predict(frame)

    if len(results) > 0 and len(results[0].boxes) > 0:
        # 获取第一个检测框的坐标
        box = results[0].boxes[0]

        # 确保获取到的 box 是一个一维张量并且有 4 个元素
        xyxy = box.xyxy.detach().cpu().numpy().flatten()
        if len(xyxy) != 4:
            print("Unexpected shape for box coordinates:", xyxy)
            return None

        x1, y1, x2, y2 = map(int, xyxy)

        # 裁剪出车牌部分
        license_plate = frame[y1:y2, x1:x2]

        # 调整尺寸到 94x24
        resized_license_plate = cv2.resize(license_plate, output_size)

        return resized_license_plate, box
    else:
        print("No license plate detected.")
        return None

def decode_plate(resized_img, lpr_net, CHARS, use_cuda=False):
    # Ensure the image is a float tensor and normalize if necessary
    resized_img = resized_img.astype(np.float32) / 255.0  # Normalize to [0, 1] range
    image_tensor = torch.tensor(resized_img).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.permute(0, 3, 1, 2)  # Change shape to [batch_size, channels, height, width]

    if use_cuda:
        image_tensor = image_tensor.cuda()

    # Forward pass through the network
    with torch.no_grad():
        prebs = lpr_net(image_tensor)

    # Move the predictions to CPU and convert to NumPy
    prebs = prebs.cpu().numpy()

    # Greedy decoding
    preb_labels = []
    preb = prebs[0, :, :]  # Assuming input batch size is 1: shape [68, 18]
    preb_label = [np.argmax(preb[:, j]) for j in range(preb.shape[1])]

    no_repeat_blank_label = []
    pre_c = preb_label[0]
    if pre_c != len(CHARS) - 1:  # Exclude blank character (e.g., '-')
        no_repeat_blank_label.append(CHARS[pre_c])

    for c in preb_label:
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(CHARS[c])
        pre_c = c

    return ''.join(no_repeat_blank_label)

def draw_box_and_plate(image_path, box, license_plate_str, font_path='simsun.ttc'):
    # 读取图片文件
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image file.")
        return None

    # 确保获取到的 box 是一个一维张量并且有 4 个元素
    xyxy = box.xyxy.detach().cpu().numpy().flatten()
    if len(xyxy) != 4:
        print("Unexpected shape for box coordinates:", xyxy)
        return

    x1, y1, x2, y2 = map(int, xyxy)

    # 转换为 PIL 图像以便绘制中文字符
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # 加载字体
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)

    # 绘制矩形框
    draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)

    # 获取文本的边界框尺寸
    text_bbox = draw.textbbox((0, 0), license_plate_str, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 确定文本位置，确保不会超出图像边界
    text_x = x1
    text_y = y1 - text_height - 5 if y1 - text_height - 5 > 10 else y1 + 5

    # 绘制文本背景
    draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=(255, 0, 0))

    # 绘制文本
    draw.text((text_x, text_y), license_plate_str, font=font, fill=(255, 255, 255))

    # 将结果转换回 OpenCV 格式
    result_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # 显示图像
    cv2.imshow('License Plate Detection', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


args = get_args()
# 加载模型
lprnet = LPRNet(lpr_max_len=args.lpr_max_len, phase=False, class_num=len(CHARS), dropout_rate=args.dropout_rate)
device = torch.device("cuda:0" if args.cuda else "cpu")
lprnet.to(device)
lprnet.load_state_dict(torch.load(r'../weights/runslprnet-pretrain.pth'))

image_file_path = "D:/CCPD2019/CCPD2019/ccpd_base/02625-88_87-278&467_556&558-552&547_281&563_291&467_562&451-0_0_22_27_30_24_27-144-91.jpg"  # 请替换为你的实际文件路径

resized_img, box = extract_and_resize_license_plate(image_file_path, output_size=(94, 24))

license_plate = decode_plate(resized_img, lprnet, CHARS, use_cuda=args.cuda)
print(license_plate)

draw_box_and_plate(image_file_path, box, license_plate)


