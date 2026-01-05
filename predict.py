import torch
import cv2
import numpy as np
import config
from model import YoloV1
from utils import convert_cellboxes, nonMaxSuppression
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse

def predict_and_show(image_path, weights_path):
    model = YoloV1().to(config.DEVICE)
    
    checkpoint = torch.load(weights_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    # read image and prepare it for the model
    img_pil = Image.open(image_path).convert("RGB")
    img_resized = img_pil.resize((config.IMG_SIZE, config.IMG_SIZE))
    
    transform = transforms.ToTensor()
    img_tensor = transform(img_resized).unsqueeze(0).to(config.DEVICE)

    # prediction
    with torch.no_grad():
        predictions = model(img_tensor)

    # decode output
    # convert_cellboxes returneaza coordonate (x, y, w, h) relative la imagine
    bboxes = convert_cellboxes(predictions, S=config.S)
    
    # first image from batch and we flatten it
    bboxes = bboxes[0].reshape(-1, 6) 
    
    # convert to simple list for NMS function
    bboxes_list = bboxes.tolist()
    nms_boxes = nonMaxSuppression(bboxes_list, iou_threshold=0.5, threshold=0.2)

    # openCV draws the image
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print("Error: OpenCV could not read image")
        return
        
    h_orig, w_orig, _ = image_cv.shape

    if not nms_boxes:
        print("No object detected with confidence over 0.2")
    
    for box in nms_boxes:
        # box: [x, y, w, h, conf, class] values in [0, 1]
        class_idx = int(box[5])
        conf = box[4]
        
        bx, by, bw, bh = box[0], box[1], box[2], box[3]

        # convert to pixels
        x1 = int((bx - bw / 2) * w_orig)
        y1 = int((by - bh / 2) * h_orig)
        x2 = int((bx + bw / 2) * w_orig)
        y2 = int((by + bh / 2) * h_orig)

        # draw bounding box over object
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"Class {class_idx}: {conf:.2f}"
        cv2.putText(image_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"Detected: {text} la [{x1}, {y1}, {x2}, {y2}]")

    cv2.imshow("Result YOLO", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/test/images/test_image.jpg", help="test image path")
    parser.add_argument("--weights", type=str, default="weights/yolov1_epoch100.pth", help="weights path")
    args = parser.parse_args()

    predict_and_show(args.image, args.weights)