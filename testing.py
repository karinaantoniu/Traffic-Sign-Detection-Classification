import torch
from model import YoloV1
from dataset import GetData
from utils import get_bboxes, mean_average_precision
import config
import os
from torch.utils.data import DataLoader

def main():
    model = YoloV1().to(config.DEVICE)
    weights_path = "weights/yolov1_epoch100.pth" 
    model.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))

    files = os.listdir(config.TEST_IMG_DIR)
    img_files = []
    label_files = []
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(config.TEST_IMG_DIR, file)
            label_name = file.rsplit('.', 1)[0] + ".txt"
            label_path = os.path.join(config.TEST_LABEL_DIR, label_name)
            if os.path.exists(label_path):
                img_files.append(img_path)
                label_files.append(label_path)

    test_dataset = GetData(imgDir=img_files, labelsDir=label_files, S=config.S, B=config.B, C=config.C)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Getting bounding boxes from test set...")
    pred_boxes, true_boxes = get_bboxes(test_loader, model, iou_threshold=config.iouThresh, conf_threshold=config.nmsThresh, device=config.DEVICE)
    
    corrected_pred_boxes = []
    for box in pred_boxes:
        # [0:Idx, 1:X, 2:Y, 3:W, 4:H, 5:Conf, 6:Class]
        new_box = [
            box[0],      # Idx
            box[6],      # Class (mutat de la sfarsit)
            box[5],      # Conf (mutat)
            box[1],      # X
            box[2],      # Y
            box[3],      # W
            box[4]       # H
        ]
        corrected_pred_boxes.append(new_box)

    mAP = mean_average_precision(corrected_pred_boxes, true_boxes, iou_threshold=config.iouThresh)
    print(f"Mean Average Precision (mAP): {mAP * 100:.2f}%")

if __name__ == "__main__":
    main()