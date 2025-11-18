# 5th module

def IoU(currentBox, targetBox):

    # ambele boxuri sunt de forma (x_center, y_center, w, h), cu shape-ul (N, S, S, 4)

    x1, y1, x2, y2 = currentBox 
    x3, y3, x4, y4 = targetBox
    x1 = currentBox[..., 0:1] - currentBox[..., 2:3] / 2 #(de observat cum arata tensorul initial)

    x_inter1 = torch.max(x1, x3)
    y_inter1 = torch.max(y1, y3)

    x_inter2 = torch.min(x2, x4)
    y_inter2 = torch.min(y2, y4)

    width_inter = x_inter2 - x_inter1
    height_inter = y_inter2 - y_inter1
    area_inter = torch.max(width_inter, 0) * torch.max(height_inter, 0)

    width_box1 = (x2 - x1)
    height_box1 = (y2 - y1)
    width_box2 = (x4 - x3)
    height_box2 = (y4 - y3)

    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2

    area_union = area_box1 + area_box2 - area_inter

    IoU = area_inter / area_union

    return IoU

def nonMaxSuppression(predictions):
    # predictions are forma (N, 7, 7, 55), N = nr de imagini
    # structura ultimului vector de 55x1 e asa: 
    # [0:4] coordonatele primei cutii, 
    # [4] - scorul de confidenta
    # [5:9] - coordonatele cutiei 2,
    # [9] - confidenta scor 2
    # [10:55] - logits (per celula, nu per box) [-2.1, 0.5 ...]

    coordBox1 = predictions[..., 0:4]
    coordBox2 = predictions[..., 5:9]

    scoreBox1 = torch.sigmoid(predictions[..., 4:5]) # place it in interval [0, 1]
    scoreBox2 = torch.sigmoid(predictions[..., 9:10])
    
    classIdx = torch.argmax(predictions[..., 10:55], dim=-1).unsqueeze(-1).float()

    box1 = torch.cat([coordBox1, scoreBox1, classIdx], dim=-1)
    box2 = torch.cat([coordBox2, scoreBox2, classIdx], dim=-1)

    box1 = box1.reshape(-1, 6)
    box2 = box2.reshape(-1, 6)

    data = torch.cat([box1, box2], dim=0)

    scores = data[:, 4]
    sortedIdx = torch.argsort(scores, descending=True)
    data = data[sortedIdx]

    thresh = 0.5 # threshold for both confidence score and iou score

    # removing any box with the probability below 50%
    data = data[data[:, 4] > thresh]

    finalBoxes = []
    uniqueClasses = torch.unique(data[:, 5])
    for c in uniqueClasses:
        classMask = data[:, 5] == c
        classBoxes = data[classMask]

        #NMS algorithm
        while classBoxes.size(dim=0) > 0:
            best = classBoxes[0]
            finalBoxes.append(best)

            ious = IoU(best[..., :4], classBoxes[..., :4])
            underThresh = ious < thresh
            classBoxes = classBoxes[underThresh]

    return finalBoxes 
