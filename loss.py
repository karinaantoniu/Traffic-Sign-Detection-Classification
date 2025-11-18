# 3rd module 

import torch
from utils import IoU

class CostFunction(nn.Module):
    # implement 3 functions
    # 1 confidence error
    # 2 localization error (SSE)
    # 3 classification error (Categorical Cross Entropy Loss)

    # each vector has the following structrure [x, y, w, h, C, class1, .... classC], where:
    #  x = coordonata x a centrului relativa la celula
    #  y = coordonata y a centrului relativa la celula
    #  w = latimea w a boxului
    #  h = inaltimea h a boxului
    #  C = confidence score
    #  vectorul one hot encoding al clasei [0 .. 0 .. 1 ... 0]

    def __init__(self):
        super().__init__()

    def forward(self, predictions, target):

        # predictions are forma (N, 7, 7, 55), N = nr de imagini
        # structura ultimului vector de 55x1 e asa: 
        # [0:4] coordonatele primei cutii, 
        # [4] - scorul de confidenta
        # [5:9] - coordonatele cutiei 2,
        # [9] - confidenta scor 2
        # [10:55] - logits (per celula, nu per box) [-2.1, 0.5 ...]

        # target are forma (N, 7, 7, 50)
        # [0] - coordonata x bounding box-ului
        # [1] - coordonata y
        # [2] - latimea
        # [3] - inaltimea
        # [4] - masca (iObj) 1.0 daca exista obiect in celula, 0 altfel
        # [5:50] - one hot encoding 

        lossConf, b1RespTrue, b2RespTrue = self.confidenceError(predictions, target)
        lossLoc = self.localizationError(target, predictions, b1RespTrue, b2RespTrue)
        lossCross = self.categoricalCrossEntropyLoss(target, predictions)

        totalLoss = lossLoc + lossConf + lossCross
        return totalLoss / predictions.shape[0]
    

    # the probability that an object exists in that box and how well does the box matches the real object
    def confidenceError(self, predictions, target):
        # setting the weights
        lambdaNoobj = 0.5
        lambdaObj = 1.0

        box1Coords = predictions[..., 0:4] # pun ... sa ia si dimensiunile de dinainte
        box1Conf = predictions[..., 4:5]

        box2Coords = predictions[..., 5:9]
        box2Conf = predictions[..., 9:10] # pun 9:10 ca sa imi adauge ultima dimensiune 1 sa pot aplica mai departe calculele

        targetBoxCoords = target[..., 0:4]

        iouScore1 = utils.IoU(box1Coords, targetBoxCoords)
        iouScore2 = utils.IoU(box2Coords, targetBoxCoords)

        # one of the masks is responsible for the object (1 / 0)
        b1Responsible = (iouScore1 >= iouScore2).float()
        b2Responsible = 1.0 - b1Responsible

        # another mask; (1 / 0) value if the box is responsible and the object is there
        b1RespTrue = target[..., 4:5] * b1Responsible
        b2RespTrue = target[..., 4:5] * b2Responsible

        # now each box is penalized differently 
        lossObj = torch.sum(b1RespTrue * (box1Conf - iouScore1) ** 2) + torch.sum(b2RespTrue * (box2Conf - iouScore2) ** 2)

        b1Noobj = 1.0 - b1RespTrue
        b2Noobj = 1.0 - b2RespTrue

        lossNoobj = torch.sum(b1Noobj * (box1Conf - 0.0) ** 2) + torch.sum(b2Noobj * (box2Conf - 0.0) ** 2)

        return (lambdaObj * lossObj + lambdaNoobj * lossNoobj), b1RespTrue, b2RespTrue

    # penalizes the model if the position and dimension are wrongly predicted
    # we are using the Error Sum of Squares
    def localizationError(self, target, predictions, b1RespTrue, b2RespTrue):
        # weight to penalize the wrong position or dimension of the box
        lambdaCoord = 5.0 

        # to measure the difference between the predicted box and the real box we measure on coordinates:
        # x difference for Box1
        firstBoxXY = (predictions[..., 0:2] - target[..., 0:2]) ** 2
        secondBoxXY = (predictions[..., 5:7] - target[..., 0:2]) ** 2
        L1 = torch.sum(b1RespTrue * firstBoxXY + b2RespTrue * secondBoxXY)

        firstBoxWH = (torch.sqrt(torch.abs(predictions[..., 2:4])) - torch.sqrt(torch.abs(target[..., 2:4]))) ** 2
        secondBoxWH = (torch.sqrt(torch.abs(predictions[..., 7:9])) - torch.sqrt(torch.abs(target[..., 2:4]))) ** 2
        L2 = torch.sum(b1RespTrue * firstBoxWH + b2RespTrue * secondBoxWH)

        return lambdaCoord * (L1 + L2)

    # penalizes the model if it makes a wrong guess regarding the object class
    def categoricalCrossEntropyLoss(self, target, predictions):
        lambdaObj = 1.0
        isObj = target[..., 4].unsqueeze(-1) # adauga ultima dimensiune
        L = target[..., 5:50] * torch.log(torch.softmax(predictions[..., 10:55], dim=-1)) # softmax makes sure the numbers are probabilities
        L = torch.sum(L * isObj)

        return (-1) * L * lambdaObj

        # s-ar putea sa am nevoie de un epsilon foarte mic 1e6 sa il adun la log si la sqrt (o sa imi dea log din 0 nan si termina programul)
