# 2nd module

import torch
from PIL import Image

class GetData(Dataset):
    def __init__(self, imgDir, labelsDir, S=7, B=2, C=45):
        self.labelsDir = labelsDir
        self.imgDir = imgDir
        self.S = S 
        self.B = B
        self.C = C 
    
    def __getitem__(self, index):
        image = Image.open(self.imgDir[index]).convert("RGB") # load the index image
        
        # get a list of all annotations
        boxes = [] 
        with open(labelsDir[index]) as file:
            for line in file.readlines():
                splitNr = line.split() # imparte linia
                classNr = int(splitNr[4]) # clasa e a 5 variabila
                x, y, w, h = float(splitNr[0:4])
                label = [x, y, w, h, classNr]
                boxes.append(label)

        # resize to 448 x 448 and convert it to tensor for the model input
        image = image.resize((448, 448))
        image = image.ToTensor()

        # now i need to create the target tensor (S, S, C + 5)
        target = torch.zeros((self.S, self.S, self.C + 5))
        for box in boxes:
            x, y, w, h, classNr = box

            # get the responsible grid cell by computing the coordinates
            i = int(self.S * x)
            j = int(self.S * y)

            # compute the offset of the center of the object regarding that point (the position of the object is relative to the specific grid not to the whole image)
            xLeft = self.S - i 
            yLeft = self.S - j

            # construct the tensor
            target[i, j, 0:4] = torch.tensor([xLeft, yLeft, w, h])
            target[i, j, 4] = 1 # set the confidence
            target[i, j, 5 + classNr] = 1
        
        return image, target