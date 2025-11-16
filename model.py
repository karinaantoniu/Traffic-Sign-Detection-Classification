# 1st module 

import torch

# When creating a new neural network, you would usually go about creating a new class and inheriting from nn.Module, and defining two methods: __init__ (the initializer, where you define your layers) and forward (the inference code of your module, where you use your layers). That's all you need, since PyTorch will handle backward pass with Autograd. 

class YoloV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.S = 7 # size i will resize the image
        self.C = 45 # number of classes i want to detect (nr of != trafic signs)
        self.B = 2 # number of bounding boxes each cell from the grill will try to predict (one for high boxes and one for wide boxes)
        self.outputDim = self.B * 5 + self.C # the length of the prediction vector for one grid cell

        # for each box the model has to predict 5 things: x (center coord), y, w (width), h (height), confidence (the confidence that this box contains an obj)
        # add to it the type of object it is

        #the filters needed for the model
        # out_channels are the number of filters
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding='same'),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding='same'),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding='same'),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), padding='same'),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), padding='same'), 
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding='same'),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding='same', stride=(2, 2)),
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding='same'),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding='same')
        )

        self.block7 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakuReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * self.outputDim),
            nn.Sigmoid()   
        )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = x.reshape(-1, self.S, self.S, self.outputDim)

            return x