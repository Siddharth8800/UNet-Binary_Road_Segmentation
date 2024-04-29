import torch.nn as nn
from parts import UpSample, DoubleConv, DownSample

class Unet(nn.Module):
    def __init__(self, in_channles, num_classes):
        super(Unet, self).__init__()
        self.down1 = DownSample(in_channles, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        conv_ans1, ans1 = self.down1(x)
        conv_ans2, ans2 = self.down2(ans1)
        conv_ans3, ans3 = self.down3(ans2)
        conv_ans4, ans4 = self.down4(ans3)

        bottle = self.bottle_neck(ans4)

        up_ans1 = self.up1(bottle, conv_ans4)
        up_ans2 = self.up2(up_ans1, conv_ans3)
        up_ans3 = self.up3(up_ans2, conv_ans2)
        up_ans4 = self.up4(up_ans3, conv_ans1)
        ans = self.out(up_ans4)
        return ans