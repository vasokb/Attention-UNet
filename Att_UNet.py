from torch import nn
import torch


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        #gating signal
        g1 = self.W_g(g)
        # l
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 1，Sigmoid,
        psi = self.psi(psi)
        #  x
        return x * psi


class Att_UNet(nn.Module):
    def __init__(self,  n_inputs, par):
        super().__init__()

        self.feat = par.feature_maps
        self.n_inputs = n_inputs

        self.Maxpool = nn.MaxPool2d(2)

        self.Conv1 = conv_block(self.n_inputs, self.feat)
        self.Conv2 = conv_block(self.feat, self.feat * 2)
        self.Conv3 = conv_block(self.feat * 2, self.feat * 4)
        self.Conv4 = conv_block(self.feat * 4, self.feat * 8)

        self.Up4 = up_conv(self.feat * 8, self.feat * 4)
        self.Att4 = Attention_block(F_g=self.feat * 4, F_l=self.feat * 4, F_int=self.feat * 2)
        self.Up_conv4 = conv_block(self.feat * 8, self.feat * 4)

        self.Up3 = up_conv(self.feat * 4, self.feat * 2)
        self.Att3 = Attention_block(F_g=self.feat * 2, F_l=self.feat * 2, F_int=self.feat)
        self.Up_conv3 = conv_block(self.feat * 4, self.feat * 2)

        self.Up2 = up_conv(self.feat * 2, self.feat)
        self.Att2 = Attention_block(F_g=self.feat, F_l=self.feat, F_int=self.feat // 2)
        self.Up_conv2 = conv_block(self.feat * 2, self.feat)

        self.Conv_1x1 = nn.Conv2d(self.feat, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)
        d1 = torch.squeeze(d1, dim=1)

        return d1
