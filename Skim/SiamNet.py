import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SiamNet_Alex(nn.Module):

    def __init__(self):
        super(SiamNet_Alex, self).__init__()

        # architecture (AlexNet like)
        self.feat_extraction_x = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),  # conv1
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),  # conv2, group convolution
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),  # conv3
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2),  # conv4
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2)  # conv5
        )

        self.feat_extraction_z = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),  # conv1
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),  # conv2, group convolution
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),  # conv3
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2),  # conv4
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2)  # conv5
        )

        self.feat_z = nn.Sequential(
            self.feat_extraction_z,
            nn.AvgPool2d(7, stride=1)
        )

        self.feat_x = nn.Sequential(
            self.feat_extraction_x,
            nn.AvgPool2d(7, stride=1)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1, 1),
            nn.Dropout2d(0.5)
        )


    def forward(self, z, x):
        """
        forward pass
        z: examplare, BxCxHxW
        x: search region, BxCxH1xW1
        """
        # get features for z and x
        z = self.feat_extraction_z(z)
        z = F.avg_pool2d(z, 7, 1)
        x = self.feat_extraction_x(x)
        x = F.avg_pool2d(x, 7, 1)

        x = x * z.expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x = F.max_pool2d(x, (x.size(2),x.size(2)), stride=1)
        x = torch.cat([x, z], dim=1)

        return self.classifier(x).view(-1)

    def z_branch(self, z):
        return self.feat_z(z)

    def x_branch(self, x, z):
        z = z.expand(x.size(0), z.size(1), 1, 1)
        x = self.feat_x(x)

        x = x * z.expand(x.size(0), z.size(1), x.size(2), x.size(3))
        x = F.max_pool2d(x, x.size(2), stride=1)
        x = torch.cat([x, z], dim=1)

        x = self.classifier(x).view(-1)

        return x

    def load_params(self, model_path):

        pretrain_dict = torch.load(model_path)
        tmp_dict = self.state_dict()
        for key, value in pretrain_dict.items():
            if key in tmp_dict:
                # print('load pretrain params: %s'%key)
                tmp_dict[key] = value
        self.load_state_dict(tmp_dict)


class SiamNet_MobileV1(nn.Module):
    def __init__(self):
        super(SiamNet_MobileV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.feat_extraction_x = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),  # 150

            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),  # 75

            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),  # 38

            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),  # 19

            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),  # 10
        )

        self.feat_extraction_z = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),  # 150

            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),  # 75

            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),  # 38

            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),  # 19

            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),  # 10
        )

        self.feat_z = nn.Sequential(
            self.feat_extraction_z,
            nn.AvgPool2d(5, stride=1)
        )

        self.feat_x = nn.Sequential(
            self.feat_extraction_x,
            nn.AvgPool2d(5, stride=1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(2048, 1, 1),
        )

        self._initialize_weights()

    def forward(self, z, x):
        """
        forward pass
        z: examplare, BxCxHxW
        x: search region, BxCxH1xW1
        """
        # get features for z and x
        z = self.feat_z(z)
        x = self.feat_x(x)

        x = x * z.expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x = F.max_pool2d(x, x.size(2), stride=1)
        x = torch.cat([x, z], dim=1)

        x = self.classifier(x).view(-1)

        return x


    def z_branch(self, z):
        return self.feat_z(z)

    def x_branch(self, x, z):
        z = z.expand(x.size(0), z.size(1), 1, 1)
        x = self.feat_x(x)

        x = x * z.expand(x.size(0), z.size(1), x.size(2), x.size(3))
        x = F.max_pool2d(x, x.size(2), stride=1)
        x = torch.cat([x, z], dim=1)

        x = self.classifier(x).view(-1)

        return x


    def load_params(self, model_path):

        pretrain_dict = torch.load(model_path)
        tmp_dict = self.state_dict()
        for key, value in pretrain_dict.items():
            if key in tmp_dict:
                # print('load pretrain params: %s' % key)
                tmp_dict[key] = value
        self.load_state_dict(tmp_dict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



