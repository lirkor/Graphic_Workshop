import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from mask_utils import WarpImage

debug = False

class E1(nn.Module):
    def __init__(self, sep, size):
        super(E1, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, (512 - self.sep), 4, 2, 1),
            nn.BatchNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d((512 - self.sep), (512 - self.sep), 4, 2, 1),
            nn.BatchNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, net):
        if(debug):
            print("E1 pass")
        net = self.full(net)
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        return net


class E2(nn.Module):
    def __init__(self, sep, size):
        super(E2, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, self.sep, 4, 2, 1),
            nn.BatchNorm2d(self.sep),
            nn.LeakyReLU(0.2),
        )

    def forward(self, net):
        if (debug):
            print("E2 pass")
        net = self.full(net)
        net = net.view(-1, self.sep * self.size * self.size)
        return net


class D_B(nn.Module):
    def __init__(self, size):
        super(D_B, self).__init__()
        self.size = size

        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 4, 4, 2, 1),
        )

    def warpImg(self, img, theta):
        grid = F.affine_grid(theta, img.size())
        img = F.grid_sample(img, grid)
        return img

    def forwardImgAToSTN(self, imgEncoding, targetImage, stn, showTheta = False):
        if (len(targetImage.size()) == 3):
            toStn = targetImage.unsqueeze(0)
        else:
            toStn = targetImage
        theta = stn.stnTheta(toStn)
        if (showTheta):
            print(theta.cpu())
        if (debug):
            print("D_B pass")
        imgEncoding = imgEncoding.view(-1, 512, self.size, self.size)
        output = self.net(imgEncoding)
        mask = torch.sigmoid(output[:, :1])
        oimg = torch.tanh(output[:, 1:])
        mask = mask.repeat(1, 3, 1, 1)

        mask = self.warpImg(mask, theta)
        oimg = self.warpImg(oimg, theta)
        return oimg, mask

    def forwardImgAImgBToSTN(self, imgEncoding, domB_img, targetImage, stn, showTheta=False):
        if (len(targetImage.size()) == 3):
            imgToStn = torch.cat([targetImage.unsqueeze(0), domB_img.unsqueeze(0)], dim=1)
        else:
            imgToStn = torch.cat([targetImage, domB_img], dim=1)
        theta = stn.stnTheta(imgToStn)
        if (showTheta):
            print(theta.cpu())
        if (debug):
            print("D_B pass")
        imgEncoding = imgEncoding.view(-1, 512, self.size, self.size)
        output = self.net(imgEncoding)
        mask = torch.sigmoid(output[:, :1])
        oimg = torch.tanh(output[:, 1:])
        mask = mask.repeat(1, 3, 1, 1)

        mask = self.warpImg(mask, theta)
        oimg = self.warpImg(oimg, theta)
        return oimg, mask

    def forwardImgAImgBAndEncodingToSTN(self, imgEncoding, domB_img, targetImage, stn, showTheta=False):
        if (len(targetImage.size()) == 3):
            imgToStn = torch.cat([targetImage.unsqueeze(0), domB_img.unsqueeze(0)], dim=1)
        else:
            imgToStn = torch.cat([targetImage, domB_img], dim=1)

        imgEncodingForTheta = imgEncoding.view(-1, 2048)

        theta = stn.stnThetaCombined(imgToStn, imgEncodingForTheta)
        if (showTheta):
            print(theta.cpu())
        if (debug):
            print("D_B pass")

        imgEncoding = imgEncoding.view(-1, 512, self.size, self.size)
        output = self.net(imgEncoding)
        mask = torch.sigmoid(output[:, :1])
        oimg = torch.tanh(output[:, 1:])
        mask = mask.repeat(1, 3, 1, 1)

        mask = self.warpImg(mask, theta)
        oimg = self.warpImg(oimg, theta)
        return oimg, mask


    def forwardToSTNPerPixel(self, imgEncoding, domB_img, targetImage, stn, showTheta=False):
        if (len(targetImage.size()) == 3):
            imgToStn = torch.cat([targetImage.unsqueeze(0), domB_img.unsqueeze(0)], dim=1)
        else:
            imgToStn = torch.cat([targetImage, domB_img], dim=1)

        imgEncodingForTheta = imgEncoding.view(-1, 2048)

        UV = stn.stnPerPixel(imgToStn, imgEncodingForTheta).detach().cpu().numpy()
        if (debug):
            print("D_B pass")

        imgEncoding = imgEncoding.view(-1, 512, self.size, self.size)
        output = self.net(imgEncoding)
        mask = torch.sigmoid(output[:, :1])
        oimg = torch.tanh(output[:, 1:])
        mask = mask.repeat(1, 3, 1, 1)

        mask = WarpImage(mask, UV)
        oimg = WarpImage(oimg, UV)
        return oimg, mask

    def generateOutput(self, oimg, mask, my_input):
        oimg = oimg * mask + my_input * (1 - mask)
        return oimg, mask

    def forward(self, imgEncoding, targetImage, domB_img, stn, encodingForSTN, showTheta = False):
        #oimg, mask = self.forwardImgAToSTN(imgEncoding, targetImage, stn, showTheta = False)
        #oimg, mask = self.forwardImgAImgBToSTN(imgEncoding, targetImage, domB_img, stn, showTheta)
        oimg, mask = self.forwardImgAImgBAndEncodingToSTN(imgEncoding, targetImage, domB_img, stn, showTheta)
        #oimg, mask = self.forwardToSTNPerPixel(imgEncoding, targetImage, domB_img, stn, showTheta)
        oimg, mask = self.generateOutput(oimg, mask, targetImage)
        return oimg, mask


class D_B_removal(nn.Module):
    def __init__(self, size):
        super(D_B_removal, self).__init__()
        self.size = size

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 4, 4, 2, 1),
        )

    def forward(self, net, my_input, other_input, threshold):
        if (debug):
            print("start D_B pass")
        net = net.view(-1, 512, self.size, self.size)
        output = self.main(net)
        mask = torch.sigmoid(output[:, :1])
        mask = mask.ge(threshold)
        mask = mask.type(torch.cuda.FloatTensor)
        mask = mask.repeat(1, 3, 1, 1)
        oimg = other_input * mask + my_input * (1 - mask)
        return oimg, mask

class D_A(nn.Module):
    def __init__(self, size):
        super(D_A, self).__init__()
        self.size = size

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, net):
        if (debug):
            print("D_A pass")
        net = net.view(-1, 512, self.size, self.size)
        net = self.main(net)
        return net

class Disc(nn.Module):
    def __init__(self, sep, size):
        super(Disc, self).__init__()
        self.sep = sep
        self.size = size

        self.classify = nn.Sequential(
            nn.Linear((512 - self.sep) * self.size * self.size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, net):
        if (debug):
            print("Disc pass")
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        net = self.classify(net)
        net = net.view(-1)
        return net


class STN(nn.Module):
    def __init__(self, sep, size, localizationDepth):
        super(STN, self).__init__()
        # self.conv0 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        # self.conv1 = nn.ConvTranspose2d(512, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)
        self.sep = sep
        self.size = size
        self.localizationDepth = localizationDepth

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, localizationDepth, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(localizationDepth * 28 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0,  0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def toStn(self, x):
        if (debug):
            print("self.sep {}".format(self.sep))
            print("self.size {}".format(self.size))
            print("x shape {}".format(x.shape))
        xs = self.localization(x)
        if (debug):
            print("xs shape {}".format(xs.shape))

        xs = xs.view(-1, self.localizationDepth * 28 * 28)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        if (debug):
            print("theta shape {}".format(theta.shape))
            print("x size {}".format(x.size()))

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        #x = x.view(-1, (512 - self.sep) * self.size * self.size)
        return x

    def stnTheta(self, x):
        xs = self.localization(x)
        if (debug):
            print("xs shape {}".format(xs.shape))
        # xs = xs.view(-1, 1536 * 3 * 3)
        xs = xs.view(-1, 10 * 28 * 28)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta

    def forward(self, x):
        if (debug):
            print("start stn pass")
        # transform the input
        x = self.toStn(x)
        return x


class V2_STN(nn.Module):
    def __init__(self, sep, size, localizationDepth):
        super(V2_STN, self).__init__()
        # self.conv0 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        # self.conv1 = nn.ConvTranspose2d(512, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)
        self.sep = sep
        self.size = size
        self.localizationDepth = localizationDepth

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(12, localizationDepth, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(localizationDepth * 28 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 3)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_combined = nn.Sequential(
            nn.Linear(localizationDepth * 28 * 28 + 2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 3)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_per_pixel = nn.Sequential(
            nn.Linear(localizationDepth * 28 * 28 + 2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Linear(4096, 16384),
            nn.BatchNorm1d(16384),
            nn.ReLU(True),
            nn.Linear(16384, 32768)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0,  0, 1, 0], dtype=torch.float))

        self.fc_loc_combined[-1].weight.data.zero_()
        self.fc_loc_combined[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        #TODO change this to none zero values
        self.fc_per_pixel[-1].weight.data.zero_()

    # Spatial transformer network forward function
    def toStn(self, x):
        if (debug):
            print("self.sep {}".format(self.sep))
            print("self.size {}".format(self.size))
            print("x shape {}".format(x.shape))
        xs = self.localization(x)
        if (debug):
            print("xs shape {}".format(xs.shape))

        xs = xs.view(-1, self.localizationDepth * 28 * 28)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        if (debug):
            print("theta shape {}".format(theta.shape))
            print("x size {}".format(x.size()))

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        #x = x.view(-1, (512 - self.sep) * self.size * self.size)
        return x

    def stnTheta(self, x):
        xs = self.localization(x)
        if (debug):
            print("xs shape {}".format(xs.shape))
        # xs = xs.view(-1, 1536 * 3 * 3)
        xs = xs.view(-1, self.localizationDepth * 28 * 28)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta

    def stnThetaCombined(self, x, imgEncoding):
        xs = self.localization(x)
        if (debug):
            print("xs shape {}".format(xs.shape))
        # xs = xs.view(-1, 1536 * 3 * 3)size
        xs = xs.view(-1, self.localizationDepth * 28 * 28)
        #print(xs.size())
        #print(imgEncoding.size())
        to_fc = torch.cat([xs, imgEncoding], dim=1)
        #print(to_fc.size())
        theta = self.fc_loc_combined(to_fc)
        theta = theta.view(-1, 2, 3)
        return theta

    def stnPerPixel(self, x, imgEncoding):
        xs = self.localization(x)
        if (debug):
            print("xs shape {}".format(xs.shape))
        # xs = xs.view(-1, 1536 * 3 * 3)size
        xs = xs.view(-1, self.localizationDepth * 28 * 28)
        #print(xs.size())
        #print(imgEncoding.size())
        to_fc = torch.cat([xs, imgEncoding], dim=1)
        #print(to_fc.size())
        UV = self.fc_per_pixel(to_fc)
        UV = UV.view(-1, 128, 128, 2)
        #print(UV.size())
        return UV

    def forward(self, x):
        if (debug):
            print("start stn pass")
        # transform the input
        x = self.toStn(x)
        return x


