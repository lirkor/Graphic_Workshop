import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision



class PatchSTN(nn.Module):


    def __init__(self, spatial_size, num_channels):
        super(PatchSTN, self).__init__()

        self.num_input_channels = num_channels
        self.localization = nn.Sequential(
            nn.Conv2d(num_channels, num_channels+4, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(num_channels+4, num_channels+8, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(num_channels+8, num_channels+16, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        self.localization_output_size = self._compute_localization_output_size(spatial_size)
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear((num_channels+16) * self.localization_output_size * self.localization_output_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self,x):
        xs = self.localization(x)
        xs = xs.view(-1, (self.num_input_channels+16) * self.localization_output_size*self.localization_output_size)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x



    def _compute_localization_output_size(self, spatial_input_size):
        first_block = self._compute_convlution_output_size(
                        self._compute_convlution_output_size(spatial_input_size, 7), 2, stride=2)
        second_block = self._compute_convlution_output_size(
                        self._compute_convlution_output_size(first_block, 5), 2, stride=2)
        third_block = self._compute_convlution_output_size(
                        self._compute_convlution_output_size(second_block, 3), 2, stride=2)

        return third_block


    def _compute_convlution_output_size(self, input_size,kernel, pad=0,stride=1):
        return (int((input_size+2*pad - (kernel-1) -1)/stride) +1)



class MultiPatchSTN(nn.Module):


    def __init__(self, in_channels):
        super(MultiPatchSTN,self).__init__()

        self.stn_stage_9patch_branch1 = MultiPatchSTNLayer(9,in_channels)
        self.stn_stage_9patch_branch2 = MultiPatchSTNLayer(9,in_channels)
        self.branch2_dwconv_pool1 = nn.Sequential(nn.Conv2d(in_channels,in_channels,7,1,3,groups=in_channels), nn.Hardtanh(),
                                                  nn.Conv2d(in_channels,in_channels,1,1), nn.Hardtanh(),
                                                nn.MaxPool2d(3,2))
        self.deconv1_branch2 = nn.Sequential(nn.ConvTranspose2d(in_channels,in_channels,4,2,groups=1), nn.Hardtanh())
        # Add
        self.combine_streams1 = nn.Sequential(nn.Conv2d(8,4,1,1), nn.Hardtanh())
        self.stn_stage_all1 = PatchSTN(128,4)
        self.stn_stage_4patch_branch21 =MultiPatchSTNLayer(4,in_channels)
        self.stn_stage_4patch_branch22 =MultiPatchSTNLayer(4,in_channels)
        self.branch22_dwconv_pool1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 7, 1, 3, groups=in_channels),
                                                   nn.Tanh(), nn.Conv2d(in_channels, in_channels, 1, 1),
                                                  nn.Tanh(),
                                                  nn.MaxPool2d(3, 2))
        self.deconv2_branch22 = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels, 4, 2, groups=1), nn.Tanh())
        self.combine_streams2 = nn.Sequential(nn.Conv2d(8, 4, 1, 1), nn.Hardtanh())
        # Add
        self.stn_stage_all2 =PatchSTN(128,4)
        self.mask_brightness_dim_weights = nn.Parameter(torch.tensor(torch.ones((128,128))),requires_grad=True)


    def apply_mask_mul(self, tensor4d):
        out_sb_stn1_mask_thresh = tensor4d[:, :1, :, :].repeat((1,3,1,1))
        out_sb_stn1 = torch.cat(
            (tensor4d[:, :1, :, :], torch.mul(out_sb_stn1_mask_thresh, tensor4d[:, 1:, :, :])), 1)

        return out_sb_stn1

    def forward(self, tensor_4d, my_input, return_mask = False):
        # x = self.apply_mask_mul(tensor_4d)[:,1:,:,:]
        out_stn1 = self.stn_stage_9patch_branch1(tensor_4d)
        out_sb_stn1 = self.deconv1_branch2(self.branch2_dwconv_pool1(self.stn_stage_9patch_branch2(tensor_4d)))

        # out_sb_stn1 = self.apply_mask_mul(out_sb_stn1)
        # out_res_branch1 = torch.add(out_stn1,out_sb_stn1)
        out_b1_concat = torch.cat((out_stn1,out_sb_stn1),1)
        out_b1 = self.combine_streams1(out_b1_concat)
        out_res_branch1 = self.stn_stage_all1(out_b1)

        out_stn2 = self.stn_stage_4patch_branch21(out_res_branch1)
        out_sb_stn2 = self.deconv2_branch22(self.branch22_dwconv_pool1(self.stn_stage_4patch_branch22(out_res_branch1)))

        # out_sb_stn2 = self.apply_mask_mul(out_sb_stn2)
        # out_res_branch2 = torch.add(out_stn2, out_sb_stn2)
        out_b2concat = torch.cat((out_stn2, out_sb_stn2),1)
        out_b2 = self.combine_streams1(out_b2concat)
        out = self.stn_stage_all2(out_b2)

        # out_mask = out_res_branch2[:,:1,:,:]
        # out_decoded = out_res_branch2[:,1:,:,:]
        # out_mask = torch.sigmoid(out_mask)

        # out = torch.tanh(out_res_branch2)
        input_with_new_content, mask = self.finallize_transform(out, my_input)
        # out_mask = out_mask.repeat(1, 3, 1, 1)
        # out = out_decoded * out_mask + my_input * (1 - out_mask)

        if return_mask:
            return input_with_new_content,out, mask

        return input_with_new_content, out

    def finallize_transform(self, out_tensor4d, input_to_apply):
        # _device = "cuda" if torch.cuda.is_available() else "cpu"
        mask = out_tensor4d[:,:1,:,:]
        oimg = out_tensor4d[:,1:,:,:]
        mask = torch.mul(self.mask_brightness_dim_weights , mask)
        mask = torch.sigmoid(mask)
        oimg = torch.tanh(oimg)
        # output = torch.cat((mask, oimg), 1)
        mask = mask.repeat(1, 3, 1, 1)
        oimg = oimg * mask + input_to_apply * (1 - mask)
        # rgb_to_gray=torch.tensor([0.2126,0.7152,0.0722], device=_device).unsqueeze(0).view(-1,3,1,1)\
        #     .repeat((new_color_mask.shape[0],1,1,1))
        # _binary_mask = torch.sum((torch.sigmoid(new_color_mask) * rgb_to_gray),dim=1,keepdim=True).repeat((1,3,1,1))
        # out = new_color_mask + input_to_apply * (1 - _binary_mask)

        return oimg, mask





class MultiPatchSTNLayer(nn.Module):

    def __init__(self, num_pathces, num_channels,input_size = 128):
        super(MultiPatchSTNLayer, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_patches = num_pathces
        self.num_patches_per_row_col = int(num_pathces**0.5)
        self.input_size =input_size
        self.single_patch_size = int(np.ceil(input_size/self.num_patches_per_row_col))
        self.zeros_diff = abs(input_size-self.single_patch_size*self.num_patches_per_row_col)
        for patch_num in range(num_pathces):
            self.__setattr__("patch_{}".format(patch_num), PatchSTN(self.single_patch_size, num_channels))



    def forward(self,x):
        if self.zeros_diff>0:
            x = F.pad(x,(0,self.zeros_diff,0,self.zeros_diff))
        # print(x.device)
        # tensor_patches = []
        recon_tensor = torch.tensor([], device=self.device)
        rows_tensor = torch.tensor([], device=self.device)
        for stn_patch_num in range(self.num_patches):
            patch_row, patch_col = divmod(stn_patch_num, self.num_patches_per_row_col)
            tensor_row_start = patch_row * self.single_patch_size
            tensor_row_length = min(self.single_patch_size, self.input_size-self.single_patch_size*patch_row)
            tensor_row_end = (patch_row+1) * self.single_patch_size
            tensor_col_start = patch_col * self.single_patch_size
            tensor_col_length = min(self.single_patch_size, self.input_size - self.single_patch_size*patch_col)
            tensor_col_end = (patch_col+1) * self.single_patch_size
            _tensor = torch.narrow(x,2,tensor_row_start,tensor_row_length).narrow(3,tensor_col_start,tensor_col_length)
            # _tensor = x[:,:,tensor_row_start:tensor_row_end,tensor_col_start:tensor_col_end]
            out = self.__getattr__("patch_{}".format(stn_patch_num))(_tensor)
            rows_tensor = torch.cat([rows_tensor, out],3)
            if patch_col == self.num_patches_per_row_col-1:
                recon_tensor =torch.cat([recon_tensor, rows_tensor],2)
                del rows_tensor
                rows_tensor = torch.tensor([], device=self.device)

            # tensor_patches.append(out)

        # tensor_out_rows = []
        # for row in range(self.num_patches_per_row_col):
        #     tensor_out_rows.append(torch.cat(tensor_patches[row*self.num_patches_per_row_col:
        #                                                     (row+1)*self.num_patches_per_row_col], 3))
        #
        # out = torch.cat(tensor_out_rows, 2)

        # assert torch.equal(out, recon_tensor)
        if recon_tensor.shape[2] != self.input_size:
            recon_tensor =recon_tensor[:,:,0:self.input_size,0:self.input_size]

        return recon_tensor



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
        net = self.full(net)
        net = net.view(-1, self.sep * self.size * self.size)
        return net


class D_B(nn.Module):
    def __init__(self, size):
        super(D_B, self).__init__()
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



    def forward(self, net, my_input, apply_mask=False):
        net = net.view(-1, 512, self.size, self.size)
        output = self.main(net)
        if not apply_mask:
            return output
        stn_mask = output[:, :1]
        _pre_act_oimg = output[:, 1:]
        mask = torch.sigmoid(stn_mask)
        oimg = torch.tanh(_pre_act_oimg)
        output = torch.cat((mask, oimg),1)
        mask = mask.repeat(1, 3, 1, 1)

        oimg = oimg * mask + my_input * (1 - mask)
        return oimg, output


    @staticmethod
    def finilaize_db(self):
        print("OK")

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

        self.stn_part = STN()

    def forward(self, net, my_input, other_input, threshold, run_stn=True):
        net = net.view(-1, 512, self.size, self.size)
        output = self.main(net)

        stn_mask = output[:, :1]
        if run_stn:
            stn_mask = self.stn_part(stn_mask)

        mask = torch.sigmoid(stn_mask)
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
        net = net.view(-1, (512 - self.sep) * self.size * self.size)
        net = self.classify(net)
        net = net.view(-1)
        return net





















     # self.stn_mask_localization = nn.Sequential(
        #     nn.Conv2d(1, 4, kernel_size=7),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(4, 8, kernel_size=5),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 10, kernel_size=7),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(10, 10, kernel_size=5),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True)
        # )

        # Regressor for the 3 * 2 affine matrix
        # self.stn_mask_fc_loc = nn.Sequential(
        #     nn.Linear(10 * 3 * 3, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, 2 * 3)
        # )

        # Initialize the weights/bias with identity transformation
        # self.stn_mask_fc_loc[2].weight.data.zero_()
        # self.stn_mask_fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        #
        # self.stn_z_raw_localization = nn.Sequential(
        #     nn.Conv2d(3, 8, kernel_size=7),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 10, kernel_size=7),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(10, 16, kernel_size=7),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     # nn.Conv2d(16, 24, kernel_size=5),
        #     # nn.MaxPool2d(2, stride=2),
        #     # nn.ReLU(True)
        # )

        # Regressor for the 3 * 2 affine matrix
        # self.stn_z_raw_fc_loc = nn.Sequential(
        #     nn.Linear(16*10*10, 160),
        #     nn.ReLU(True),
        #     nn.Linear(160, 80),
        #     nn.ReLU(True),
        #     nn.Linear(80, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, 3 * 4)
        # )

        # Initialize the weights/bias with identity transformation
        # self.stn_z_raw_fc_loc[-1].weight.data.zero_()
        # self.stn_z_raw_fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))


    # def forward_stn_mask(self,x_mask):
    #     xs = self.stn_mask_localization(x_mask)
    #     xs = xs.view(-1, 10 * 3 * 3)
    #     theta = self.stn_mask_fc_loc(xs)
    #     theta = theta.view(-1, 2, 3)
    #
    #     grid = F.affine_grid(theta, x_mask.size())
    #     x = F.grid_sample(x_mask, grid)
    #
    #     return x

    # def forward_stn_z_raw(self, x_z):
    #     xs = self.stn_z_raw_localization(x_z)
    #     xs = xs.view(-1, 16 * 10*10)
    #     theta = self.stn_z_raw_fc_loc(xs)
    #     theta = theta.view(-1, 3, 4)
    #     x_z = x_z.unsqueeze(1)
    #     # x_z = x_z.repeat([1,3,1,1,1])
    #     grid = F.affine_grid(theta, x_z.size())
    #     x = F.grid_sample(x_z, grid)[:,0,:,:,:]
    #     return x