import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torchvision


class STN(nn.Module):
    def __init__(self,):
        super(STN, self).__init__()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conv1=nn.Sequential(nn.Conv2d(4,32,kernel_size=3,stride=2,padding=1), nn.LeakyReLU())
        self.conv1_1=nn.Sequential(nn.Conv2d(32,32,kernel_size=3,stride=1, padding=1), nn.LeakyReLU())
        self.conv2=nn.Sequential(nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1), nn.LeakyReLU())
        self.conv2_1=nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64,96,kernel_size=3,stride=2,padding=1), nn.LeakyReLU())
        self.conv3_1 = nn.Sequential(nn.Conv2d(96,96,kernel_size=3,stride=1, padding=1), nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1), nn.LeakyReLU())
        self.conv4_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())

        self.predict_flow1=nn.Conv2d(128,2,3,1,1,bias=True)
        self.upsample_flow1_to_2 = nn.ConvTranspose2d(2,2,4,2,1,bias=True)
        self.deconv1=nn.Sequential(nn.ConvTranspose2d(128,32,4,2,1,bias=True), nn.ReLU6())
        # cat conv3,deconv1,upsample = 96+32+2 = 128

        self.predict_flow2=nn.Conv2d(130,2,3,1,1,bias=True)
        self.upsample_flow2_to_3 = nn.ConvTranspose2d(2,2,4,2,1,bias=True)
        self.deconv2=nn.Sequential(nn.ConvTranspose2d(130,64,4,2,1,bias=True) , nn.ReLU6())
        # cat conv2,deconv2,upsample = 64+32+2 = 98

        self.predict_flow3=nn.Conv2d(130,2,3,1,1,bias=True)
        self.upsample_flow3_to_4 = nn.ConvTranspose2d(2,2,4,2,1,bias=True)
        self.deconv3=nn.Sequential(nn.ConvTranspose2d(130,32,4,2,1,bias=True),nn.Sigmoid())
        # cat conv1,deconv3,upsample = 32+32+2 = 66

        self.predict_flow4=nn.Conv2d(66,2,3,1,1,bias=True)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')

    # Spatial transformer network forward function
    def forward(self, mask, my_input):
        out_conv1 = self.conv1(mask)
        out_conv1_1 = self.conv1_1(out_conv1)
        out_conv2 = self.conv2(out_conv1)
        out_conv2_1 = self.conv2_1(out_conv2)
        out_conv3 = self.conv3(out_conv2)
        out_conv3_1 = self.conv3_1(out_conv3)
        out_conv4 = self.conv4(out_conv3)
        out_conv4_1 = self.conv4_1(out_conv4)

        flow1 = self.predict_flow1(out_conv4_1)
        flow1_up = self.upsample_flow1_to_2(flow1)
        out_deconv1 = self.deconv1(out_conv4_1)
        # print(out_conv3_1.shape, out_deconv1.shape, flow1_up.shape)

        concat1 = torch.cat((out_conv3_1,out_deconv1,flow1_up), 1)

        flow2 = self.predict_flow2(concat1)
        flow2_up = self.upsample_flow2_to_3(flow2)
        out_deconv2=self.deconv2(concat1)
        # print(out_conv2_1.shape, out_deconv2.shape, flow2_up.shape)

        concat2 = torch.cat((out_conv2_1, out_deconv2, flow2_up),1)

        flow3 = self.predict_flow3(concat2)
        flow3_up = self.upsample_flow3_to_4(flow3)
        out_deconv3 = self.deconv3(concat2)
        # print(out_conv1_1.shape, out_deconv3.shape, flow3_up.shape)

        concat3 = torch.cat((out_conv1_1, out_deconv3, flow3_up), 1)

        out_flow = self.upsample(self.predict_flow4(concat3))
        out_flow = torch.tanh(out_flow)

        out_ = self.move_mask(mask,out_flow)
        mask = out_[:, :1]
        recon_attr = out_[:, 1:]

        # mask = torch.sigmoid(mask)
        recon_attr=torch.tanh(recon_attr)
        # output = torch.cat((mask, _dec_sep_oimg),1)
        mask = mask.repeat(1, 3, 1, 1)
        oimg = recon_attr * mask + my_input * (1 - mask)


        # out_ = out_.repeat(1,3,1,1)
        return oimg,mask, recon_attr


    def move_mask(self,orig_mask, flow_move, max_pixel_move = 32):
        # u = flow_move[:,0, :, :].flatten(1)
        # v = flow_move[:,1, :, :].flatten(1)
        # orig_mask_x = orig_mask[:,0,:,:].flatten(1)
        # orig_mask_y = orig_mask[:,1, :, :].flatten(1)
        #
        # maxu = u.max(1).values
        # maxv = v.max(1).values
        # minu = u.min(1).values
        # minv = v.min(1).values
        #
        # u_range = (maxu-minu).view(-1,1)
        # v_range = (maxv-minv).view(-1,1)
        #
        # u_quant_level = u_range/float(max_pixel_move)
        # v_quant_level = v_range/float(max_pixel_move)
        #
        # u_move_map = (u/u_quant_level).type(torch.int).clamp(-max_pixel_move,max_pixel_move)
        # v_move_map =  (v/v_quant_level).type(torch.int).clamp(-max_pixel_move,max_pixel_move)
        # u_new_indices = (torch.arange(0,128*128).repeat(2,1) + u_move_map).clamp(0,128*128-1)
        # v_new_indices = (torch.arange(0,128*128).repeat(2,1) + v_move_map).clamp(0,128*128-1)
        # orig_mask_x.gather(1, u_new_indices)
        # orig_mask_y.gather(1, v_new_indices)
        # moved_mask = torch.cat((moved_mask_x_axis,moved_mask_y_axis),1)


        _batch_size = orig_mask.shape[0]
        u = flow_move[:, 0, :, :]
        v = flow_move[:, 1, :, :]
        orig_mask_m = orig_mask[:, 0, :, :]
        orig_mask_r = orig_mask[:, 1, :, :]
        orig_mask_g = orig_mask[:, 2, :, :]
        orig_mask_b = orig_mask[:, 3, :, :]

        maxu = u.max(1).values.max(1).values
        maxv = v.max(1).values.max(1).values
        minu = u.min(1).values.max(1).values
        minv = v.min(1).values.max(1).values

        # u_range = (maxu - minu).view(-1, 1,1)
        # v_range = (maxv - minv).view(-1, 1,1)

        # u_quant_level = u_range / float(max_pixel_move)
        # v_quant_level = v_range / float(max_pixel_move)
        u_quant_level = 1 /float(64)
        v_quant_level = 1 /float(64)
        u_move_map = (u / u_quant_level).type(torch.int).clamp(-max_pixel_move, max_pixel_move)
        v_move_map = (v / v_quant_level).type(torch.int).clamp(-max_pixel_move, max_pixel_move)
        u_new_indices = (torch.arange(0, 128,device=self._device).repeat(_batch_size, 128,1) + u_move_map).clamp(0, 128 - 1)
        v_new_indices = (torch.arange(0, 128, device=self._device).repeat(_batch_size, 128,1) + v_move_map).clamp(0, 128 - 1)

        moved_mask_r = orig_mask_r.gather(1, u_new_indices)
        moved_mask_r = moved_mask_r.transpose(1,2).gather(1, v_new_indices).transpose(1,2)

        moved_mask_g = orig_mask_g.gather(1, u_new_indices)
        moved_mask_g = moved_mask_g.transpose(1, 2).gather(1, v_new_indices).transpose(1, 2)

        moved_mask_b = orig_mask_b.gather(1, u_new_indices)
        moved_mask_b = moved_mask_b.transpose(1, 2).gather(1, v_new_indices).transpose(1, 2)

        moved_mask_m = orig_mask_m.gather(1, u_new_indices)
        moved_mask_m = moved_mask_m.transpose(1, 2).gather(1, v_new_indices).transpose(1, 2)
        moved_mask_r=moved_mask_r.unsqueeze(1)
        moved_mask_g=moved_mask_g.unsqueeze(1)
        moved_mask_b=moved_mask_b.unsqueeze(1)
        moved_mask_m=moved_mask_m.unsqueeze(1)
        moved_mask = torch.cat((moved_mask_r,moved_mask_g,moved_mask_b,moved_mask_m), 1)


        return moved_mask



class _STN(nn.Module):

    def __init__(self, conf_num, num_channels):
        super(_STN, self).__init__()

        self.conf = conf_num
        self.num_channels = num_channels

        if conf_num==1:
            self.localization = nn.Sequential(
                nn.Conv2d(num_channels, 4, kernel_size=5), nn.MaxPool2d(2, stride=2), nn.ReLU(True),
                nn.Conv2d(4, 8, kernel_size=5), nn.MaxPool2d(2, stride=2),nn.ReLU(True),
                nn.Conv2d(8, 12, kernel_size=3),nn.MaxPool2d(2, stride=2), nn.ReLU(True),
                nn.Conv2d(12, 16, kernel_size=3), nn.MaxPool2d(2, stride=2), nn.ReLU(True))
            self.fc_loc = nn.Sequential(
                nn.Linear(400, 128),
                nn.ReLU(True),
                nn.Linear(128, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2),
            )

        elif conf_num==2:
            self.localization = nn.Sequential(
                nn.Conv2d(num_channels, 4, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(4, 8, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 12, kernel_size=3),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True))

            self.fc_loc = nn.Sequential(
                nn.Linear(300, 128),
                nn.ReLU(True),
                nn.Linear(128, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2),
            )

        # Regressor for the 3 * 2 affine matrix
        # self.fc_loc = nn.Sequential(
        #     nn.Linear(400, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128,32),
        #     nn.ReLU(True),
        #     nn.Linear(32,3*2),
        # )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self,x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.shape[1] * 5*5)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # x_z = x.unsqueeze(1)
        # x_z = x_z.repeat([1,3,1,1,1])
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x



class MainTransformer(nn.Module):

    def __init__(self):
        super(MainTransformer, self).__init__()
        # input is mask*decoder image

        self.first_stn = _STN(1,3)
        self.dw_conv1= nn.Sequential(nn.Conv2d(3,3,3,1,1,groups=3), nn.ReLU())
        # cat with input
        self.dw_conv2 = nn.Sequential(nn.Conv2d(6,3,3,2,groups=3), nn.LeakyReLU(0.2))
        self.second_stn = _STN(2,3)
        self.dw_conv3 =nn.Sequential(nn.Conv2d(3,3,3,1,1,groups=3), nn.LeakyReLU(inplace=True))
        self.conv1=nn.Sequential(nn.Conv2d(3,3,3,1,1), nn.LeakyReLU(inplace=True))
        #cat both depthwise
        self.conv2_merged_stream1=nn.Sequential(nn.Conv2d(6,3,3,1,1, groups=3), nn.LeakyReLU(inplace=True))
        self.dw_conv_input = nn.Sequential(nn.Conv2d(3,3,3,2,0,groups=3), nn.LeakyReLU(inplace=True), nn.Conv2d(3,3,1,1,0))
        # cat conv2_merged with input dw
        self.maxpool1= nn.MaxPool2d(3,2)
        self.dw_conv4 = nn.Sequential(nn.Conv2d(6,3,3,1,1,groups=3), nn.LeakyReLU(inplace=True), nn.Conv2d(3,3,1,1,0))
        self.third_stn = _STN(2,3)
        self.deconv1 = nn.ConvTranspose2d(3,3,4,2,0,0,groups=3)
        # self.project_normlize_color_conv = nn.Conv2d(3,1,1,1,0, bias=False)


    def zig_zag_cat(self, t1,t2):
        zigzag_cat_1 = torch.stack((t1, t2), 2).view(
            (-1, 6, t1.shape[2], t2.shape[3]))
        # _zigzag_1 = torch.cat((t1[0][0:1], t2[0][0:1], t1[0][1:2], t2[0][1:2],
        #                        t1[0][2:3], t2[0][2:3]), 0)
        # assert (torch.equal(_zigzag_1, zigzag_cat_1[0]))
        return zigzag_cat_1

    def forward(self,color_mask, my_input, return_mask =False):
        transformed1 = self.dw_conv1(self.first_stn(color_mask))

        color_mask_reshape = self.maxpool1(color_mask)
        zigzag_cat_1 = self.zig_zag_cat(transformed1, color_mask)
        dw_conv2 = self.dw_conv2(zigzag_cat_1)
        dw_conv2_input_boosted = torch.add(dw_conv2,color_mask_reshape)

        transformed2=self.second_stn(dw_conv2_input_boosted)
        dw_conv3_out = self.dw_conv3(transformed2)
        conv1_out = self.conv1(transformed2)


        # Add eltwise to add color mask and zigzag
        color_mask_reshape = self.maxpool1(color_mask)
        conv1_input_boost = torch.add(conv1_out, color_mask_reshape)

        zigzag_cat_2 = self.zig_zag_cat(dw_conv3_out, conv1_input_boost)
        conv2_merged_out = self.conv2_merged_stream1(zigzag_cat_2)
        dw_input_conv = self.dw_conv_input(color_mask)

        zigzag_cat_3 = self.zig_zag_cat(conv2_merged_out, dw_input_conv)
        out = self.deconv1(self.third_stn(self.dw_conv4(zigzag_cat_3)))
        # out_thresh = torchvision.transforms.Grayscale(3)(out)
        # out_thresh = out_thresh(out)
        out_thresh = out.mean(1, keepdim=True)
        out = torch.tanh(out)
        # out_thresh=self.project_normlize_color_conv(out)
        out_thresh = torch.sigmoid(out_thresh)

        A_decoded = (1. - out_thresh) * my_input + out * out_thresh

        # if A_decoded.max()>=1. or A_decoded.min()<=-1:
        #     print("Clipping A_decoded")
        #     A_decoded = A_decoded.clamp(-1,1)

        if return_mask:
            return A_decoded, out_thresh
        else:
            return A_decoded







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


    def __init__(self):
        super(MultiPatchSTN,self).__init__()

        self.stn_stage_9patch_branch1 = MultiPatchSTNLayer(9,4)
        self.stn_stage_9patch_branch2 = MultiPatchSTNLayer(9,4)
        self.branch2_dwconv_pool1 = nn.Sequential(nn.Conv2d(4,4,7,1,3,groups=4), nn.Tanh(), nn.Conv2d(4,4,1,1), nn.Tanh(),
                                                nn.MaxPool2d(3,2))
        self.deconv1_branch2 = nn.Sequential(nn.ConvTranspose2d(4,4,4,2,groups=1), nn.Tanh())
        # Add
        self.stn_stage_all1 = PatchSTN(128,4)
        self.stn_stage_4patch_branch21 =MultiPatchSTNLayer(4,4)
        self.stn_stage_4patch_branch22 =MultiPatchSTNLayer(4,4)
        self.branch22_dwconv_pool1 = nn.Sequential(nn.Conv2d(4, 4, 7, 1, 3, groups=4), nn.Tanh(), nn.Conv2d(4, 4, 1, 1),
                                                  nn.Tanh(),
                                                  nn.MaxPool2d(3, 2))
        self.deconv2_branch22 = nn.Sequential(nn.ConvTranspose2d(4, 4, 4, 2, groups=1), nn.Tanh())
        # Add
        self.stn_stage_all2 =PatchSTN(128,4)



    def forward(self, x, my_input, return_mask = False):
        out_stn1 = self.stn_stage_9patch_branch1(x)
        out_sb_stn1 = self.deconv1_branch2(self.branch2_dwconv_pool1(self.stn_stage_9patch_branch2(x)))
        out_res_branch1 = torch.add(out_stn1,out_sb_stn1)

        out_res_branch1 = self.stn_stage_all1(out_res_branch1)

        out_stn2 = self.stn_stage_4patch_branch21(out_res_branch1)
        out_sb_stn2 = self.deconv2_branch22(self.branch22_dwconv_pool1(self.stn_stage_4patch_branch22(out_res_branch1)))
        out_res_branch2 = torch.add(out_stn2, out_sb_stn2)

        out_res_branch2 = self.stn_stage_all2(out_res_branch2)

        out_mask = out_res_branch2[:,:1,:,:]
        out_decoded = out_res_branch2[:,1:,:,:]
        out_mask = torch.sigmoid(out_mask)
        out_decoded = torch.tanh(out_decoded)
        out_mask = out_mask.repeat(1, 3, 1, 1)
        out = out_decoded * out_mask + my_input * (1 - out_mask)

        if return_mask:
            return out, out_mask

        return out




class MultiPatchSTNLayer(nn.Module):

    def __init__(self, num_pathces, num_channels,input_size = 128):
        super(MultiPatchSTNLayer, self).__init__()
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

        tensor_patches = []
        for stn_patch_num in range(self.num_patches):
            patch_row, patch_col = divmod(stn_patch_num, self.num_patches_per_row_col)
            tensor_row_start = patch_row * self.single_patch_size
            tensor_row_end = (patch_row+1) * self.single_patch_size
            tensor_col_start = patch_col * self.single_patch_size
            tensor_col_end = (patch_col+1) * self.single_patch_size
            _tensor = x[:,:,tensor_row_start:tensor_row_end,tensor_col_start:tensor_col_end]
            out = self.__getattr__("patch_{}".format(stn_patch_num))(_tensor)
            tensor_patches.append(out)

        tensor_out_rows = []
        for row in range(self.num_patches_per_row_col):
            tensor_out_rows.append(torch.cat(tensor_patches[row*self.num_patches_per_row_col:
                                                            (row+1)*self.num_patches_per_row_col], 3))

        out = torch.cat(tensor_out_rows, 2)

        if out.shape[2] != self.input_size:
            out =out [:,:,0:self.input_size,0:self.input_size]

        return out



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



    def forward(self, net, my_input, run_stn = False):
        net = net.view(-1, 512, self.size, self.size)
        output = self.main(net)
        stn_mask = output[:, :1]
        _pre_act_oimg = output[:, 1:]
        if run_stn:
            stn_mask = self.forward_stn_mask(stn_mask)
            _pre_act_oimg = self.forward_stn_z_raw(_pre_act_oimg)
        mask = torch.sigmoid(stn_mask)
        _dec_sep_oimg = torch.tanh(_pre_act_oimg)
        output = torch.cat((mask, _dec_sep_oimg),1)
        mask = mask.repeat(1, 3, 1, 1)
        # oimg = _dec_sep_oimg * mask + my_input * (1 - mask)
        return  output, my_input



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