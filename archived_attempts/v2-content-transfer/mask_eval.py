import argparse
import os
import torch
from mask_models import E1, E2, D_A, D_B, STN, V2_STN
from mask_utils import save_imgs, load_model_for_eval, load_model_for_eval_pretrained, get_test_imgs
from matplotlib import pyplot as plt
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np


def eval(args):

    if args.gpu > -1:
        torch.cuda.set_device(args.gpu)

    if args.stn_only:
        return evalStnOnly()

    e1 = E1(args.sep, args.resize // 64)
    e2 = E2(args.sep, args.resize // 64)
    d_a = D_A(args.resize // 64)
    d_b = D_B(args.resize // 64)
    stn = V2_STN(args.sep, args.resize // 64, 10)

    if torch.cuda.is_available():
        e1 = e1.cuda()
        e2 = e2.cuda()
        d_a = d_a.cuda()
        d_b = d_b.cuda()
        stn = stn.cuda()

    if args.load != '':
        save_file = os.path.join(args.load, args.check)
        if not args.old_model:
            _iter = load_model_for_eval(save_file, e1, e2, d_a, d_b, stn)
        else:
            _iter = load_model_for_eval_pretrained(save_file, e1, e2, d_a, d_b, stn)

    e1 = e1.eval()
    e2 = e2.eval()
    d_a = d_a.eval()
    d_b = d_b.eval()
    stn = stn.eval()

    if not os.path.exists(args.out) and args.out != "":
        os.mkdir(args.out)

    save_imgs(args, e1, e2, d_a, d_b, stn, _iter)



def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def evalStnOnly():
    IMG_A_INDEX = 3
    IMG_B_INDEX = 2

    e1 = E1(args.sep, args.resize // 64)
    e2 = E2(args.sep, args.resize // 64)
    d_a = D_A(args.resize // 64)
    d_b = D_B(args.resize // 64)
    stn = V2_STN(args.sep, args.resize // 64, 10)

    if torch.cuda.is_available():
        e1 = e1.cuda()
        e2 = e2.cuda()
        d_a = d_a.cuda()
        d_b = d_b.cuda()
        stn = stn.cuda()

    if args.load != '':
        save_file = os.path.join(args.load, args.check)
        _iter = load_model_for_eval(save_file, e1, e2, d_a, d_b, stn)

    test_domA, test_domB = get_test_imgs(args)
    numTests = 20
    fig, axes = plt.subplots(numTests, 2)

    imgA = test_domA[IMG_A_INDEX].unsqueeze(0)
    vutils.save_image(imgA, "before.png")
    imgB = test_domB[IMG_B_INDEX].unsqueeze(0)
    vutils.save_image(imgB, "contentImage.png")

    imagesToSTN = torch.cat([test_domA[IMG_A_INDEX], test_domB[IMG_B_INDEX]], dim=0)
    theta = stn.stnTheta(imagesToSTN.unsqueeze(0)).cpu()
    print("theta is - {}".format(theta))

    # theta = torch.tensor([[1.4357,  0.0954,  0.4000],
    #      [-0.6847,  1.0773,  0.7944]]).unsqueeze_(0)
    # theta = torch.tensor([[15.2544, 2.0866, 2.5938],
    # [-4.0583, 2.0978, 8.4977]]).unsqueeze_(0)
    # theta = torch.tensor([[1.0, 0.0, 0.0],
    #                       [0.0, 1.0, 0.0]]).unsqueeze_(0)
    theta = theta.cuda()
    grid = F.affine_grid(theta, imgA.size())
    transformedImg = F.grid_sample(imgA, grid)
    vutils.save_image(transformedImg, "after.png")

    # img = test_domB[0]
    # with torch.no_grad():
    #     input_tensor = img.cpu()
    #     transformed_input_tensor = stn(img.unsqueeze(0)).cpu()
    #     print(transformed_input_tensor.np()[0])
    #
    #     in_grid = convert_image_np(
    #         vutils.make_grid(input_tensor))
    #
    #     out_grid = convert_image_np(
    #         vutils.make_grid(transformed_input_tensor))
    #
    #     # Plot the results side-by-side
    #     f, axarr = plt.subplots(1, 2)
    #     axarr[0].imshow(in_grid)
    #     axarr[0].set_title('Dataset Images')
    #
    #     axarr[1].imshow(out_grid)
    #     axarr[1].set_title('Transformed Images')
    #     plt.savefig("beforafter")








if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--out', default='out')
    parser.add_argument('--check', default='checkpoint')
    parser.add_argument('--eval_folder', default='')
    parser.add_argument('--load', default='')
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--num_display', type=int, default=6)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--old_model', type=bool, default=False)
    parser.add_argument('--stn_only', type=bool, default=False)

    args = parser.parse_args()

    eval(args)
