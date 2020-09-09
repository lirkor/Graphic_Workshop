import os, sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,os.getcwd())
# sys.path.insert(0, os.path.join(os.getcwd(), "mbu_content_transfer"))
# from module_stn import STN
# from mbu_content_tansfer import mask_models,mask_utils
import mask_models, mask_utils

import torch
import torchvision
from torch import nn
from torch import optim
import argparse
from random import uniform, randint
import numpy as np
import time








def train(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    if args.gpu > -1:
        torch.cuda.set_device(args.gpu)

    print("=====  args ======")
    for a in args._get_kwargs():
        print("{} = {}".format(a[0],a[1]))
    # print("Alpha1 is " + str(args.alpha1))
    # print("Alpha2 is " + str(args.alpha2))
    # print("Beta1 is " + str(args.beta1))
    # print("Beta2 is " + str(args.beta2))
    # print("Gama is " + str(args.gama))
    # print("Delta is " + str(args.delta))
    # print("discweight is " + str(args.discweight))
    # print("Noise is: " + str(args.noise))
    # print("Iter is: "+ str(args.iters))
    print("==================", end = "\n\n")

    _iter = 0

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.resize, args.resize)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    domA_train = mask_utils.CustomDataset(os.path.join(args.root, 'trainA.txt'), transform=transform)
    domB_train = mask_utils.CustomDataset(os.path.join(args.root, 'trainB.txt'), transform=transform)

    A_label = torch.full((args.bs,), 1)
    B_label = torch.full((args.bs,), 0)
    B_separate = torch.full((args.bs, args.sep * (args.resize // 64) * (args.resize // 64)), 0)

    e1 = mask_models.E1(args.sep, args.resize // 64)
    e2 = mask_models.E2(args.sep, args.resize // 64)
    d_a = mask_models.D_A(args.resize // 64)
    disc = mask_models.Disc(args.sep, args.resize // 64)
    disc_out = mask_models.Disc(509, args.resize)
    d_b = mask_models.D_B(args.resize // 64)
    # stn = mask_models.MainTransformer()
    stn = mask_models.MultiPatchSTN(4)

    mse = nn.MSELoss()
    bce = nn.BCELoss()
    bce_ct = nn.BCELoss()
    l1 = nn.L1Loss()
    mse_match = nn.MSELoss()

    if torch.cuda.is_available():
        print("Cude is detected!\n")
        e1 = e1.cuda()
        e2 = e2.cuda()
        d_a = d_a.cuda()
        d_b = d_b.cuda()
        disc = disc.cuda()
        disc_out = disc_out.cuda()
        stn = stn.cuda()
        A_label = A_label.cuda()
        B_label = B_label.cuda()
        B_separate = B_separate.cuda()
        mse = mse.cuda()
        bce = bce.cuda()
        bce_ct = bce_ct.cuda()
        l1 = l1.cuda()
        mse_match = mse_match.cuda()

    ae_params = list(e1.parameters()) + list(e2.parameters()) + list(d_a.parameters()) + list(d_b.parameters()) \
                + list(stn.parameters())
    ae_optimizer = optim.Adam(ae_params, lr=args.lr, betas=(0.5, 0.999))

    disc_params = disc.parameters()
    disc_out_params = disc_out.parameters()
    disc_optimizer = optim.Adam(disc_params, lr=args.disclr, betas=(0.5, 0.999))
    disc_out_optimizer = optim.Adam(disc_out_params, lr=args.disclr, betas=(0.5, 0.999))

    if args.load != '':
        save_file = os.path.join(args.load, 'checkpoint')
        _iter = mask_utils.load_model(save_file, e1, e2, d_a, ae_optimizer, disc, disc_optimizer)

    e1 = e1.train()
    e2 = e2.train()
    d_a = d_a.train()
    d_b = d_b.train()
    disc = disc.train()
    disc_out = disc_out.train()
    stn = stn.train()

    print('Started training...')
    while True:

        domA_loader = torch.utils.data.DataLoader(dataset=domA_train, batch_size=args.bs, shuffle=True)
        domB_loader = torch.utils.data.DataLoader(dataset=domB_train, batch_size=args.bs, shuffle=True)

        if _iter >= args.iters:
            break

        for domA_img, domB_img in zip(domA_loader, domB_loader):
            if args.print_runtime:
                t0 = time.time()

            if domA_img.size(0) != args.bs or domB_img.size(0) != args.bs:
                break

            if torch.cuda.is_available():
                domA_img = domA_img.cuda()
                domB_img = domB_img.cuda()
            else:
                domA_img = domA_img
                domB_img = domB_img

            domA_img = domA_img.view((-1, 3, args.resize, args.resize))
            domB_img = domB_img.view((-1, 3, args.resize, args.resize))

            ae_optimizer.zero_grad()

            if args.noise>0:
                noise_ratio = args.noise
                val = uniform(0,1)
                if val<noise_ratio:
                    roll_noise_h = randint(-int(uniform(0.4,0.8)*32),int(uniform(0.4,0.9)*32))
                    roll_noise_w = randint(-int(uniform(0.4,0.8)*32),int(uniform(0.4,0.9)*32))
                    domA_img = domA_img.roll((roll_noise_h,roll_noise_w),dims=(2,3))
                    roll_noise_h = randint(-int(uniform(0.4, 0.8) * 32), int(uniform(0.45, 0.85) * 32))
                    roll_noise_w = randint(-int(uniform(0.4, 0.8) * 32), int(uniform(0.45, 0.85) * 32))
                    domB_img = domB_img.roll((roll_noise_h,roll_noise_w),dims=(2,3))


            A_common = e1(domA_img)
            A_separate = e2(domA_img)
            A_encoding = torch.cat([A_common, A_separate], dim=1)
            A_shaved_encoding = torch.cat([A_common, B_separate], dim=1)


            B_common = e1(domB_img)
            B_encoding = torch.cat([B_common, B_separate], dim=1)

            A_decoded, _ = d_b(A_encoding, d_a(A_shaved_encoding), True)

            # colored_mask_to_apply = mask_utils.process_post_db(zraw_mask)
            # maybe here to put directly to recon1 loss, no STN.
            # A_decoded = stn(zraw_mask, da_shaved)

            # A_decoded = (1-new_threshold_mask)*domA_img + new_colored_mask

            B_decoding = d_a(B_encoding)

            #Reconstruction
            loss = args.gama * l1(A_decoded, domA_img) + args.delta * l1(B_decoding, domB_img)

            ''' ======== Cycle ============='''
            C_encoding = torch.cat([B_common, A_separate], dim=1)
            C_zraw_mask = d_b(C_encoding, domB_img)

            # colored_mask_to_apply = mask_utils.process_post_db(C_zraw_mask)
            C_decoded, stn_out = stn(C_zraw_mask, domB_img)

            # new_colored_mask, new_threshold_mask = stn(colored_mask_to_apply)
            # C_decoded = (1 - new_threshold_mask) * domB_img + new_colored_mask


            e1_b = e1(C_decoded)
            e2_a = e2(C_decoded)
            # Cycle loss
            loss += args.beta1 * mse(e1_b, B_common) + args.beta2 * mse(e2_a, A_separate)

            # move loss
            if args.stn_match_loss_const > 0:
                loss += (args.stn_match_loss_const * (args.stn_match_loss_decay ** _iter)) *\
                        mse_match(C_zraw_mask, stn_out)


            ''' ========== Recon2 =========='''
            B_decoded, _ = d_b(B_encoding, domB_img, True)
            A_decoded, _ = d_b(A_encoding, domA_img, True)

            # colored_mask_to_apply = mask_utils.process_post_db(B_zraw)
            # B_decoded = stn(B_zraw, domB_img)
            #
            # # colored_mask_to_apply = mask_utils.process_post_db(A_zraw)
            # A_decoded = stn(A_zraw, domA_img)
            #Reconstruction 2
            mask_loss = args.alpha1 * l1(A_decoded, domA_img) + args.alpha2 * l1(B_decoded, domB_img)
            loss += mask_loss




            if args.discweight > 0:
                preds_A = disc(A_common)
                preds_B = disc(B_common)
                loss += args.discweight * (bce(preds_A, B_label) + bce(preds_B, B_label))

            if args.output_disc_weight > 0:
                pred_c_out = disc_out(C_decoded)
                pred_A_out = disc_out(A_decoded)
                loss += args.output_disc_weight * (bce_ct(pred_c_out, A_label) + 0.7 * bce_ct(pred_A_out, A_label))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_params, 5)
            ae_optimizer.step()

            if args.discweight > 0:
                disc_optimizer.zero_grad()

                A_common = e1(domA_img)
                B_common = e1(domB_img)

                disc_A = disc(A_common)
                disc_B = disc(B_common)
                loss = bce(disc_A, A_label) + bce(disc_B, B_label)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 5)
                disc_optimizer.step()


            if args.output_disc_weight > 0:
                disc_out_optimizer.zero_grad()

                A_separate = e2(domA_img)
                B_common = e1(domB_img)
                C_encoding = torch.cat([B_common, A_separate], dim=1)
                C_zraw_mask= d_b(C_encoding, domB_img)

                C_decoded, _ = stn(C_zraw_mask, domB_img)


                # real_no_glasses_domain = disc_out(domB_img)
                real_with_glasses_domain = disc_out(domA_img)
                fake_content_transfer = disc_out(C_decoded)
                # bce_ct(real_no_glasses_domain, A_label)
                loss =  bce_ct(real_with_glasses_domain, A_label)\
                       + bce_ct(fake_content_transfer, B_label)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, 5)
                disc_out_optimizer.step()

            if _iter % args.progress_iter == 0:
                print('Outfile: %s <<>> Iteration %d' % (args.out, _iter))
                sys.stdout.flush()

            if _iter % args.display_iter == 0 and _iter > 0:
                e1 = e1.eval()
                e2 = e2.eval()
                d_a = d_a.eval()
                d_b = d_b.eval()
                stn = stn.eval()

                mask_utils.save_imgs(args, e1, e2, d_a, d_b, stn, _iter)

                e1 = e1.train()
                e2 = e2.train()
                d_a = d_a.train()
                d_b = d_b.train()
                stn=stn.train()

            if _iter % args.save_iter == 0 and _iter > 0:
                save_file = os.path.join(args.out, 'checkpoint')
                mask_utils.save_model(save_file, e1, e2, d_a, d_b, stn, ae_optimizer, disc, disc_optimizer, _iter)

            _iter += 1
            if args.print_runtime:
                t1 = time.time()
                print("\nTotal Elapsed time per batch: {}\n".format(t1 - t0))
    # t1= time.time()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--out', default='out')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--iters', type=int, default=10000)
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--discweight', type=float, default=0.005)
    parser.add_argument('--output_disc_weight', type=float, default=0.007)
    parser.add_argument('--disclr', type=float, default=0.0002)
    parser.add_argument('--progress_iter', type=int, default=2500)
    parser.add_argument('--display_iter', type=int, default=10000)
    parser.add_argument('--save_iter', type=int, default=10000)
    parser.add_argument('--load', default='')
    parser.add_argument('--num_display', type=int, default=10)
    parser.add_argument('--alpha1', type=float, default=0.7)
    parser.add_argument('--alpha2', type=float, default=0.7)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.001)
    parser.add_argument('--gama', type=float, default=7.0)
    parser.add_argument('--lambda1', type=float, default=0.2)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=5.0)
    parser.add_argument('--noise', type=float, default=0.0001)
    parser.add_argument('--stn_match_loss_const', type=float, default=5)
    parser.add_argument('--stn_match_loss_decay', type=float, default=0.9)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--print-runtime', type=int, default=0)

    args = parser.parse_args()


    train(args)

