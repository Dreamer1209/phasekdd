"""PHASE: physiology-aware hyperspectral reconstruction via SSDA (NTIRE -> Hyper-Skin)
with a Mean-Teacher MST++ backbone plus PCR masking and PCA prototype alignment.
Objective: L = L_sup + lambda_un(t) * (L_con + lambda_pca * L_PCA)."""
import os
import math
import time
import random
import argparse
import itertools
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms

from helpers import utils, metrics
from helpers.pcr import sam_importance, ChannelWiseMaskGenerator
from helpers.pca import DualPrototypeBank, pca_loss, collect_teacher_features
from hsiData import HyperData
from hsiData.HyperSkinData import SkinLoad
from models.reconstruction import MST_Plus_Plus


# ============================================================
# 0) Reproducibility
# ============================================================
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.benchmark = True


# ============================================================
# 1) EMA teacher update
# ============================================================
@torch.no_grad()
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def sigmoid_rampup(current, rampup_length):
    """Mean-Teacher sigmoid ramp-up in [0,1]; 0 at start, ~1 after rampup_length."""
    if rampup_length <= 0:
        return 1.0
    phase = 1.0 - min(current, rampup_length) / rampup_length
    return float(math.exp(-5.0 * phase * phase))


# ============================================================
# 2) Target 3-/10-shot split (same protocol as the source impl.)
# ============================================================
def split_data_nshot(data_dir, val_ratio=0.2, num_shots=3, random_seed_dataset=42):
    random.seed(random_seed_dataset)
    np.random.seed(random_seed_dataset)
    all_files = os.listdir(data_dir)
    random.shuffle(all_files)
    val_size = int(val_ratio * len(all_files))
    val_files = all_files[:val_size]
    train_files = all_files[val_size:]
    labeled_files = train_files[:num_shots]
    unlabeled_files = train_files[num_shots:]
    return labeled_files, unlabeled_files, val_files


# ============================================================
# 3) Feature flattening helper
# ============================================================
def flatten_feat(feat):
    """[B, C', H', W'] -> [N, C']"""
    B, C, H, W = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(-1, C)


# ============================================================
# 4) Args
# ============================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='', required=True)
parser.add_argument('--camera_type', type=str, default='CIE')
parser.add_argument('--model_name', type=str, default='PHASE')
parser.add_argument('--saved_dir', type=str, default='saved-models')
parser.add_argument('--logged_dir', type=str, default='log')
parser.add_argument('--reconstructed_dir', type=str, default='reconstructed-hsi')
parser.add_argument('--external_dir', type=str, default=None)
parser.add_argument('--num_shot', type=int, default=10)
parser.add_argument('--max_iterations', type=int, default=10000)  # paper: 10k iters
parser.add_argument('--base_lr', type=float, default=1e-4)        # paper: 1e-4, cosine
parser.add_argument('--bs_source', type=int, default=8)           # paper batch sizes
parser.add_argument('--bs_unlabeled', type=int, default=8)
parser.add_argument('--bs_labeled', type=int, default=2)
parser.add_argument('--eval_every', type=int, default=600)

# Mean-Teacher
parser.add_argument('--ema_decay', type=float, default=0.99)

# PCA (Sec. 4.4) hyper-parameters
parser.add_argument('--num_target', type=int, default=16)   # K
parser.add_argument('--num_source', type=int, default=64)   # M
parser.add_argument('--mu_pro', type=float, default=0.9)    # B_T EMA momentum
parser.add_argument('--tau_p', type=float, default=0.2)     # prototype temperature
parser.add_argument('--tau_g', type=float, default=0.1)     # reliability gate temperature
parser.add_argument('--lambda_max', type=float, default=0.4)  # unsup ramp ceiling
parser.add_argument('--lambda_pca', type=float, default=0.3)  # PCA loss weight
parser.add_argument('--pca_start_iter', type=int, default=500)  # delay PCA after warm-up

# PCR (Sec. 4.3)
parser.add_argument('--mask_block', type=int, default=16)
parser.add_argument('--r_min', type=float, default=0.3)  # paper: r_min=0.3
parser.add_argument('--r_max', type=float, default=0.7)  # paper: r_max=0.7


if __name__ == '__main__':
    args = parser.parse_args()

    # ---------------- paths ----------------
    train_rgb_source = f'{args.data_dir}/HSI/HSI/NTIRE/rgb'
    train_hsi_source = f'{args.data_dir}/HSI/HSI/NTIRE/872hsi'
    rgb_target = f'{args.data_dir}/HSI/HSI/Hyper-Skin(RGB,VIS)/RGB_CIE'
    hsi_target = f'{args.data_dir}/HSI/HSI/Hyper-Skin(RGB,VIS)/VIS/'

    labeled_filelist, unlabeled_filelist, val_filelist = split_data_nshot(
        rgb_target, num_shots=args.num_shot, random_seed_dataset=42
    )
    print("Labeled:", len(labeled_filelist), labeled_filelist)
    print("Unlabeled:", len(unlabeled_filelist))
    print("Val:", len(val_filelist))

    # ---------------- folders & logger ----------------
    exp_logged_dir, exp_saved_dir, exp_reconstructed_dir = utils.create_folders_for(
        saved_dir=args.saved_dir,
        logged_dir=args.logged_dir,
        reconstructed_dir=args.reconstructed_dir,
        model_name=args.model_name,
        camera_type=args.camera_type,
        external_dir=args.external_dir,
    )
    logger = utils.initiate_logger(f'{exp_logged_dir}/train_phase', 'train')

    # ---------------- datasets ----------------
    do_aug, do_shift, do_shuffle, to_chw, load_img_type = True, False, True, True, 'rgb'
    crop_size = 256
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(crop_size, crop_size),
    ])

    train_dataset_source = HyperData.NTIRE2022(
        image_dir=train_rgb_source, spectral_dir=train_hsi_source,
        do_crop=False, do_aug=do_aug, do_shuffle=do_shuffle, do_shift=do_shift,
        transform=train_transform, to_chw=to_chw, load_img_type=load_img_type,
    )
    source_trainloader = torch.utils.data.DataLoader(
        train_dataset_source, batch_size=args.bs_source, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
    )

    train_dataset_labeled = SkinLoad(
        rgb_dir=rgb_target, hsi_dir=hsi_target, filelist=labeled_filelist,
        do_crop=False, do_aug=do_aug, do_shift=do_shift, transform=train_transform,
        to_chw=to_chw, load_img_type=load_img_type, unsupervised=False,
    )
    labeled_trainloader = torch.utils.data.DataLoader(
        train_dataset_labeled, batch_size=min(args.bs_labeled, args.num_shot), shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True,
    )

    train_dataset_unlabeled = SkinLoad(
        rgb_dir=rgb_target, hsi_dir=hsi_target, filelist=unlabeled_filelist,
        do_crop=False, do_aug=do_aug, do_shift=do_shift, transform=train_transform,
        to_chw=to_chw, load_img_type=load_img_type, unsupervised=True,
    )
    unlabeled_trainloader = torch.utils.data.DataLoader(
        train_dataset_unlabeled, batch_size=args.bs_unlabeled, shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True, drop_last=True,
    )

    valid_transform = transforms.Compose([transforms.ToTensor()])
    valid_dataset = SkinLoad(
        rgb_dir=rgb_target, hsi_dir=hsi_target, filelist=val_filelist,
        do_crop=False, do_aug=False, do_shift=False, transform=valid_transform,
        to_chw=True, load_img_type=load_img_type,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False,
    )

    # ---------------- model + EMA teacher ----------------
    model = MST_Plus_Plus(in_channels=3, out_channels=31, n_feat=31, stage=3).to(device)
    ema_model = MST_Plus_Plus(in_channels=3, out_channels=31, n_feat=31, stage=3).to(device)
    ema_model.load_state_dict(model.state_dict(), strict=True)
    for p in ema_model.parameters():
        p.detach_()

    # probe the intermediate feature dim C' (MST++ bottleneck; expected 124)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 64, 64, device=device)
        _, probe_feat = ema_model(dummy)
        feat_dim = probe_feat.shape[1]
    print(f"Intermediate feature dim C' = {feat_dim}")

    # ---------------- PCA dual prototype banks ----------------
    bank = DualPrototypeBank(
        feat_dim=feat_dim, num_target=args.num_target, num_source=args.num_source,
        mu=args.mu_pro, tau_p=args.tau_p, tau_g=args.tau_g,
    ).to(device)

    total_model_parameters = sum(p.numel() for p in model.parameters())
    msg = (
        f"[PHASE Experiment]\n"
        f"Model: {args.model_name} | params: {total_model_parameters}\n"
        f"feat_dim={feat_dim} K={args.num_target} M={args.num_source} "
        f"mu={args.mu_pro} tau_p={args.tau_p} tau_g={args.tau_g} "
        f"lambda_max={args.lambda_max} lambda_pca={args.lambda_pca}\n"
        f"Saved at: {exp_saved_dir} | Log at: {exp_logged_dir}\n"
        "=================================================================="
    )
    logger.info(msg)
    logger.info(f"random_seed: {random_seed}")
    print(msg)

    # ---------------- losses / optim ----------------
    criterion_l1 = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999))

    # ---------------- PCR masking ratios (Sec. 4.3) ----------------
    print('Estimating PCR spectral-density mask ratios...')
    mask_ratios = sam_importance(labeled_trainloader, r_min=args.r_min, r_max=args.r_max)
    mask_gen = ChannelWiseMaskGenerator(mask_ratios, block=args.mask_block).to(device)

    # ---------------- PCA bank initialization (k-means on EMA teacher feats) -------
    print('Initializing dual prototype banks via k-means on EMA-teacher features...')
    source_feats = collect_teacher_features(ema_model, source_trainloader, device)
    target_feats = collect_teacher_features(ema_model, labeled_trainloader, device)
    bank.initialize(source_feats, target_feats)
    print(f"Banks initialized: B_S={tuple(bank.B_S.shape)} (frozen), "
          f"B_T={tuple(bank.B_T.shape)} (online).")
    logger.info(f"Banks initialized: B_S={tuple(bank.B_S.shape)} B_T={tuple(bank.B_T.shape)}")
    del source_feats, target_feats

    # ---------------- iters (iteration-driven; paper trains 10k iters) ----------------
    labeled_iter = itertools.cycle(labeled_trainloader)
    source_iter = itertools.cycle(source_trainloader)
    unlabeled_iter = itertools.cycle(unlabeled_trainloader)
    max_iterations = args.max_iterations
    # Gaussian ramp-up reaches lambda_max at ~40% of training.
    rampup_len = max_iterations * 0.4
    iter_num = 0

    # ---------------- training ----------------
    if True:
        for _ in tqdm(range(max_iterations), desc='Iters', unit='it'):
            unlabeled_rgb = next(unlabeled_iter)
            labeled_rgb, labeled_hsi = next(labeled_iter)
            source_rgb, source_hsi = next(source_iter)

            labeled_rgb = labeled_rgb.to(device, non_blocking=True).float()
            labeled_hsi = labeled_hsi.to(device, non_blocking=True).float()
            source_rgb = source_rgb.to(device, non_blocking=True).float()
            source_hsi = source_hsi.to(device, non_blocking=True).float()
            unlabeled_rgb = unlabeled_rgb.to(device, non_blocking=True).float()

            # ---- teacher / student views (Mean Teacher) ----
            noise_student = torch.clamp(torch.randn_like(unlabeled_rgb) * 0.10, -0.1, 0.1)
            noise_teacher = torch.clamp(torch.randn_like(unlabeled_rgb) * 0.01, -0.1, 0.1)
            teacher_rgb = unlabeled_rgb + noise_teacher
            student_rgb = unlabeled_rgb + noise_student

            with torch.no_grad():
                ema_output, feat_tea = ema_model(teacher_rgb)   # [B,31,H,W], [B,C',H/4,W/4]

            # PCR strong perturbation on the student view
            student_rgb = mask_gen(student_rgb)

            # ---- single merged student forward: source + labeled + unlabeled ----
            model.train()
            bs, bl, bu = source_rgb.size(0), labeled_rgb.size(0), student_rgb.size(0)
            x_all = torch.cat([source_rgb, labeled_rgb, student_rgb], dim=0)
            out_all, feat_all = model(x_all)
            src_out = out_all[:bs]
            lbl_out = out_all[bs:bs + bl]
            unl_out = out_all[bs + bl:bs + bl + bu]
            feat_stu_unl = feat_all[bs + bl:bs + bl + bu]        # student feats on target

            # ---- losses ----
            lambda_un = args.lambda_max * sigmoid_rampup(iter_num, rampup_len)

            # (a) supervised reconstruction (source + target labeled)
            src_l1 = criterion_l1(src_out, source_hsi)
            tgt_l1 = criterion_l1(lbl_out, labeled_hsi)
            sup_loss = src_l1 + tgt_l1

            # (b) consistency on unlabeled target
            con_loss = criterion_l1(unl_out, ema_output.detach())

            # (c) physiologically constrained alignment (delayed start)
            if bank.initialized and iter_num >= args.pca_start_iter:
                pca_l, gate_mean = pca_loss(feat_stu_unl, feat_tea.detach(), bank)
            else:
                pca_l = torch.zeros((), device=device)
                gate_mean = torch.zeros((), device=device)

            total_loss = sup_loss + lambda_un * (con_loss + args.lambda_pca * pca_l)

            # ---- backward & step ----
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # ---- EMA teacher + online B_T update ----
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            if bank.initialized and iter_num >= args.pca_start_iter:
                with torch.no_grad():
                    bank.update_target_bank(flatten_feat(feat_tea))

            # ---- lr schedule (cosine annealing; paper) ----
            iter_num += 1
            lr_ = max(0.5 * args.base_lr * (1.0 + math.cos(math.pi * iter_num / max_iterations)), 1e-8)
            optimizer.param_groups[0]['lr'] = lr_

            # ---- logging ----
            if iter_num % 50 == 0:
                print(f"[I{iter_num}/{max_iterations}] lr {lr_:.6f} "
                      f"loss {total_loss.item():.4f} sup {sup_loss.item():.4f} "
                      f"con {con_loss.item():.4f} pca {float(pca_l):.4f} "
                      f"gate {float(gate_mean):.3f} lam_un {lambda_un:.3f}")
                logger.info(f"iter {iter_num} | loss {total_loss.item():.4f} | "
                            f"sup {sup_loss.item():.4f} | con {con_loss.item():.4f} | "
                            f"pca {float(pca_l):.4f} | gate {float(gate_mean):.3f} | "
                            f"lam_un {lambda_un:.3f} | lr {lr_:.6f}")

            # ---- evaluation ----
            if iter_num % args.eval_every == 0 or iter_num == max_iterations:
                torch.cuda.empty_cache()
                if device == 'cuda':
                    torch.cuda.synchronize()
                print("evaluating on validation set...")
                eval_start = time.time()
                results = {"ssim_score": [], "sam_score": [], "psnr_score": []}
                model.eval()
                with torch.no_grad():
                    for _, (x, y) in enumerate(tqdm(valid_loader, desc="Evaluating")):
                        x = x.float().to(device)
                        y = y.float().to(device)
                        pred, _ = model(x)
                        ssim_score, _ = metrics.ssim_fn(pred, y)
                        sam_score, _ = metrics.sam_fn(pred, y)
                        psnr_score = metrics.psnr_fn(pred=pred, target=y, max_val=1.0)
                        results["ssim_score"].append(ssim_score.cpu().numpy())
                        results["sam_score"].append(sam_score.cpu().numpy())
                        results["psnr_score"].append(psnr_score.cpu().numpy())
                torch.cuda.empty_cache()

                ssim_mean = float(np.array(results["ssim_score"]).mean())
                ssim_std = float(np.array(results["ssim_score"]).std())
                sam_mean = float(np.array(results["sam_score"]).mean())
                sam_std = float(np.array(results["sam_score"]).std())
                psnr_mean = float(np.array(results["psnr_score"]).mean())
                psnr_std = float(np.array(results["psnr_score"]).std())

                eval_time = time.time() - eval_start
                print(f"Eval {eval_time:.2f}s | SSIM {ssim_mean:.4f}±{ssim_std:.4f} | "
                      f"SAM {sam_mean:.4f}±{sam_std:.4f} | PSNR {psnr_mean:.4f}±{psnr_std:.4f}")
                logger.info(f"Iter {iter_num} (eval {eval_time:.2f}s) | "
                            f"SSIM {ssim_mean:.4f}±{ssim_std:.4f} | "
                            f"SAM {sam_mean:.4f}±{sam_std:.4f} | "
                            f"PSNR {psnr_mean:.4f}±{psnr_std:.4f}")

                model_saved_path = (f'{exp_saved_dir}/model_{args.num_shot}shot_'
                                    f'ssim{ssim_mean:.4f}_iter{iter_num}.pt')
                torch.save(model.state_dict(), model_saved_path)
                print(f"Model saved to {model_saved_path}")
                model.train()
