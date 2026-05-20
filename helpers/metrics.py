import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import SpectralAngleMapper


ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

def ssim_fn(pred, target, data_range=1.0):
    global ssim_metric
    if getattr(ssim_metric, "data_range", None) != data_range:
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range)
    ssim_metric = ssim_metric.to(pred.device)   # 关键：每次都对齐 device（开销很小）
    score = ssim_metric(pred, target)
    return score, None

def psnr_fn(pred, target, max_val=1.0):
    '''
    pred, target: [batch_size, channels, width, height] (assumes image tensor with batch size, channel, width, height)
    max_val: The maximum possible pixel value of the image, default is 1.0
    '''
    mse = F.mse_loss(pred, target, reduction='mean')
    psnr = 10 * torch.log10(max_val**2 / mse)
    
    return psnr

def sam_fn(pred, target):
    '''
    pred, target: [c, w, h]
    '''
    pred, target = pred.squeeze(), target.squeeze()
    up = torch.sum((target*pred), dim = 0)   # [w, h]
    down1 = torch.sum((target**2), dim = 0).sqrt()
    down2 = torch.sum((pred**2), dim = 0).sqrt()

    map = torch.arccos(up / (down1 * down2))
    score = torch.mean(map[~torch.isnan(map)])
    map[torch.isnan(map)] = 0
    return score, map