import torch
from PIL import Image
from IQA_pytorch import utils, DISTS, MS_SSIM, CW_SSIM, MAD, VSI, NLPD
from IQA_pytorch.utils import prepare_image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def vsi(ref, dist):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image(ref.convert("RGB")).to(device)
    dist = prepare_image(dist.convert("RGB")).to(device)

    model = VSI().to(device)
    score = model(dist, ref, as_loss=False)
    return score.item()

def nlpd(ref, dist):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image(ref.convert("L")).to(device)
    dist = prepare_image(dist.convert("L")).to(device)

    model = NLPD(channels=1).to(device)

    score = model(dist, ref, as_loss=False)
    return score.item()
