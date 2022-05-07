from IQA_pytorch import utils, DISTS, MS_SSIM
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(ref_pillow, dist_pillow, model, device_required: bool):
    ref = utils.prepare_image(ref_pillow.convert("RGB")).to(device)
    dist = utils.prepare_image(dist_pillow.convert("RGB")).to(device)
    model = model().to(device) if device_required else model()
    score = model(dist, ref, as_loss=False)
    return score.item()

def run2(ref_pillow, dist_pillow, model, device_required: bool, convert="RGB"):
    ref = utils.prepare_image(ref_pillow.convert(convert)).to(device)
    dist = utils.prepare_image(dist_pillow.convert(convert)).to(device)
    model = model().to(device) if device_required else model()
    score = model(dist, ref, as_loss=False)
    return score.item()
