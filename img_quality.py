import csv
from math import log10, sqrt
import cv2
import numpy as np
import scipy.stats
from IQA_pytorch import DISTS, utils, MS_SSIM
from PIL import Image
import torch
from SSIM_PIL import compare_ssim
import argparse
import imutils
import cv2
from scipy.stats import spearmanr
from piq import MultiScaleSSIMLoss, VIFLoss, FSIMLoss
import torchvision.transforms as transforms




def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def dists(ref, dist):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ref_path = 'r0.png'
    # dist_path = 'r1.png'
    #
    # ref = utils.prepare_image(Image.open(ref_path).convert("RGB")).to(device)
    # dist = utils.prepare_image(Image.open(dist_path).convert("RGB")).to(device)

    ref = utils.prepare_image(ref.convert("RGB")).to(device)
    dist = utils.prepare_image(dist.convert("RGB")).to(device)

    model = DISTS().to(device)

    score = model(dist, ref, as_loss=False)
    # print('score: %.4f' % score.item())

    return score.item()

def ms_ssim(ref, dist):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = utils.prepare_image(ref.convert("RGB")).to(device)
    dist = utils.prepare_image(dist.convert("RGB")).to(device)

    model = MS_SSIM().to(device)

    score = model(dist, ref, as_loss=False)
    # print('score: %.4f' % score.item())

    return score.item()

def msSsim(imageA, imageB):
    loss = MultiScaleSSIMLoss()
    output = loss(imageA, imageB)
    return output.backward()

def VifLoss(imageA, imageB):
    loss = VIFLoss()
    output = loss(imageA, imageB)
    return output.backward()

def FsimLoss(imageA, imageB):
    loss = FSIMLoss()
    output = loss(imageA, imageB)
    return output.backward()

def ssim(imageA, imageB):
    return compare_ssim(imageA, imageB)

def srcc(imageA, imageB):
    return spearmanr(imageA, imageB)

def minmax_normalize(x, min, max):
    return (x - min) / (max - min)

def pearsonr():
    csv_path = './kadid10k/dmos.csv'
    folder = './normalized2.csv'
    with open(csv_path, 'r',  newline='') as f:
        reader = csv.reader(f)
        next(reader)
        dmos = [i for i in reader]
        with open(folder, 'r', newline='') as f2:
            reader2 = csv.reader(f2)
            # first_norm = []
            # for item in reader2:
            #     first_norm.append(item)
            #     break
            # next(reader2)
            # first_norm = [i for i in next(reader2)]
            first_norm = next(reader2)
            norm = [i for i in reader2]
            # li1 = []
            # li2 = []
            # psnr = []
            # ssim = []
            # dist = []
            # for i in range(len(norm[0])):
            for i, label in enumerate(first_norm):
                li1 = []
                li2 = []
                for pair1, pair2 in zip(dmos, norm):
                    # print(len(pair2), i)
                    li1.append(float(pair1[2]))
                    li2.append(float(pair2[i]))

                # print(len(li1))
                # print(len(li2))
                a = scipy.stats.pearsonr(li1, li2)
                print(label, a)

        # a = scipy.stats.pearsonr(li1, li2)
        # print(a)



### 수요일 숙제###
#두개의 열을 골라서 교체하기, 각 열의 합을 구해서 곱해라 + 라이브
# matrix = [[1, 2, 3, 4],
#           [5, 6, 7, 8],
#           [9, 10, 11, 12]]

if __name__ == '__main__':
    pearsonr()
