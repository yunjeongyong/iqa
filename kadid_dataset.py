import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import csv
import os
import img_quality as q
import numpy as np

import iqa2
import iqas
from dmos_normalize import minmax_normalize
from IQA_pytorch import DISTS, MS_SSIM, CW_SSIM, FSIM, VSI, GMSD, NLPD, MAD, VIF, LPIPSvgg, SSIM


class KadidDataset:
    def __init__(self, csv_path='./kadid10k/dmos.csv'):
        self.data = []
        ref_dict = {}

        self.tf_toTensor = ToTensor()


        # csv_path = './kadid10k/dmos.csv'
        folder = './kadid10k/images/'
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for dist, ref, dmos, var in reader:
                ref = os.path.join('./kadid10k/ref/', ref)
                if ref not in ref_dict:
                    ref_pillow = Image.open(ref)
                    ref_tensor = torch.Tensor(self.tf_toTensor(ref_pillow))
                    d = torch.randn([2, 3, 384, 512])
                    ref_tensor1 = torch.cat((d, ref_tensor.unsqueeze(0)), dim=0)

                    ref_dict[ref] = (ref_pillow, np.array(ref_pillow), ref_tensor1)
                    # ref_dict[ref] = Image.open(ref)
                    # ref_dict[ref] = np.array(ref_dict[ref])
                self.data.append((os.path.join(folder, dist),
                                  ref_dict[ref],
                                  dmos,
                                  var))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        dist, ref_imgs, _, _ = self.data[idx]
        dist_pillow = Image.open(dist)
        dist_numpy = np.array(dist_pillow)
        dist_tensor = torch.Tensor(self.tf_toTensor(dist_pillow))
        d = torch.randn([2, 3, 384, 512])
        new_inps = torch.cat((d, dist_tensor.unsqueeze(0)),dim=0)
        return (dist_pillow, dist_numpy, new_inps), ref_imgs

    def iter(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)


if __name__ == '__main__':
    dataset = KadidDataset()

    # normalization 계산을 위해 계산 결과들이 저장될 배열
    psnr_array = []
    ssim_array = []
    dist_array = []
    # vif_array = []
    vsi_array = []
    nlpd_array = []

    # cw_ssim_array = []
    # mad_array = []
    # ms_ssim_array = []

    iqas_dict = {
        # IQA_NAME: (model, array, normalized, required, convert)
        "SSIM": (SSIM, [], [], True, "RGB"),
        "MS_SSIM": (MS_SSIM, [], [], True, "L"),
        "GMSD": (GMSD, [], [], True, "RGB"),
        "FSIM": (FSIM, [], [], True, "RGB"),
        # "VSI": (VSI, [], [], True, "RGB"),
        # "CW_SSIM": (CW_SSIM, [], [], True, "L"),
        # "NLPD": (NLPD, [], [], True, "L"),
        # "MAD": (MAD, [], [], True, "L"),
        # "VIF": (VIF, [], [], True, "L"),
        # "LPIPS": (LPIPSvgg, [], [], True, "RGB")
    }
    iqas_names = ["SSIM", "MS_SSIM", "GMSD", "FSIM"]

    # psnr numpy
    # ssim pillow
    # dist pillow

    for idx, ((dist_pillow, dist_numpy, new_inps), (ref_pillow, ref_numpy, ref_tensor1)) in enumerate(dataset.iter()):
        # print(q.PSNR(ref_numpy, dist_numpy))
        # print(q.ssim(ref_pillow, dist_pillow))
        # print(q.dists(ref_pillow, dist_pillow))

        print('%d / %d' % (idx, dataset.__len__()))
        psnr_array.append(q.PSNR(ref_numpy, dist_numpy))
        ssim_array.append(q.ssim(ref_pillow, dist_pillow))
        dist_array.append(q.dists(ref_pillow, dist_pillow))
        vsi_array.append(iqa2.vsi(ref_pillow, dist_pillow))
        nlpd_array.append(iqa2.nlpd(ref_pillow, dist_pillow))
        # cw_ssim_array.append(iqa2.cw_ssim(dist_pillow, ref_pillow))
        # mad_array.append(iqa2.mad(dist_pillow, ref_pillow))
        # ms_ssim_array.append(iqa2.ms_ssim(dist_pillow, ref_pillow))
        # vif_array.append(q.VIFLoss(ref_tensor1, new_inps))
        # print(q.VIFLoss(ref_tensor1, new_inps))

        for key in iqas_dict.keys():
            print(key)
            res = iqas.run2(ref_pillow, dist_pillow, iqas_dict[key][0], iqas_dict[key][3], iqas_dict[key][4])
            # print(key, res)
            iqas_dict[key][1].append(res)

        # print(q.ms_ssim(ref_pillow, dist_pillow))
        # if idx == 5:
        #     break

        # if idx == 10:
        #     break

    # 배열로부터 normalize 계산
    def normalized(array):
        _min = min(array)
        _max = max(array)
        return [minmax_normalize(x, _min, _max) for x in array]

    for key in iqas_dict.keys():
        iqas_dict[key][2].extend(normalized(iqas_dict[key][1]))

    # psnr_array = normalized(psnr_array)
    ssim_array = normalized(ssim_array)
    dist_array = normalized(dist_array)
    vsi_array = normalized(vsi_array)
    nlpd_array = normalized(nlpd_array)
    # cw_ssim_array = normalized(cw_ssim_array)
    # mad_array = normalized(mad_array)
    # ms_ssim_array = normalized(ms_ssim_array)

    # # 계산된 normalize로 csv 생성
    with open('./normalized3.csv', 'w+') as f:
        f.write('psnr,ssim,dist,vsi,nlpd,%s\n' % ','.join(iqas_names))
        # for psnr, ssim, dist, cw_ssim, mad, ms_ssim in zip(psnr_array, ssim_array, dist_array, cw_ssim_array, mad_array, ms_ssim_array):
        #     iqa_values = [psnr, ssim, dist, cw_ssim, mad, ms_ssim]
        #     values = ','.join([str(v) for v in iqa_values])
        #     f.write(values + '\n')
        for i in range(len(psnr_array)):
            iqa_values = [
                psnr_array[i],
                ssim_array[i],
                dist_array[i],
                vsi_array[i],
                nlpd_array[i],
            ]
            for name in iqas_names:
                iqa_values.append(iqas_dict[name][2][i])

            values = ','.join(map(str, iqa_values))
            f.write(values + '\n')
