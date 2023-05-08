import time
# import os
import gc
import torch
# import math
import albumentations as A
import numpy as np
import torch.nn.functional as F

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
# from torchsummary import summary
from skimage import color
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from Model import Generator
from Utils import load_checkpoint
from Dataset import ABDataset

gc.collect()
torch.cuda.empty_cache()


def PSNR_SSIM(orig_img, gen_img):
    data_range = max(np.amax(orig_img), np.amax(gen_img)) - min(np.amin(orig_img), np.amin(gen_img))
    v_ssim = structural_similarity(orig_img, gen_img, multichannel=True, data_range=data_range)

    orig_img = color.rgb2gray(orig_img)
    gen_img = color.rgb2gray(gen_img)

    data_range = max(np.amax(orig_img), np.amax(gen_img)) - min(np.amin(orig_img), np.amin(gen_img))
    v_psnr = peak_signal_noise_ratio(orig_img, gen_img, data_range=data_range)

    return round(v_psnr, 3), round(v_ssim, 3)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDER = f"results great"
IMG_SIZE = 1024
checkpoint = f"{FOLDER}/{IMG_SIZE}/gen.pth.tar"
TEST_DIR = "datasets/EyeQ dataset/test"

gen = Generator(patch_size=IMG_SIZE//64).to(DEVICE)
load_checkpoint(checkpoint, gen, None, None)
print(gen)

transforms = A.Compose(
    [
        A.Resize(height=2048, width=2048),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
)

dataset = ABDataset(root_a=TEST_DIR, transform=transforms)
loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
loop = tqdm(loader, leave=True)

psnr_values = []
ssim_values = []
start = time.time()

for idx, image in enumerate(loop):
    image = image.type(torch.float16).to(DEVICE)
    gen_hr_image = F.interpolate(image, size=IMG_SIZE, mode='bicubic').type(torch.float16).to(DEVICE)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            gen_hr_image = gen(gen_hr_image)

            gen_hr_image = gen_hr_image*0.5+0.5
            image = image*0.5+0.5

            save_image(gen_hr_image, f"{FOLDER}/{IMG_SIZE}/test/{idx}_fake.png")
            save_image(image, f"{FOLDER}/{IMG_SIZE}/test/{idx}_hr.png")

            image = image[0].permute(1, 2, 0).detach().cpu().numpy()
            gen_hr_image = gen_hr_image[0].permute(1, 2, 0).detach().cpu().numpy()

            psnr_values.append(PSNR_SSIM(image, gen_hr_image)[0])
            ssim_values.append(PSNR_SSIM(image, gen_hr_image)[1])

    loop.set_postfix(psnr=round(sum(psnr_values) / (idx+1), 3), ssim=round(sum(ssim_values) / (idx+1), 3))

end = time.time()
print(round((end - start)/1000, 3), "seconds")

metrics = [
    round(sum(psnr_values) / len(loader), 3),
    round(sum(ssim_values) / len(loader), 3),
    round((end - start)/len(loader), 3)
]

f = open(f"{FOLDER}/{IMG_SIZE}/test/results.txt", 'w')
f.write(f"Testing PSNR: {metrics[0]} dB\n")
f.write(f"Testing SSIM: {metrics[1]}\n")
f.write(f"Single image time: {metrics[2]} seconds\n")
f.close()
