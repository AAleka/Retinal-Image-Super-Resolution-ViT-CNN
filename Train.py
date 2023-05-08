import os
import torch
import torch.nn.functional as F
import albumentations as A
import numpy as np

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from torchgeometry.losses import SSIM
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from skimage import color
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from Utils import save_checkpoint, load_checkpoint
from Model import Generator
from Dataset import ABDataset

import warnings
import gc

warnings.filterwarnings("ignore")
torch.manual_seed(1)


def PSNR_SSIM(orig_img, gen_img):
    data_range = max(np.amax(orig_img), np.amax(gen_img)) - min(np.amin(orig_img), np.amin(gen_img))
    v_ssim = structural_similarity(orig_img, gen_img, multichannel=True, data_range=data_range)

    orig_img = color.rgb2gray(orig_img)
    gen_img = color.rgb2gray(gen_img)

    data_range = max(np.amax(orig_img), np.amax(gen_img)) - min(np.amin(orig_img), np.amin(gen_img))
    v_psnr = peak_signal_noise_ratio(orig_img, gen_img, data_range=data_range)

    return round(v_psnr, 3), round(v_ssim, 3)


def validate(val_loader, epoch):
    gc.collect()
    torch.cuda.empty_cache()

    loop = tqdm(val_loader, leave=True)

    psnr_values = []
    ssim_values = []

    if not os.path.exists(f"results/{IMG_SIZE}/validation/{epoch}"):
        os.makedirs(f"results/{IMG_SIZE}/validation/{epoch}")

    for idx, image in enumerate(loop):
        hr_image = image.type(torch.float16).to(DEVICE)
        gen_hr_image = F.interpolate(image, size=IMG_SIZE, mode='bicubic').type(torch.float16).to(DEVICE)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                gen_hr_image = gen_hr(gen_hr_image)

                gen_hr_image = gen_hr_image*0.5+0.5
                hr_image = hr_image*0.5+0.5

                save_image(gen_hr_image, f"results/{IMG_SIZE}/validation/{epoch}/{idx}_fake.png")
                save_image(hr_image, f"results/{IMG_SIZE}/validation/{epoch}/{idx}_hr.png")

                hr_image = hr_image[0].permute(1, 2, 0).detach().cpu().numpy()
                gen_hr_image = gen_hr_image[0].permute(1, 2, 0).detach().cpu().numpy()

                psnr_values.append(PSNR_SSIM(hr_image, gen_hr_image)[0])
                ssim_values.append(PSNR_SSIM(hr_image, gen_hr_image)[1])

    metrics = [
        round(sum(psnr_values) / len(val_loader), 3),
        round(sum(ssim_values) / len(val_loader), 3),
    ]

    return metrics


def train_fn(gen_hr, loader, opt_gen, l1, mse, ssim, g_scaler, epoch):
    gc.collect()
    torch.cuda.empty_cache()

    global count

    loss_G = 0

    loop = tqdm(loader, leave=True)

    for idx, image in enumerate(loop):
        hr_image = image.type(torch.float16).to(DEVICE)
        gen_hr_image = F.interpolate(image, size=IMG_SIZE, mode='bicubic').type(torch.float16).to(DEVICE)

        with torch.cuda.amp.autocast():
            gen_hr_image = gen_hr(gen_hr_image)
            loss_G_ssim = ssim(gen_hr_image, hr_image)
            loss_G_l1 = l1(gen_hr_image, hr_image)
            loss_G_mse = mse(gen_hr_image, hr_image)

            G_loss = (LAMBDA_SSIM*loss_G_ssim + LAMBDA_MSE*loss_G_mse + LAMBDA_L1*loss_G_l1)

            loss_G += G_loss.item()

        if LOAD_MODEL:
            opt_gen.param_groups[0]['capturable'] = True

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % (len(loader) // 20) == 0:
            save_image(gen_hr_image * 0.5 + 0.5, f"results/{IMG_SIZE}/images/{count}_fake.png")
            save_image(hr_image * 0.5 + 0.5, f"results/{IMG_SIZE}/images/{count}_real.png")
            count += 1

        loop.set_postfix(epoch=epoch, loss_G=loss_G / (idx + 1))


def main():
    best_gen = gen_hr

    transforms = A.Compose(
        [
            A.Resize(height=INPUT_IMG_SIZE, width=INPUT_IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ]
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=INPUT_IMG_SIZE, width=INPUT_IMG_SIZE),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ]
    )

    dataset = ABDataset(root_a=TRAIN_DIR, transform=transforms)
    val_dataset = ABDataset(root_a=VAL_DIR, transform=val_transforms)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    g_scaler = torch.cuda.amp.GradScaler()

    stop_count = best_psnr = best_ssim = 0

    for epoch in range(NUM_EPOCHS):
        train_fn(gen_hr, loader, opt_gen, l1, mse, ssim, g_scaler, epoch)
        avg_psnr, avg_ssim = validate(val_loader, epoch)

        f = open(f"results/{IMG_SIZE}/val_checkpoints.txt", 'a')
        f.write(f"[{epoch}: PSNR: {round(avg_psnr, 3)} SSIM: {round(avg_ssim, 3)}] ")

        if best_ssim < avg_ssim or (best_ssim == avg_ssim and best_psnr < avg_psnr):
            f.write(f"[{round(avg_psnr - best_psnr, 3)} {round(avg_ssim - best_ssim, 3)}] ")
            if best_psnr < avg_psnr:
                best_psnr = avg_psnr
            best_ssim = avg_ssim
            best_gen = gen_hr

            stop_count = 0

            if SAVE_MODEL:
                save_checkpoint(gen_hr, opt_gen, filename=f'{CHECKPOINT}/gen.pth.tar')

        else:
            stop_count += 1

        f.write("\n")
        f.close()

        if stop_count == STOP:
            if SAVE_MODEL:
                save_checkpoint(gen_hr, opt_gen, filename=f'{CHECKPOINT}/gen_final.pth.tar')

            break

    return best_gen


if __name__ == "__main__":
    STOP = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_DIR = "datasets/EyeQ dataset/train"
    VAL_DIR = "datasets/EyeQ dataset/validation"
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-5
    # LAMBDA_ADV = 10
    # LAMBDA_ENC = 10
    LAMBDA_L1 = 10
    LAMBDA_SSIM = 10
    LAMBDA_MSE = 10
    NUM_WORKERS = 1
    NUM_EPOCHS = 200
    SAVE_MODEL = True
    LOAD_MODEL = False
    INPUT_IMG_SIZE = 2048
    IMG_SIZES = (1024, 512, 256)
    count = 0
    LOAD_CHECKPOINT = "results great/1024"

    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    ssim = SSIM(window_size=3, reduction='mean').to(DEVICE)

    if not os.path.exists("results"):
        os.makedirs("results")

    for IMG_SIZE in IMG_SIZES:
        if IMG_SIZE == IMG_SIZES[0]:
            gen_hr = Generator(patch_size=IMG_SIZE//64).to(DEVICE)

            opt_gen = optim.Adam(
                gen_hr.parameters(),
                lr=LEARNING_RATE,
                betas=(0.5, 0.999),
            )

            if LOAD_MODEL:
                load_checkpoint(
                    f'{LOAD_CHECKPOINT}/gen.pth.tar', gen_hr, opt_gen, LEARNING_RATE,
                )

        else:
            gen_hr = Generator(patch_size=IMG_SIZE//64).to(DEVICE)

            gen_hr.TransformerEncoder = best_gen_hr.TransformerEncoder
            gen_hr.up_blocks = best_gen_hr.up_blocks
            gen_hr.last = best_gen_hr.last

            opt_gen = optim.Adam(
                gen_hr.parameters(),
                lr=LEARNING_RATE,
                betas=(0.5, 0.999),
            )

        print(f"Image size = {IMG_SIZE}")
        if not os.path.exists(f"results/{IMG_SIZE}"):
            os.makedirs(f"results/{IMG_SIZE}")
            os.makedirs(f"results/{IMG_SIZE}/images")
            os.makedirs(f"results/{IMG_SIZE}/validation")
            os.makedirs(f"results/{IMG_SIZE}/test")
            os.makedirs(f"results/test/{IMG_SIZE}/images")

            open(f"results/{IMG_SIZE}/val_checkpoints.txt", 'x').close()

        CHECKPOINT = f"results/{IMG_SIZE}"
        best_gen_hr = main()
