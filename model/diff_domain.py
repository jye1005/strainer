import os
import torch
import torch.nn as nn
import numpy as np
import random
import imageio
import glob
from tqdm import tqdm
from torchvision.transforms import Resize, ToTensor, CenterCrop, Normalize, Compose
import math
from collections import defaultdict
from copy import deepcopy

# --------------------------
# Model 정의
# --------------------------

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, init_weights=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if init_weights:
            self.init_weights()
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SIREN(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30., pos_encode=False, no_init=False):
        super().__init__()
        self.pos_encode = pos_encode
        self.nonlin = SineLayer
        self.net = []
        if hidden_layers != 0:
            self.net.append(self.nonlin(in_features, hidden_features,
                                          is_first=True, omega_0=first_omega_0, init_weights=(not no_init)))
        hidden_layers = hidden_layers - 1 if (hidden_layers > 0 and outermost_linear is True) else hidden_layers
        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features,
                                          is_first=False, omega_0=hidden_omega_0, init_weights=(not no_init)))
        if outermost_linear or (hidden_layers == 0):
            final_linear = nn.Linear(hidden_features, out_features)
            self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
        return self.net(coords)


class STRAINER(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30., pos_encode=False,
                 shared_encoder_layers=None, num_decoders=None, no_init=False):
        super().__init__()
        self.shared_encoder_layers = shared_encoder_layers
        self.num_decoders = num_decoders
        self.encoderINR = SIREN(
            in_features=in_features, hidden_features=hidden_features,
            hidden_layers=self.shared_encoder_layers - 1,
            out_features=hidden_features, outermost_linear=False,
            first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0,
            pos_encode=pos_encode, no_init=no_init
        )
        self.num_decoder_layers = hidden_layers - self.shared_encoder_layers
        self.decoderINRs = nn.ModuleList([
            SIREN(
                in_features=hidden_features, hidden_features=hidden_features,
                hidden_layers=self.num_decoder_layers - 1, out_features=out_features,
                outermost_linear=outermost_linear, first_omega_0=first_omega_0,
                hidden_omega_0=hidden_omega_0, pos_encode=pos_encode, no_init=no_init
            ) for _ in range(self.num_decoders)
        ])
    def forward(self, coords):
        encoded_features = self.encoderINR(coords)
        outputs = [decoder(encoded_features) for decoder in self.decoderINRs]
        return outputs
    # encoder weight 복사 메서드
    def load_encoder_weights_from(self, fellow_model):
        self.encoderINR.load_state_dict(deepcopy(fellow_model.encoderINR.state_dict()))


# --------------------------
# Training/Testing 함수 정의
# --------------------------

import os
import torchvision.utils as vutils
from tqdm import tqdm
import torch

def fit_inr(coords, data, model, optim, config={}, mlogger=None, name=None):
    assert name is not None, "`name` must be provided as metric logger needs it"
    gt_tensor = data['gt']  # list or tensor

    tbar = tqdm(range(config['epochs']))
    psnr_vals = []

    # 이미지 저장 폴더 생성
    save_dir = os.path.join('logs_STRAINER', 'reconstructed_images_same_mnist', name)
    os.makedirs(save_dir, exist_ok=True)

    final_recon_img = None  # 마지막 복원 이미지를 저장할 변수

    for epoch in tbar:
        outputs = model(coords)  # list of outputs: each [1, H*W, 3]
        stacked_outputs = torch.stack(outputs, dim=0)  # [N, 1, H*W, 3]

        # Ground truth 정리
        if isinstance(gt_tensor, list):
            stacked_gt = torch.stack(gt_tensor, dim=0)
        else:
            stacked_gt = gt_tensor[None, ...]

        # Loss 계산 및 backward
        loss = ((stacked_outputs - stacked_gt) ** 2).mean(dim=[1, 2, 3]).sum()
        optim.zero_grad()
        loss.backward()
        optim.step()

        # PSNR 계산
        stacked_outs = stacked_outputs / 2 + 0.5
        stacked_gts = stacked_gt / 2 + 0.5
        mse_total = ((stacked_outs - stacked_gts) ** 2).mean(dim=[1, 2, 3]).sum()
        psnr = -10 * torch.log10(mse_total)
        psnr_vals.append(float(psnr))

        tbar.set_description(f"Iter {epoch}/{config['epochs']} Loss = {loss.item():.6f} PSNR = {psnr:.4f}")
        tbar.refresh()

        if epoch == config['epochs'] - 1:
            recon = stacked_outs[0].reshape(config['image_size'][0], config['image_size'][1], 3)
            recon = recon.permute(2, 0, 1).clamp(0.0, 1.0)
            save_path = os.path.join(save_dir, f'final_reconstruction.png')
            vutils.save_image(recon, save_path)

    # 마지막 이미지 업데이트
    final_recon_img = recon.detach().cpu()
    return {
        "psnr": psnr_vals,
        "state_dict": model.state_dict(),
        "reconstructed_img": final_recon_img  # 마지막 복원 이미지 포함
    }


# shared_encoder_training 함수 정의 (공유된 인코더 훈련을 위한 함수)
def shared_encoder_training(coords, data, model, optim, config={}, mlogger=None, name=None):
    assert name is not None, "`name` must be provided as metric logger needs it"
    gt_tensor = data['gt']

    psnr_vals = []
    tbar = tqdm(range(config['epochs']))

    for epoch in tbar:
        outputs = model(coords) 
        stacked_outputs = torch.stack(outputs, dim=0)
        stacked_gt = torch.stack(gt_tensor, dim=0)
        loss = ((stacked_outputs - stacked_gt)**2).mean(dim=[1,2,3]).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        tbar.set_description(f"Iter {epoch}/{config['epochs']} Loss = {loss.item():6f}")
        tbar.refresh()

    return {
        "psnr" : psnr_vals,
        "state_dict" : model.state_dict()
    }



# --------------------------
# 데이터 로딩 함수들
# --------------------------

def get_train_data(path, zero_mean=True, sidelen=256, out_feature=3, take=10, device=torch.device('cuda'), seed=1234):
    files = sorted(glob.glob(os.path.join(path, "*.png")))
    print(f"[DEBUG] Found {len(files)} training files in {path}")
    if take > len(files):
        print(f"[WARNING] Requested take={take} is larger than available files ({len(files)}); adjusting take.")
        take = len(files)
    sample = random.sample(range(len(files)), take)
    os.makedirs('logs_STRAINER', exist_ok=True)
    with open(f'logs_STRAINER/config_{seed}.txt', 'w') as f:
        f.write(f'Randomly selected {take} images from {len(files)} images: \n{sample}')
    print(f"Randomly selected {take} images from {len(files)} images: \n{sample}")
    files = [files[i] for i in sample]
    images = []
    for fname in files:
        img = np.array(imageio.imread(fname), dtype=np.float32) / 255.
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        H, W, _ = img.shape
        aug_list = [ToTensor(), CenterCrop(min(H, W)), Resize((sidelen, sidelen))]
        if zero_mean:
            aug_list.append(Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])))
        transform = Compose(aug_list)
        img = transform(img).permute(1, 2, 0)
        images.append(img)
    return torch.stack(images).float().to(device)


def get_test_data(path, zero_mean=True, sidelen=256, out_feature=3, idx=0, device=torch.device('cuda')):
    files = sorted(glob.glob(os.path.join(path, "*.jpg")))
    print(f"[DEBUG] Found {len(files)} testing files in {path}")
    if idx >= len(files):
        raise ValueError(f"Index {idx} is out of range for {len(files)} files.")
    img = np.array(imageio.imread(files[idx]), dtype=np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    H, W, _ = img.shape
    aug_list = [ToTensor(), CenterCrop(min(H, W)), Resize((sidelen, sidelen))]
    if zero_mean:
        aug_list.append(Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])))
    transform = Compose(aug_list)
    img = transform(img).permute(1, 2, 0)
    return torch.stack([img]).float().to(device)


def get_coords(H, W, T=None, device=torch.device('cuda')):
    x = torch.linspace(-1, 1, W).to(device)
    y = torch.linspace(-1, 1, H).to(device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
    return coords

def get_train_data_from_mnistm(tensor_data, zero_mean=True, sidelen=256, idx=0, device=torch.device('cuda')):
    img = tensor_data[idx]  # [3, 28, 28]
    img = img.float() / 255.0  # Normalize to [0, 1]

    # Ensure the tensor is in (C, H, W) format for torchvision
    if img.ndim == 3 and img.shape[0] == 3:
        pass  # already in correct format
    elif img.ndim == 3 and img.shape[2] == 3:
        img = img.permute(2, 0, 1)  # convert from HWC to CHW if needed

    aug_list = [Resize((sidelen, sidelen))]
    if zero_mean:
        aug_list.append(Normalize(torch.Tensor([0.5]*3), torch.Tensor([0.5]*3)))
    transform = Compose(aug_list)

    img = transform(img)  # shape: [3, H, W]
    img = img.permute(1, 2, 0)  # convert to (H, W, C)
    return torch.stack([img]).float().to(device)


def get_test_data_from_mnistm(tensor_data, zero_mean=True, sidelen=256, idx=0, device=torch.device('cuda')):
    img = tensor_data[idx]  # [3, 28, 28]
    img = img.float() / 255.0  # Normalize to [0, 1]

    # Ensure the tensor is in (C, H, W) format for torchvision
    if img.ndim == 3 and img.shape[0] == 3:
        pass  # already in correct format
    elif img.ndim == 3 and img.shape[2] == 3:
        img = img.permute(2, 0, 1)  # convert from HWC to CHW if needed

    aug_list = [Resize((sidelen, sidelen))]
    if zero_mean:
        aug_list.append(Normalize(torch.Tensor([0.5]*3), torch.Tensor([0.5]*3)))
    transform = Compose(aug_list)

    img = transform(img)  # shape: [3, H, W]
    img = img.permute(1, 2, 0)  # convert to (H, W, C)
    return torch.stack([img]).float().to(device)


# --------------------------
# Main: 학습 및 테스트
# --------------------------
if __name__ == '__main__':
    seed = 3333
    torch.manual_seed(seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    IMG_SIZE = (256, 256)
    POS_ENCODE = False

    config = defaultdict()
    config['epochs'] = 2000
    config['learning_rate'] = 1e-4
    config['plot_every'] = 100
    config['image_size'] = IMG_SIZE
    config['num_layers'] = 5
    config['hidden_features'] = 256
    config['in_channels'] = 2
    config['out_channels'] = 3
    config['shared_encoder_layers'] = 4
    config['num_decoders'] = 1
    config['nonlin'] = 'siren'

    coords = get_coords(*IMG_SIZE, device=device)

    # ---------- 학습: MNIST-M 사용 ----------
    TRAINING_PATH = "/local_datasets/MNIST-M/processed/mnist_m_train.pt"
    mnistm_dataset = torch.load(TRAINING_PATH)
    mnistm_data = mnistm_dataset[0]

# MNIST-M 이미지 10장 로딩 (각각 [3, 28, 28] → [256, 256, 3] → [1, H*W, 3])
    im_tensor_train = [get_train_data_from_mnistm(mnistm_data, idx=i, sidelen=IMG_SIZE[0], device=device)[0] for i in range(10)]
    data_dict_train_strainer = {
    'image_size': IMG_SIZE,
    'gt': [x.reshape(1, -1, 3) for x in im_tensor_train]
    }

    inr_strainer_10decoders_train = STRAINER(
        in_features=config['in_channels'],
        hidden_features=config['hidden_features'],
        hidden_layers=config['num_layers'],
        shared_encoder_layers=config['shared_encoder_layers'],
        num_decoders=10,
        out_features=config['out_channels']
    ).to(device)

    optim_siren_strainer10decoder_train = torch.optim.Adam(
        lr=config['learning_rate'],
        params=inr_strainer_10decoders_train.parameters())

    print('\nTraining STRAINER 10 decoder\n')
    config_train = deepcopy(config)
    config_train['epochs'] = 5000
    ret_strainer10decoder_train = shared_encoder_training(
        coords=coords, data=data_dict_train_strainer,
        model=inr_strainer_10decoders_train,
        optim=optim_siren_strainer10decoder_train,
        config=config_train, mlogger=None, name="strainer_encoder_only_10decoder")

    # ---------- 테스트: mnist-m 사용 ----------
    TESTING_PATH = "/local_datasets/MNIST-M/processed/mnist_m_test.pt"
    mnistm_dataset = torch.load(TESTING_PATH)
    mnistm_data = mnistm_dataset[0]
    ret_strainer_10decoder_test = {}

    print('\nTesting STRAINER on MNIST-M\n')
    for idx in range(100):
        im_tensors = get_test_data_from_mnistm(mnistm_data, idx=idx, device=device)
        data_dict_test_strainer = {'image_size': IMG_SIZE, 'gt': im_tensors[0].reshape(1, -1, 3)}

        inr_strainer_test = STRAINER(
            in_features=config['in_channels'],
            hidden_features=config['hidden_features'],
            hidden_layers=config['num_layers'],
            shared_encoder_layers=config['shared_encoder_layers'],
            num_decoders=1,
            out_features=config['out_channels']
        ).to(device)

        inr_strainer_test.load_encoder_weights_from(inr_strainer_10decoders_train)

        optim_siren_strainer_test = torch.optim.Adam(
            lr=config['learning_rate'],
            params=inr_strainer_test.parameters())

        ret_strainer_10decoder_test[str(idx+1).zfill(2)] = fit_inr(
            coords=coords,
            data=data_dict_test_strainer,
            model=inr_strainer_test,
            optim=optim_siren_strainer_test,
            config=config, mlogger=None, name=f"strainer_test_{idx}img")

    torch.save(ret_strainer_10decoder_test, f'./logs_STRAINER/mnistm_test_mnistm_seed{seed}.pt')
    print(f"Successfully saved in ./logs_STRAINER/mnistm_test_mnistm_seed{seed}.pt")

