# 전체 SIREN 실행 및 평가 코드 (MNIST-M 테스트 + 복원 이미지 저장)

import os
import torch
import torch.nn as nn
import numpy as np
import random
import glob
from tqdm import tqdm
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
import torchvision.utils as vutils
from collections import defaultdict

# --------------------------
# 모델 정의
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
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for _ in range(hidden_layers - 1):
            self.net.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))
        if outermost_linear:
            self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        B, N, _ = coords.shape
        out = self.net(coords.view(-1, 2))
        return out.view(B, N, -1)


# --------------------------
# 학습 함수
# --------------------------

def fit_siren(coords, data, model, optim, config, name):
    gt_tensor = data['gt']
    psnr_vals = []
    tbar = tqdm(range(config['epochs']))
    save_dir = os.path.join('logs_SIREN', 'reconstructed_images', name)
    os.makedirs(save_dir, exist_ok=True)
    final_recon_img = None

    for epoch in tbar:
        output = model(coords)
        loss = ((output - gt_tensor) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

        output_img = output / 2 + 0.5
        target_img = gt_tensor / 2 + 0.5
        mse = ((output_img - target_img) ** 2).mean()
        psnr = -10 * torch.log10(mse)
        psnr_vals.append(psnr.item())

        tbar.set_description(f"{name} | Epoch {epoch}/{config['epochs']} | Loss: {loss.item():.6f} | PSNR: {psnr:.4f}")

        if epoch == config['epochs'] - 1:
            recon = output_img[0].reshape(config['image_size'][0], config['image_size'][1], 3)
            recon = recon.permute(2, 0, 1).clamp(0.0, 1.0)
            final_recon_img = recon.detach().cpu()
            save_path = os.path.join(save_dir, f"final_recon.png")
            vutils.save_image(final_recon_img, save_path)

    return {"psnr": psnr_vals, "state_dict": model.state_dict(), "reconstructed_img": final_recon_img}


# --------------------------
# 데이터 로딩
# --------------------------

def get_test_data_from_mnistm(tensor_data, idx=0, sidelen=256, zero_mean=True, device='cuda'):
    img = tensor_data[idx]
    img = img.float() / 255.0
    if img.ndim == 3 and img.shape[0] == 3:
        pass
    elif img.ndim == 3 and img.shape[2] == 3:
        img = img.permute(2, 0, 1)
    aug_list = [Resize((sidelen, sidelen))]
    if zero_mean:
        aug_list.append(Normalize([0.5]*3, [0.5]*3))
    transform = Compose(aug_list)
    img = transform(img).permute(1, 2, 0)
    return torch.stack([img]).float().to(device)


def get_coords(H, W, device='cuda'):
    x = torch.linspace(-1, 1, W).to(device)
    y = torch.linspace(-1, 1, H).to(device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.stack((X, Y), dim=-1).reshape(-1, 2)
    return coords


# --------------------------
# 메인 실행
# --------------------------

if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = (256, 256)

    config = defaultdict()
    config['epochs'] = 2000
    config['learning_rate'] = 1e-4
    config['num_layers'] = 5
    config['hidden_features'] = 256
    config['in_channels'] = 2
    config['out_channels'] = 3
    config['image_size'] = IMG_SIZE

    coords = get_coords(*IMG_SIZE, device=device).unsqueeze(0)

    TESTING_PATH = "/local_datasets/MNIST-M/processed/mnist_m_test.pt"
    mnistm_dataset = torch.load(TESTING_PATH)
    mnistm_data = mnistm_dataset[0]

    ret_siren_test_all = {}

    for idx in range(100):
        im_tensor = get_test_data_from_mnistm(mnistm_data, idx=idx, sidelen=IMG_SIZE[0], device=device)[0].reshape(1, -1, 3)
        model = SIREN(
            in_features=config['in_channels'],
            hidden_features=config['hidden_features'],
            hidden_layers=config['num_layers'],
            out_features=config['out_channels']
        ).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        ret = fit_siren(coords, {"gt": im_tensor}, model, optim, config, name=f"siren_test_{idx}")
        ret_siren_test_all[str(idx+1).zfill(2)] = ret

    os.makedirs("logs_SIREN", exist_ok=True)
    torch.save(ret_siren_test_all, f"logs_SIREN/mnistm_siren_test_seed{seed}.pt")
    print("SIREN 테스트 완료 및 결과 저장")