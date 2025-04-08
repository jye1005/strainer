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


# SineLayer 모델 정의
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

# SIREN 모델 정의
class SIREN(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, first_omega_0=30,
                 hidden_omega_0=30., pos_encode=False, no_init=False):
        super().__init__()
        self.pos_encode = pos_encode
        self.nonlin = SineLayer

        self.net = []
        if hidden_layers != 0:
            self.net.append(self.nonlin(in_features, hidden_features, is_first=True, omega_0=first_omega_0, init_weights=(not no_init)))
        
        hidden_layers = hidden_layers - 1 if (hidden_layers > 0 and outermost_linear is True) else hidden_layers

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, init_weights=(not no_init)))

        if outermost_linear or (hidden_layers == 0):
            final_linear = nn.Linear(hidden_features, out_features)
            self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
        return self.net(coords)

# STRAINER 모델 정의
class STRAINER(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., pos_encode=False, shared_encoder_layers=None,
                 num_decoders=None, no_init=False):
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

    # 추가: encoder weights 복사를 위한 메서드
    def load_encoder_weights_from(self, fellow_model):
        self.encoderINR.load_state_dict(deepcopy(fellow_model.encoderINR.state_dict()))


# fit_inr 함수 정의 (훈련을 위한 함수)
def fit_inr(coords, data, model, optim, config={}, mlogger=None, name=None):
    assert name is not None, "`name` must be provided as metric logger needs it"
    gt_tensor = data['gt']  # list or tensor

    tbar = tqdm(range(config['epochs']))
    psnr_vals = []

    for epoch in tbar:
        outputs = model(coords)  # list of 10 outputs: each [1, H*W, 3]
        stacked_outputs = torch.stack(outputs, dim=0)  # [10, 1, H*W, 3]

        # 🔧 여기만 수정
        if isinstance(gt_tensor, list):
            stacked_gt = torch.stack(gt_tensor, dim=0)
        else:
            stacked_gt = gt_tensor[None, ...]

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

    return {
        "psnr": psnr_vals,
        "state_dict": model.state_dict()
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

# get_train_data 및 get_test_data 정의 (데이터 로딩 함수들)
def get_train_data(path, zero_mean=True, sidelen=256, out_feature=3, take=10, device=torch.device('cuda'), seed=1234):
    files = sorted(glob.glob(os.path.join(path, "*")))
    sample = random.sample(range(len(files)), take)

    with open(f'logs_STRAINER/config_{seed}.txt', 'w') as f:
        f.write(f'Randomly selected {take} images from {len(files)} images: \n{sample}')
    print(f'Randomly selected {take} images from {len(files)} images: \n{sample}')        

    files = [files[i] for i in sample]
    images = []
    for fname in files:
        img = np.array(imageio.imread(fname), dtype=np.float32) / 255.
        img = np.expand_dims(img, axis=-1) if img.ndim == 2 else img  # Ensure 3D shape
        
        H, W, _ = img.shape
        aug_list = [ToTensor(), CenterCrop(min(H, W)), Resize((sidelen, sidelen))]
        if zero_mean:
            aug_list.append(Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])))

        transform = Compose(aug_list)
        img = transform(img).permute(1, 2, 0)
        images.append(img)

    return torch.stack(images).float().to(device)

def get_test_data(path, zero_mean=True, sidelen=256, out_feature=3, idx=0, device=torch.device('cuda')):
    files = sorted(glob.glob(os.path.join(path, "*")))
    img = np.array(imageio.imread(files[idx]), dtype=np.float32) / 255.
    img = np.expand_dims(img, axis=-1) if img.ndim == 2 else img  # Ensure 3D shape

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

# Main function for training and testing models
if __name__ == '__main__':
    seed = 1234
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

    TRAINING_PATH = "/local_datasets/div2k/train"
    TESTING_PATH = "/local_datasets/Urban100"

    coords = get_coords(*IMG_SIZE, device=device)

    im_tensor_train = get_train_data(TRAINING_PATH, take=10, device=device, seed=seed)
    data_dict_train_strainer = {'image_size': IMG_SIZE, 'gt': [x.reshape(1, -1, 3) for x in im_tensor_train]}
    
    inr_strainer_10decoders_train = STRAINER(
        in_features=config['in_channels'],
        hidden_features=config['hidden_features'],
        hidden_layers=config['num_layers'],
        shared_encoder_layers=config['shared_encoder_layers'],
        num_decoders=10,
        out_features=config['out_channels']
    ).to(device)
    
    optim_siren_strainer10decoder_train = torch.optim.Adam(lr=config['learning_rate'], params=inr_strainer_10decoders_train.parameters())
    
    print('\nTraining STRAINER 10 decoder\n')
    config_train = deepcopy(config)
    config_train['epochs'] = 5000
    ret_strainer10decoder_train = shared_encoder_training(
        coords=coords, data=data_dict_train_strainer, model=inr_strainer_10decoders_train,
        optim=optim_siren_strainer10decoder_train, config=config_train, mlogger=None, name="strainer_encoder_only_10decoder"
    )
    
    ret_strainer_10decoder_test = {}
    print('\nTesting STRAINER 10 decoder\n')
    
    for idx in range(100):
        im_tensors = get_test_data(TESTING_PATH, idx=idx)
        data_dict_test_strainer_10 = {'image_size': IMG_SIZE, 'gt': im_tensors[0].reshape(1, -1, 3)} 
        
        inr_strainer_test = STRAINER(
            in_features=config['in_channels'],
            hidden_features=config['hidden_features'],
            hidden_layers=config['num_layers'],
            shared_encoder_layers=config['shared_encoder_layers'],
            num_decoders=config['num_decoders'],
            out_features=config['out_channels']
        ).to(device)
        
        inr_strainer_test.load_encoder_weights_from(inr_strainer_10decoders_train)
        optim_siren_strainer_test = torch.optim.Adam(lr=config['learning_rate'], params=inr_strainer_test.parameters())
        
        ret_strainer_10decoder_test[str(idx+1).zfill(2)] = fit_inr(
            coords=coords, data=data_dict_test_strainer_10, model=inr_strainer_test,
            optim=optim_siren_strainer_test, config=config, mlogger=None, name=f"strainer_test_{idx}img"
        )
    
    torch.save(ret_strainer_10decoder_test, f'./logs_STRAINER/ret_strainer_10decoder_test_seed{seed}.pt')
    print(f"Successfully saved in /logs_STRAINER/ret_strainer_10decoder_test_seed{seed}.pt")
