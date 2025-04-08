import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# 저장된 결과 파일 경로
pt_path = './logs_STRAINER/diff_domain_10decoder_test_seed1234.pt'
save_img_dir = './logs_STRAINER/recon_images'
os.makedirs(save_img_dir, exist_ok=True)

# .pt 파일 불러오기 (CPU에서도 가능하도록)
ret = torch.load(pt_path, map_location='cpu')

# 테스트 결과 중 10개만 추출
for idx, key in enumerate(sorted(ret.keys())[:10]):
    output = ret[key]['output']  # 'output' 키에 복원된 결과 있다고 가정
    if isinstance(output, torch.Tensor):
        recon = output.reshape(256, 256, 3).cpu().detach().numpy()
        recon = (recon + 1) / 2  # [-1,1] -> [0,1]
        recon = np.clip(recon, 0, 1)
        plt.imsave(f'{save_img_dir}/recon_{key}.png', recon)
    else:
        print(f"[WARNING] No tensor found in key {key}")

print(f"\n복원 이미지 10장이 저장되었습니다: {save_img_dir}")
