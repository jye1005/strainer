import numpy as np
import matplotlib.pyplot as plt

# .npy 파일 불러오기 (예: 이미지 배열)
img = np.load('23506.npy')

# 데이터 범위가 0~1이라면, 0~255 범위로 변환 후 uint8로 변환
if img.max() <= 1.0:
    img = (img * 255).astype(np.uint8)

# 이미지 저장 (JPG 형식)
plt.imsave('output.jpg', img)
