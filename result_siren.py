import torch

# 체크포인트 로드
checkpoint = torch.load("logs_STRAINER/aircraft_siren_test_seed1234.pt", map_location=torch.device('cpu'))

# key 리스트
keys = [str(i).zfill(2) for i in range(1, 101)]

print(checkpoint)  # '01', '02', ..., '100' 이런 게 있어야 해요

psnr_list = []
worst_key = None
worst_psnr = float('inf')
best_key = None
best_psnr = float('-inf')

for key in keys:
    if key in checkpoint and 'psnr' in checkpoint[key]:
        psnr_vals = checkpoint[key]['psnr']
        if isinstance(psnr_vals, list) and len(psnr_vals) > 0:
            final_psnr = psnr_vals[-1]
            psnr_list.append((key, final_psnr))

            if final_psnr < worst_psnr:
                worst_psnr = final_psnr
                worst_key = key

            if final_psnr > best_psnr:
                best_psnr = final_psnr
                best_key = key

# ⚠️ ZeroDivisionError 방지
if len(psnr_list) == 0:
    print("❌ PSNR 값이 비어 있습니다. 체크포인트 경로나 내부 구조를 확인해주세요.")
else:
    avg_psnr = sum([p for _, p in psnr_list]) / len(psnr_list)
    delta_low = avg_psnr - worst_psnr
    delta_high = best_psnr - avg_psnr

    print(f"✅ 총 {len(psnr_list)}개 테스트 이미지의 결과 분석 완료")
    print(f"📉 최저 PSNR: {worst_psnr:.4f} (이미지: {worst_key})")
    print(f"📈 최고 PSNR: {best_psnr:.4f} (이미지: {best_key})")
    print(f"📊 평균 PSNR: {avg_psnr:.4f}")
    print(f"📏 평균 대비 최저 차이: -{delta_low:.4f}")
    print(f"📏 평균 대비 최고 차이: +{delta_high:.4f}")
