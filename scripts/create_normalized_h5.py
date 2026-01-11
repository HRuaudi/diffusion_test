# create_normalized_h5.py - 정규화된 H5 파일 생성
# 학습 속도 향상을 위해 사전 정규화된 데이터 저장
import h5py
import numpy as np
from pathlib import Path
from normalize import normalize_input, normalize_output, NORM_CONST

INPUT_PATH = r"D:\Diffusion_test\data\processed\tspred_v2_cleaned.h5"
OUTPUT_PATH = r"D:\Diffusion_test\data\processed\tspred_v2_normalized.h5"

# 출력 폴더 확인
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("정규화된 H5 파일 생성")
print("=" * 70)
print(f"입력: {INPUT_PATH}")
print(f"출력: {OUTPUT_PATH}")

# 정규화 상수 저장용
print(f"\n정규화 상수:")
for k, v in NORM_CONST.items():
    print(f"  {k}: {v}")

total_cases = 0
with h5py.File(INPUT_PATH, 'r') as src, h5py.File(OUTPUT_PATH, 'w') as dst:
    scenarios = [k for k in src.keys() if not k.startswith('_')]

    # 정규화 상수를 메타데이터로 저장
    meta_grp = dst.create_group('_metadata')
    for k, v in NORM_CONST.items():
        meta_grp.attrs[k] = v

    for scenario in scenarios:
        print(f"\n처리 중: {scenario}")
        case_ids = list(src[scenario].keys())

        scenario_grp = dst.create_group(scenario)

        for i, case_id in enumerate(case_ids):
            # 원본 로드
            x = src[scenario][case_id]['input'][:]
            y = src[scenario][case_id]['output'][:]

            # 정규화
            x_norm = normalize_input(x)
            y_norm = normalize_output(y)

            # 저장
            case_grp = scenario_grp.create_group(case_id)
            case_grp.create_dataset('input', data=x_norm, dtype=np.float32)
            case_grp.create_dataset('output', data=y_norm, dtype=np.float32)

            total_cases += 1

            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(case_ids)} 완료")

        print(f"  {len(case_ids)}개 완료")

print("\n" + "=" * 70)
print("생성 완료")
print("=" * 70)
print(f"총 케이스: {total_cases}")
print(f"저장 위치: {OUTPUT_PATH}")

# 검증
print("\n검증 중...")
with h5py.File(OUTPUT_PATH, 'r') as hf:
    # 메타데이터 확인
    meta = hf['_metadata']
    print(f"메타데이터 키: {list(meta.attrs.keys())}")

    # 샘플 확인
    x = hf['FirstData']['1']['input'][:]
    y = hf['FirstData']['1']['output'][:]

    print(f"\n샘플 데이터:")
    print(f"  input shape: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  output shape: {y.shape}, range: [{y.min():.4f}, {y.max():.4f}]")

print("\n완료!")
