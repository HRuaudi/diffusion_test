# 14_preprocess_h5.py - H5 데이터 전처리
# 1. 중복 케이스 제거 (FirstData+SBO 188개)
# 2. 입력의 1000 → 72 변환
import h5py
import numpy as np
from pathlib import Path

DATA_PATH = r"D:\Diffusion_test\data\raw\tspred_v2_new.h5"
OUTPUT_PATH = r"D:\Diffusion_test\data\processed\tspred_v2_cleaned.h5"

# 출력 폴더 생성
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("H5 데이터 전처리")
print("=" * 70)

# 1. 중복 케이스 찾기
print("\n[1] 중복 케이스 탐색 중...")

with h5py.File(DATA_PATH, 'r') as hf:
    scenarios = [k for k in hf.keys() if not k.startswith('_')]

    # FirstData와 SBO의 공통 케이스 중 입력이 같은 것 찾기
    fd_cases = set(hf['FirstData'].keys())
    sbo_cases = set(hf['SBO'].keys())
    common_ids = fd_cases & sbo_cases

    duplicate_cases = []  # (시나리오, 케이스ID) - 제거 대상

    for case_id in common_ids:
        x1 = hf['FirstData'][case_id]['input'][:]
        x2 = hf['SBO'][case_id]['input'][:]

        if np.allclose(x1, x2):
            # SBO에서 제거 (FirstData 유지)
            duplicate_cases.append(('SBO', case_id))

    # FirstData와 PSV 중복도 확인
    psv_cases = set(hf['PSV'].keys())
    common_fd_psv = fd_cases & psv_cases

    for case_id in common_fd_psv:
        x1 = hf['FirstData'][case_id]['input'][:]
        x2 = hf['PSV'][case_id]['input'][:]

        if np.allclose(x1, x2):
            duplicate_cases.append(('PSV', case_id))

print(f"   중복 케이스 (제거 대상): {len(duplicate_cases)}개")

# 제거 대상을 set으로 변환
remove_set = set(duplicate_cases)

# 2. 새 H5 파일 생성
print("\n[2] 전처리된 H5 파일 생성 중...")
print(f"   출력 경로: {OUTPUT_PATH}")

total_original = 0
total_new = 0
replaced_1000_count = 0

with h5py.File(DATA_PATH, 'r') as src, h5py.File(OUTPUT_PATH, 'w') as dst:

    for scenario in scenarios:
        print(f"\n   처리 중: {scenario}")

        case_ids = list(src[scenario].keys())
        total_original += len(case_ids)

        # 시나리오 그룹 생성
        scenario_grp = dst.create_group(scenario)

        copied = 0
        skipped = 0

        for case_id in case_ids:
            # 중복 케이스 제거
            if (scenario, case_id) in remove_set:
                skipped += 1
                continue

            # 입력 로드 및 1000 → 72 변환
            x = src[scenario][case_id]['input'][:]
            x_modified = x.copy()

            # 1000을 72로 변환
            mask_1000 = (x_modified == 1000)
            if mask_1000.any():
                replaced_1000_count += mask_1000.sum()
                x_modified[mask_1000] = 72

            # 출력 로드
            y = src[scenario][case_id]['output'][:]

            # 새 케이스 그룹 생성
            case_grp = scenario_grp.create_group(case_id)
            case_grp.create_dataset('input', data=x_modified)
            case_grp.create_dataset('output', data=y)

            copied += 1

        total_new += copied
        print(f"      원본: {len(case_ids)}, 복사: {copied}, 제거: {skipped}")

print("\n" + "=" * 70)
print("전처리 완료")
print("=" * 70)
print(f"\n원본 케이스 수: {total_original}")
print(f"전처리 후 케이스 수: {total_new}")
print(f"제거된 케이스: {total_original - total_new}")
print(f"1000→72 변환 횟수: {replaced_1000_count}")

# 3. 검증
print("\n[3] 검증 중...")

with h5py.File(OUTPUT_PATH, 'r') as hf:
    scenarios = [k for k in hf.keys() if not k.startswith('_')]

    total = 0
    has_1000 = 0

    for scenario in scenarios:
        for case_id in hf[scenario].keys():
            x = hf[scenario][case_id]['input'][:]
            total += 1
            if (x == 1000).any():
                has_1000 += 1

    print(f"   전처리 후 총 케이스: {total}")
    print(f"   1000 값 포함 케이스: {has_1000}")

    if has_1000 == 0:
        print("   ✓ 1000→72 변환 성공")
    else:
        print("   ✗ 아직 1000 값 존재")

print(f"\n새 파일 저장 완료: {OUTPUT_PATH}")
