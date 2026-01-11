# normalize.py - 물리적 의미 유지 정규화 함수
# 실제 데이터 기반 물리량별 통일 정규화
import numpy as np

# =============================================================================
# 정규화 상수 (실제 데이터에서 산출)
# =============================================================================
NORM_CONST = {
    # 입력 정규화
    'time_max': 259200.0,       # 72시간 (초)
    'seal_area_max': 5.0,       # 누설 면적 max (cm²)

    # 출력 정규화 - 물리량별 통일
    'temp_min': 235.0,          # 온도 통일 min (°C)
    'temp_max': 3755.0,         # 온도 통일 max (°C)
    'level_max': 13.55,         # 수위 통일 max (m)
    'flow_max': 100.3,          # 유량 통일 max (kg/s)
    'pressure_max': 19304118.0, # 압력 max (Pa)
    'zwdc_max': 15.02,          # ZWDC2SG max
}

# 변수 인덱스 매핑
INPUT_NAMES = [
    'AC1_ON', 'AC1_OFF', 'AC2_ON', 'AC2_OFF', 'AC3_ON', 'AC3_OFF', 'AC4_ON', 'AC4_OFF',
    'SI1_ON', 'SI1_OFF', 'SI2_ON', 'SI2_OFF', 'FW1_ON', 'FW1_OFF', 'FW2_ON', 'FW2_OFF',
    'PSV_STUCK', 'SDS', 'CSS', 'RECIRC_ON', 'RECIRC_OFF',
    'SEAL_TIME', 'SEAL_AREA', 'CASE_NUM'
]

OUTPUT_NAMES = ['zw_vessel', 'TCLAD_HOT', 'PPS', 'TWCR', 'ZWDC2SG', 'TWSG', 'HPSI', 'RWST', 'WAFWSG']


# =============================================================================
# 입력 정규화
# =============================================================================
def normalize_input(x):
    """
    입력 24D → 47D (23값 + 24마스크, 케이스번호 제외)

    Parameters
    ----------
    x : np.ndarray, shape (24,)
        원본 입력 벡터

    Returns
    -------
    x_norm : np.ndarray, shape (47,)
        정규화된 입력 (23 값 + 24 마스크)
    """
    C = NORM_CONST
    x_norm = np.zeros(47, dtype=np.float32)
    x_mask = np.zeros(24, dtype=np.float32)

    # 시간 변수 (0-15, 19-21): ÷259200
    time_dims = list(range(16)) + [19, 20, 21]
    for i in time_dims:
        if x[i] == -1:
            x_norm[i] = 0.0
            x_mask[i] = 1.0
        else:
            x_norm[i] = x[i] / C['time_max']

    # 플래그 변수 (16-18): 그대로
    for i in [16, 17, 18]:
        x_norm[i] = float(x[i])

    # SEAL_AREA (22): ÷5
    if x[22] == -1:
        x_norm[22] = 0.0
        x_mask[22] = 1.0
    else:
        x_norm[22] = x[22] / C['seal_area_max']

    # 마스크 추가 (케이스번호 23도 마스크에 포함)
    x_norm[23:47] = x_mask

    return x_norm  # (47,)


def normalize_input_batch(X):
    """
    배치 입력 정규화

    Parameters
    ----------
    X : np.ndarray, shape (N, 24)
        배치 입력

    Returns
    -------
    X_norm : np.ndarray, shape (N, 47)
        정규화된 배치 입력
    """
    N = X.shape[0]
    X_norm = np.zeros((N, 47), dtype=np.float32)
    for i in range(N):
        X_norm[i] = normalize_input(X[i])
    return X_norm


# =============================================================================
# 출력 정규화
# =============================================================================
def normalize_output(y):
    """
    출력 (T, 9) 정규화 - 실제 데이터 기반 물리량별 통일

    Parameters
    ----------
    y : np.ndarray, shape (T, 9)
        원본 출력 시계열

    Returns
    -------
    y_norm : np.ndarray, shape (T, 9)
        정규화된 출력 시계열
    """
    C = NORM_CONST
    y_norm = np.zeros_like(y, dtype=np.float32)

    # 수위 (통일: [0, 13.55]m → [0, 1])
    y_norm[:, 0] = y[:, 0] / C['level_max']   # zw_vessel
    y_norm[:, 7] = y[:, 7] / C['level_max']   # RWST

    # 온도 (통일: [235, 3755]°C → [0, 1])
    temp_range = C['temp_max'] - C['temp_min']
    y_norm[:, 1] = (y[:, 1] - C['temp_min']) / temp_range  # TCLAD_HOT
    y_norm[:, 3] = (y[:, 3] - C['temp_min']) / temp_range  # TWCR
    y_norm[:, 5] = (y[:, 5] - C['temp_min']) / temp_range  # TWSG

    # 압력 (Log 스케일)
    y_norm[:, 2] = np.log10(np.maximum(y[:, 2], 1.0)) / np.log10(C['pressure_max'])  # PPS

    # 유량 (통일: [0, 100.3]kg/s → [0, 1])
    y_norm[:, 6] = y[:, 6] / C['flow_max']   # HPSI
    y_norm[:, 8] = y[:, 8] / C['flow_max']   # WAFWSG

    # 기타
    y_norm[:, 4] = y[:, 4] / C['zwdc_max']   # ZWDC2SG

    return y_norm


def normalize_output_batch(Y):
    """
    배치 출력 정규화

    Parameters
    ----------
    Y : list of np.ndarray, each shape (T_i, 9)
        가변 길이 시계열 배치

    Returns
    -------
    Y_norm : list of np.ndarray, each shape (T_i, 9)
        정규화된 배치 출력
    """
    return [normalize_output(y) for y in Y]


# =============================================================================
# 역정규화 (생성 후 복원)
# =============================================================================
def denormalize_output(y_norm):
    """
    정규화된 출력을 물리적 값으로 복원

    Parameters
    ----------
    y_norm : np.ndarray, shape (T, 9)
        정규화된 출력 시계열

    Returns
    -------
    y : np.ndarray, shape (T, 9)
        원본 스케일 출력 시계열
    """
    C = NORM_CONST
    y = np.zeros_like(y_norm, dtype=np.float32)

    # 수위 (통일 범위에서 복원)
    y[:, 0] = y_norm[:, 0] * C['level_max']   # zw_vessel
    y[:, 7] = y_norm[:, 7] * C['level_max']   # RWST

    # 온도 (통일 범위에서 복원)
    temp_range = C['temp_max'] - C['temp_min']
    y[:, 1] = y_norm[:, 1] * temp_range + C['temp_min']  # TCLAD_HOT
    y[:, 3] = y_norm[:, 3] * temp_range + C['temp_min']  # TWCR
    y[:, 5] = y_norm[:, 5] * temp_range + C['temp_min']  # TWSG

    # 압력 (Log 역변환)
    y[:, 2] = 10**(y_norm[:, 2] * np.log10(C['pressure_max']))  # PPS

    # 유량 (통일 범위에서 복원)
    y[:, 6] = y_norm[:, 6] * C['flow_max']   # HPSI
    y[:, 8] = y_norm[:, 8] * C['flow_max']   # WAFWSG

    # 기타
    y[:, 4] = y_norm[:, 4] * C['zwdc_max']   # ZWDC2SG

    return y


def denormalize_output_batch(Y_norm):
    """
    배치 출력 역정규화

    Parameters
    ----------
    Y_norm : list of np.ndarray, each shape (T_i, 9)
        정규화된 배치 출력

    Returns
    -------
    Y : list of np.ndarray, each shape (T_i, 9)
        원본 스케일 배치 출력
    """
    return [denormalize_output(y) for y in Y_norm]


# =============================================================================
# 검증 함수
# =============================================================================
def verify_normalization(y_original, rtol=1e-4, atol=10.0):
    """
    정규화 → 역정규화 후 원본과 일치 여부 검증

    Parameters
    ----------
    y_original : np.ndarray, shape (T, 9)
        원본 출력
    rtol : float
        상대 허용 오차
    atol : float
        절대 허용 오차 (압력 Pa 단위 고려)

    Returns
    -------
    is_valid : bool
    max_rel_error : float
        최대 상대 오차 (%)
    """
    y_norm = normalize_output(y_original)
    y_restored = denormalize_output(y_norm)

    # 상대 오차 계산 (0 나눗셈 방지)
    abs_diff = np.abs(y_original - y_restored)
    rel_error = abs_diff / (np.abs(y_original) + 1e-10) * 100  # %

    max_rel_error = rel_error.max()
    is_valid = np.allclose(y_original, y_restored, rtol=rtol, atol=atol)

    return is_valid, max_rel_error


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    import h5py

    DATA_PATH = r"D:\Diffusion_test\data\processed\tspred_v2_cleaned.h5"

    print("=" * 70)
    print("정규화 함수 검증")
    print("=" * 70)

    with h5py.File(DATA_PATH, 'r') as hf:
        # 샘플 데이터 로드
        x = hf['FirstData']['1']['input'][:]
        y = hf['FirstData']['1']['output'][:]

        print(f"\n[1] 입력 정규화 테스트")
        print(f"   원본 shape: {x.shape}")
        x_norm = normalize_input(x)
        print(f"   정규화 후 shape: {x_norm.shape}")
        print(f"   값 범위: [{x_norm[:23].min():.4f}, {x_norm[:23].max():.4f}]")
        print(f"   마스크 합계: {x_norm[23:].sum():.0f}")

        print(f"\n[2] 출력 정규화 테스트")
        print(f"   원본 shape: {y.shape}")
        y_norm = normalize_output(y)
        print(f"   정규화 후 shape: {y_norm.shape}")
        print(f"   값 범위: [{y_norm.min():.4f}, {y_norm.max():.4f}]")

        print(f"\n[3] 역정규화 검증")
        is_valid, max_rel_error = verify_normalization(y)
        print(f"   검증 결과: {'PASS' if is_valid else 'FAIL'}")
        print(f"   최대 상대 오차: {max_rel_error:.4f}%")

        # 전체 데이터 샘플링 검증
        print(f"\n[4] 전체 데이터 샘플 검증 (100개)")
        scenarios = [k for k in hf.keys() if not k.startswith('_')]
        rel_errors = []
        valid_count = 0
        np.random.seed(42)

        for scenario in scenarios:
            case_ids = list(hf[scenario].keys())
            sample_ids = np.random.choice(case_ids, min(20, len(case_ids)), replace=False)

            for case_id in sample_ids:
                y = hf[scenario][case_id]['output'][:]
                is_valid, rel_err = verify_normalization(y)
                rel_errors.append(rel_err)
                if is_valid:
                    valid_count += 1

        print(f"   검증 케이스: {len(rel_errors)}개")
        print(f"   통과: {valid_count}/{len(rel_errors)}")
        print(f"   최대 상대 오차: {max(rel_errors):.4f}%")
        print(f"   평균 상대 오차: {np.mean(rel_errors):.4f}%")

        if valid_count == len(rel_errors):
            print(f"   ✓ 모든 검증 통과")
        else:
            print(f"   ✗ 일부 검증 실패")

    print("\n" + "=" * 70)
