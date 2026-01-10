# tspred_v2_new.h5 생성 가이드

## 개요
원자력 발전소 사고 시나리오 시뮬레이션 데이터를 ML용 HDF5로 변환한 파일.

## 데이터 구조

### 입력 (24D 벡터)
```
idx  변수명              설명                    특수값
───────────────────────────────────────────────────────
0-1  AC1_ON/OFF         AC전원1 ON/OFF 시간     -1=실패, 1000=계속작동
2-3  AC2_ON/OFF         AC전원2 ON/OFF 시간
4-5  AC3_ON/OFF         AC전원3 ON/OFF 시간
6-7  AC4_ON/OFF         AC전원4 ON/OFF 시간
8-9  HPSI_ON/OFF        고압안전주입 ON/OFF
10-11 TDAFW_ON/OFF      터빈구동보조급수 ON/OFF
12-13 MDAFW_ON/OFF      모터구동보조급수 ON/OFF
14-15 PLPP_ON/OFF       저압재순환펌프 ON/OFF
16   PSV_Flag           가압기안전밸브 고착 (0/1)
17   SDS_Time           급속감압계통 개방 시간
18-19 CSS_ON/OFF        격납건물살수 ON/OFF
20   Recirc_OFF         재순환 정지 시간
21   LOCA_Time          밀봉 LOCA 발생 시간
22   LOCA_Area          밀봉 LOCA 면적
23   CaseNum            케이스 번호
```

### 출력 (9D × T 시계열)
```
idx  변수명       설명
─────────────────────────
0    zw_vessel   원자로 수위
1    TCLAD_HOT   피복재 최고온도
2    PPS         가압기 압력
3    TWCR        원자로 냉각재 온도
4    ZWDC2SG(1)  1번 증기발생기 수위
5    TWSG(1)     1번 증기발생기 온도
6    HPSI        고압안전주입 유량
7    RWST        안전주입탱크 유량
8    WAFWSG(1)   보조급수 유량
```

## 시나리오별 샘플 수 (총 34,774)
- FirstData: 9,860
- PSV: 4,984
- SBO: 9,040
- SDS: 4,535
- Seal: 6,355

## 생성 파이프라인

### 1단계: 원본 데이터 위치
```
E:/250630_SboData/
├── FirstData_input/     *.inp 파일들
├── FirstData_output/    *.D12.CSV 파일들
├── z5-PSV_input/
├── z5-PSV_output/
└── ...
```

### 2단계: 전처리 (preprocess_data.py)
```bash
cd data/data_preprocessing
python preprocess_data.py
```
- `.inp` → 24D numpy 벡터
- `.D12.CSV` → 9D 시계열 CSV
- 출력: `data/processed_v2/시나리오_input_시나리오_output/`

**핵심 전처리 로직:**
```python
# ON/OFF 쌍 처리
if V[on_pos] == -1:    V[off_pos] = -1      # ON 실패 → OFF도 실패
elif V[on_pos] == 1000: V[off_pos] = 1000   # 계속 작동
elif V[off_pos] == -1:  V[off_pos] = 1000   # OFF 실패 → 1000

# PSV 플래그 (3가지 조건 OR)
if Flag_PSV_Stuck==1 or Flag_PSV_SpuriousStuck==1 or Time_PSV_SpuriousStuck==1:
    V[16] = 1

# AC Power 1000 전파 (순차적)
for idx in ac_positions:
    if V[idx] == 1000:
        V[idx+1:] = 1000  # 이후 모두 1000
```

### 3단계: HDF5 통합 (create_hdf5_dataset.py)
```bash
python create_hdf5_dataset.py
```
- 출력: `data/processed_v2/tspred_v2_new.h5`

## HDF5 구조
```
/
├── _index/
│   ├── input_features   (24,) 변수명
│   ├── output_features  (9,) 변수명
│   ├── scenario_list    시나리오 목록
│   └── case_counts/     시나리오별 샘플 수
├── FirstData/
│   ├── 1/
│   │   ├── input   (24,) float32
│   │   └── output  (T, 9) float32
│   ├── 2/
│   └── ...
├── PSV/
├── SBO/
├── SDS/
└── Seal/
```

## 사용 예시
```python
import h5py
import numpy as np

with h5py.File('tspred_v2_new.h5', 'r') as hf:
    # 메타데이터
    scenarios = [s.decode() for s in hf['_index/scenario_list'][:]]
    input_names = [f.decode() for f in hf['_index/input_features'][:]]
    output_names = [f.decode() for f in hf['_index/output_features'][:]]

    # 데이터 로드
    x = hf['FirstData']['1']['input'][:]   # (24,)
    y = hf['FirstData']['1']['output'][:]  # (5130, 9)

    # 전체 순회
    for scenario in scenarios:
        for case_id in hf[scenario].keys():
            inp = hf[f'{scenario}/{case_id}/input'][:]
            out = hf[f'{scenario}/{case_id}/output'][:]
```

## 주의사항
1. **원본 스케일**: 정규화 안됨, 물리 단위 그대로
2. **가변 길이**: 출력 시계열 길이가 샘플마다 다름 (약 4300~5200)
3. **특수값**: -1(실패), 1000(계속작동)은 도메인 의미 있음
4. **원본 데이터 필요시**: `E:/250630_SboData/` 경로 확인

## 관련 파일
- `data/data_preprocessing/preprocess_data.py` - 전처리 코드
- `data/data_preprocessing/create_hdf5_dataset.py` - HDF5 생성
- `data/data_preprocessing/process.md` - 상세 변수 설명
- `utils/DataLoad.py` - PyTorch DataLoader
