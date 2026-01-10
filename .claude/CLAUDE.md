# Diffusion_test 프로젝트 지침

## 1. 프로젝트 목표
Conditional Diffusion Model로 시계열 생성 (Bifurcation 분석)
- 입력(condition): 24D 벡터 (사고 조건)
- 출력: (T, 9) 시계열 (물리량 변화)
- 데이터: D:\Diffusion_test\data\raw\tspred_v2_new.h5

## 2. 모델 구조 (추후 결정)
```
[훈련 - 일반적인 구조]
condition (24D) ──┬──► Denoiser ──► noise_pred ──► loss
                  │
noisy_y (T, 9) ───┘

[추론]
condition (24D) ──► Denoiser ──► iterative denoise ──► y_hat (T, 9)
```
※ Denoiser 아키텍처는 Phase 6~7에서 실험 후 결정
- 후보: UNet, Transformer, TCN, MLP 등

## 3. 코드 작성 규칙

### 3.1 초기 단계 (검증 우선)
- 클래스/함수 사용 금지
- 로직별 독립 스크립트로 작성
- 셸에서 바로 실행 가능하도록 **절대 경로** 사용
- 출력 위치: `D:\Diffusion_test\sandbox\claude\{YYYY-MM-DD}\`

### 3.2 경로 규칙
```python
# 모든 스크립트 상단에 이 경로들 사용
PROJECT_ROOT = r"D:\Diffusion_test"
DATA_PATH = r"D:\Diffusion_test\data\raw\tspred_v2_new.h5"
SANDBOX_PATH = r"D:\Diffusion_test\sandbox\claude"  # + 날짜
```

### 3.3 스크립트 작성 예시
```python
# 01_load_data.py - HDF5 로드 테스트
import h5py
import numpy as np

DATA_PATH = r"D:\Diffusion_test\data\raw\tspred_v2_new.h5"

with h5py.File(DATA_PATH, 'r') as hf:
    scenarios = [k for k in hf.keys() if not k.startswith('_')]
    print(f"시나리오: {scenarios}")

    x = hf['FirstData']['1']['input'][:]   # (24,)
    y = hf['FirstData']['1']['output'][:]  # (T, 9)
    print(f"input shape: {x.shape}")
    print(f"output shape: {y.shape}")
```

## 4. 실험 로그 (필수)

모든 작업 후 `D:\Diffusion_test\docs\experiments_log.md`에 기록

### 로그 형식
```markdown
---
## YYYY-MM-DD - {작업 제목}

### 작업 내용
- {구체적 작업}

### 기대 효과
- 확인사항: {검증하려는 가설}
- 목표: {달성 목표}
- 배경: {이전 실험과의 연결고리}

### 결과
- {수치/관찰 결과}

### 다음 단계
- {후속 작업}
---
```

### 로그 작성 원칙
1. 사실만 기록 (추측/평가 금지)
2. 수치는 정확히
3. 실패도 기록
4. 다음 단계는 구체적으로

## 5. 금지 사항
- 아부/칭찬 금지
- 주관적 평가 금지
- 목표 외 작업 금지
- 불필요한 설명 금지

## 6. 데이터 구조

### 입력 (24D)
```
0-7   AC전원 1~4 ON/OFF 시간
8-15  안전주입/급수 ON/OFF
16    PSV 고착 플래그
17-20 SDS, CSS, 재순환
21-22 Seal LOCA 시간/면적
23    케이스 번호
특수값: -1=실패, 1000=계속작동
```

### 출력 (T×9)
```
0 zw_vessel   1 TCLAD_HOT  2 PPS
3 TWCR        4 ZWDC2SG    5 TWSG
6 HPSI        7 RWST       8 WAFWSG
```

## 7. 실험 로드맵

### Phase 1: 데이터 파이프라인
- [ ] HDF5 로드
- [ ] 24D/9D 추출
- [ ] 통계 확인

### Phase 2: Bifurcation 분석
- [ ] H5 인터랙션 뷰어 (다중 케이스 비교)
- [ ] 입력→출력 패턴 분류
- [ ] 임계점 탐색

### Phase 3: 전처리
- [ ] 정규화 방법
- [ ] 특수값 처리
- [ ] 마스킹/패딩

### Phase 4: DataLoader & 시각화
- [ ] 배치 구성
- [ ] 훈련/검증 시각화 도구

### Phase 5: Diffusion 기초
- [ ] Forward process
- [ ] Beta schedule
- [ ] Noise 검증

### Phase 6: 모델 아키텍처 실험
- [ ] Baseline 1: Simple MLP
- [ ] Baseline 2: 1D UNet
- [ ] Baseline 3: Transformer
- [ ] 비교 평가

### Phase 7: 최적화 & 평가
- [ ] 최종 모델 선정
- [ ] 하이퍼파라미터 튜닝
- [ ] Bifurcation 재현 평가

## 8. Multi-Agent 충돌 방지

### 8.1 작업 전 필수 확인
```python
# 모든 스크립트/작업 시작 전 실행
import os
from pathlib import Path
from datetime import datetime

# 1. 현재 상태 확인
print("=== 작업 전 상태 확인 ===")
print(f"현재 시간: {datetime.now()}")

# 2. 락 파일 확인
LOCK_DIR = Path(r"D:\Diffusion_test\.locks")
LOCK_DIR.mkdir(exist_ok=True)

active_locks = list(LOCK_DIR.glob("*.lock"))
if active_locks:
    print(f"⚠️ 활성 락 파일: {[l.name for l in active_locks]}")
    for lock in active_locks:
        print(f"   {lock.read_text()}")
```

### 8.2 에이전트별 작업 공간 분리
```
sandbox/
├── claude/           # Claude 전용
│   └── 2025-01-09/
│       ├── session_001/   # 세션별 분리
│       └── session_002/
├── codex/            # Codex 전용
│   └── 2025-01-09/
└── _shared/          # 공유 결과물 (완료된 것만)
```

### 8.3 락 파일 규칙
```python
# 작업 시작 시 락 생성
def acquire_lock(task_name, agent_id):
    lock_file = LOCK_DIR / f"{task_name}.lock"
    if lock_file.exists():
        raise RuntimeError(f"락 존재: {lock_file.read_text()}")
    lock_file.write_text(f"{agent_id}|{datetime.now()}|{task_name}")
    return lock_file

# 작업 완료 시 락 해제
def release_lock(lock_file):
    if lock_file.exists():
        lock_file.unlink()
```

### 8.4 파일 네이밍 규칙
```
{날짜}_{에이전트}_{세션}_{작업번호}_{설명}.py

예시:
20250109_claude_001_01_load_data.py
20250109_codex_002_03_train_test.py
```

### 8.5 로그 기록 시 에이전트 명시
```markdown
## 2025-01-09 - 데이터 로드 테스트 [claude-session001]

### 작업 내용
- Agent: claude
- Session: 001
- 작업: HDF5 로드 검증
```

### 8.6 공유 리소스 접근 규칙
| 리소스 | 규칙 |
|--------|------|
| HDF5 데이터 | 읽기 전용 (동시 접근 OK) |
| experiments_log.md | 락 필수, append only |
| src/ 폴더 | 락 필수, 한 에이전트만 수정 |
| sandbox/{agent}/ | 각 에이전트 전용 |
| checkpoints/ | 파일명에 에이전트+세션 포함 |

### 8.7 충돌 발생 시 처리
1. **파일 충돌**: 나중 작업을 `.conflict` 확장자로 저장
2. **GPU 충돌**: `nvidia-smi` 확인 후 빈 GPU 사용
3. **로그 충돌**: 타임스탬프로 정렬 후 병합

### 8.8 작업 완료 시 체크리스트
- [ ] 락 파일 해제
- [ ] 결과물 저장 확인
- [ ] experiments_log.md 업데이트
- [ ] 다음 에이전트를 위한 상태 메모
