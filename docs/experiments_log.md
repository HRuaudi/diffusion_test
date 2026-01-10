# Diffusion_test 실험 로그

> 모든 실험은 아래 형식으로 기록
> - 사실만 기록 (평가/아부 금지)
> - 실패도 기록

---

## 2025-01-09 - 프로젝트 초기 설정

### 작업 내용
- 프로젝트 폴더 구조 생성
- tspred_v2_new.h5 데이터 복사 (6.1GB)
- CLAUDE.md 지침 작성

### 기대 효과
- 확인사항: 프로젝트 구조가 Vibe coding에 적합한지
- 목표: 실험 환경 구축
- 배경: Conditional Diffusion 실험을 위한 새 프로젝트 필요

### 결과
- 폴더 구조 생성 완료
- HDF5 로드 테스트 성공 (34,774 샘플)
- 입력: (24,), 출력: (5130, 9) 확인

### 다음 단계
- Phase 1: 데이터 파이프라인 상세 검증
- 시나리오별 데이터 분포 확인

---
---
## 2026-01-10 - Bifurcation 분석 문서화 [codex-session001]

### 작업 내용
- Agent: codex
- Session: 001
- 재분석 결과(시간특징/데이터 기반 임계)를 문서로 정리
- 문서 위치: D:\Diffusion_test\docs\bifurcation_analysis_2026-01-10.md

### 기대 효과
- 확인사항: 분석 기준/임계/레짐 분포를 한 문서에서 확인 가능
- 목표: 조건부 확산모델 설계 시 참고 가능한 요약 제공
- 배경: 이전 bifurcation 분석 결과의 재현성 확보

### 결과
- 문서 생성 완료: D:\Diffusion_test\docs\bifurcation_analysis_2026-01-10.md

### 다음 단계
- 문서 기반으로 레짐별 입력 분포 및 시간특징 추가 검증
---
---
## 2026-01-10 - 동일 입력 조건 내 레짐 분기 분석 [codex-session001]

### 작업 내용
- Agent: codex
- Session: 001
- 24D 입력 동일성(전체/케이스번호 제외) 기준으로 케이스를 그룹화
- 데이터 기반 임계(p10/p90)로 레짐(TR, TRP) 라벨 부여 후 동일 입력 내 다중 레짐 여부 계산
- 레짐 다중 그룹의 시나리오 분포 확인

### 기대 효과
- 확인사항: 동일 조건에서 서로 다른 레짐 출력(진정한 bifurcation) 존재 여부
- 목표: 조건부 확산모델의 다중모달 필요성 판단 근거 확보
- 배경: 레짐은 케이스 내 변화가 아니라 동일 입력 간 출력 분기 여부로 정의

### 결과
- 임계값: max_TCLAD p10=609.239, p90=2903.157; min_RWST p10=0.0003169, p90=8.2218; t_peak_TCLAD p10=0.0000, p90=0.5504
- 24D(케이스번호 포함) 그룹: unique 34,586, dup_groups 188(376건), max_size 2, bifurcation 그룹 0
- 23D(케이스번호 제외) 그룹: unique 29,271, dup_groups 209(5,712건), max_size 4,491
- 23D 기준 bifurcation 그룹(레짐 TR/TRP): 12개, 해당 케이스 348건
- bifurcation 그룹은 모두 multi-scenario(>1 시나리오) 포함
- bifurcation 케이스 시나리오 분포(TR/TRP 동일): FirstData 222, PSV 119, SBO 4, SDS 3

### 다음 단계
- multi-scenario가 분기 원인인지 확인하기 위해 시나리오별로 동일 입력 그룹 재분석
- 입력 dim23(케이스 번호) 제외 외에 추가로 연속 변수 근사 매칭(라운딩) 효과 검토
---
---
## 2026-01-10 - 동일조건 분기/근사매칭/예측가능성 분석 [codex-session002]

### 작업 내용
- Agent: codex
- Session: 002
- 시나리오별 동일 입력(23D, dim23 제외) 내 레짐 다중 여부 계산
- 입력 근사 매칭(라운딩 bin) 기반 레짐 분기 비율 산출
- 입력→레짐 예측가능성(이산/연속 혼합 key) 계산
- 스크립트: D:\Diffusion_test\sandbox\codex\2026-01-10\session_002\20260110_codex_002_08_scenario_bifurcation.py
- 스크립트: D:\Diffusion_test\sandbox\codex\2026-01-10\session_002\20260110_codex_002_09_approx_bifurcation.py
- 스크립트: D:\Diffusion_test\sandbox\codex\2026-01-10\session_002\20260110_codex_002_10_regime_predictability.py

### 기대 효과
- 확인사항: 동일 조건 내 출력 레짐 분기 존재 여부
- 목표: 조건부 확산모델의 다중모달 필요성 판단 근거 강화
- 배경: 이전 분석에서 다중 레짐이 시나리오 혼합인지 확인 필요

### 결과
- 동일 입력(23D) 시나리오별 분기(레짐 TR/ TRP 기준, p10/p90 임계):
  - FirstData: total 9860, unique 7628, dup_groups 38, max_group 1745, bif_groups 2, bif_cases 212
  - PSV: total 4984, unique 4775, dup_groups 11, max_group 60, bif_groups 2, bif_cases 116
  - SBO: total 9040, unique 6140, dup_groups 28, max_group 2746, bif_groups 0
  - SDS: total 4535, unique 4535, dup_groups 0, bif_groups 0
  - Seal: total 6355, unique 6355, dup_groups 0, bif_groups 0
- 근사매칭(시나리오 포함, 23D 라운딩) 결과:
  - bin 0.5: unique 27002, bif_r1_groups 111 (706 cases), bif_r2_groups 113 (710 cases)
  - bin 1.0: unique 26246, bif_r1_groups 161 (1024 cases), bif_r2_groups 173 (1051 cases)
  - bin 5.0: unique 22129, bif_r1_groups 675 (3745 cases), bif_r2_groups 747 (4009 cases)
  - bin 10.0: unique 17679, bif_r1_groups 1217 (7572 cases), bif_r2_groups 1335 (8165 cases)
- 입력→레짐 예측가능성:
  - 레짐 분포: TM_RM 12048, TL_RF 8509, TM_RF 4829, TM_RE 3325, TL_RM 2585, TH_RM 1977, TH_RF 1134, TH_RE 367
  - baseline_majority_acc 0.3465
  - 이산차원 dims [10,16,20]만 사용: groups 8, amb_groups 8, majority_acc 0.4643, cond_entropy 1.3559 nats
  - 연속+이산(연속 상관 상위 dims 09/05/02/04/22): groups 3718, amb_groups 1455, majority_acc 0.7367, cond_entropy 0.6145 nats

### 다음 단계
- 동일 입력 bifurcation 그룹(FirstData/PSV 2개씩)의 입력 조건 상세 목록화
- 근사매칭 bin 크기에 따른 분기 증가가 시나리오/연속치 버킷팅 영향인지 추가 분리
- 예측가능성 결과를 기반으로 레짐 분류 헤드(조건 모델) 도입 여부 결정
---
---
## 2026-01-10 - 모델 성능 한계 요인 분석 및 데이터 전처리 [claude-session001]

### 작업 내용
- Agent: claude
- Session: 001
- 모델 개발 전 데이터 특성 심층 분석 (6가지)
- 시나리오 간 중복 케이스 분석 및 제거
- 입력 특수값(1000) 전처리

### 분석 스크립트
- `sandbox/claude/2026-01-10/08_prediction_difficulty.py` - 예측 난이도 (R², 입력-출력 관계)
- `sandbox/claude/2026-01-10/09_timeseries_complexity.py` - 시계열 복잡도 (자기상관, 급변)
- `sandbox/claude/2026-01-10/10_variable_difficulty.py` - 변수별 난이도 (SNR, 스펙트럼)
- `sandbox/claude/2026-01-10/11_temporal_difficulty.py` - 시간 구간별 난이도
- `sandbox/claude/2026-01-10/12_mutual_information.py` - 입력 변수 중요도 (MI)
- `sandbox/claude/2026-01-10/13_data_quality.py` - 데이터 품질 이슈
- `sandbox/claude/2026-01-10/14_preprocess_h5.py` - H5 전처리

### 기대 효과
- 확인사항: 모델 성능 저하 예상 요인 사전 파악
- 목표: 전처리된 학습 데이터 생성
- 배경: bifurcation 분석 완료 후 모델 개발 전 데이터 특성 파악 필요

### 결과

#### 1. 모델 성능 한계 요인 분석
| 분석 항목 | 주요 발견 |
|-----------|----------|
| 예측 난이도 | 평균 R²=0.78, 입력-출력 거리 상관=0.023 (낮음) |
| 시계열 복잡도 | 자기상관 높음 (Diffusion 유리), 급변은 초기 집중 |
| 변수별 난이도 | HPSI/RWST/WAFWSG 어려움 (SNR 낮음), zw_vessel/TWSG/TWCR 쉬움 |
| 시간 구간 | 초기(0-10%) 난이도 0.233, 후기(70-100%) 0.016 (9배 차이) |
| 입력 중요도 | SI1_ON, CSS, SI1_OFF 가장 중요 (MI>0.44) |
| 데이터 품질 | 특수값 17개 변수 30%+, T 범위 429-5185 |

#### 2. 시나리오 중복 분석
- FirstData + SBO 중복: 187개 (케이스 ID 동일, 입력 동일)
- FirstData + PSV 중복: 1개
- 출력 비교: 116개 완전 동일, 71개 수치 정밀도 차이 (상대차이 <0.3%)
- 결론: 진정한 bifurcation 아님, 데이터 중복 저장

#### 3. 데이터 전처리
| 항목 | 원본 | 전처리 후 |
|------|------|----------|
| 총 케이스 | 34,774 | 34,586 |
| 제거 (SBO) | - | 187 |
| 제거 (PSV) | - | 1 |
| 1000→72 변환 | 226,672회 | - |

### 생성 파일
- 원본: `data/raw/tspred_v2_new.h5`
- 전처리: `data/processed/tspred_v2_cleaned.h5`
- 시각화: `sandbox/claude/2026-01-10/*.png` (5개)

### 다음 단계
- 전처리된 데이터로 DataLoader 구성
- 초기 구간 가중치 조정 또는 별도 모델 검토
- HPSI/RWST/WAFWSG 변수 별도 처리 검토
---
