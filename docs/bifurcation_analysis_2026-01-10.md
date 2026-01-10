# Bifurcation 분석 결과 요약 (시간특징/데이터 기반 임계)

## 범위
- 데이터: D:\Diffusion_test\data\raw\tspred_v2_new.h5
- 샘플: 34,774 case, 시나리오 5개 (FirstData/PSV/SBO/SDS/Seal)
- 출력 채널: TCLAD_HOT, PPS, RWST 중심 분석

## 특징 정의 (시간 구조 포함)
- max_TCLAD = max(TCLAD_HOT)
- t_peak_TCLAD = argmax(TCLAD_HOT)/(T-1)
- auc_TCLAD = mean(TCLAD_HOT)
- early_slope_TCLAD = (TCLAD_HOT[t_10%] - TCLAD_HOT[0])/(t_10%+1)
- osc_TCLAD = sign(diff(TCLAD_HOT)) 변화 횟수
- recov_ratio_TCLAD = (max_TCLAD - TCLAD_HOT[-1]) / max_TCLAD
- max_PPS, t_peak_PPS, auc_PPS
- min_RWST, t_min_RWST

## 입력 특성
- 상수 차원: dim07=1000, dim14=-1, dim15=-1 (unique=1)
- 특수값: -1/1000 존재, 상관 계산 시 제외

## 데이터 기반 임계(분위수)
- max_TCLAD p10=609.239, p50=2692.176, p90=2903.157, p99=2911.000
- min_RWST p10=0.0003169, p50=2.0476, p90=8.2218
- t_peak_TCLAD p10=0.0000, p50=0.1465, p90=0.5504
- osc_TCLAD p90=1460

## 레짐 분포 (분위수 기준)
- low_tclad<=p10: 11,094
- high_tclad>=p99: 3,172 (p99=2911.0과 동률 값 포함)
- rwst_empty<=p10: 3,692
- rwst_full>=p90: 14,472
- low_tclad & rwst_full: 8,509
- low_tclad & ~rwst_full: 2,585
- ~low_tclad & rwst_full: 5,963
- ~low_tclad & ~rwst_full: 17,717

## 시나리오별 레짐 비율
| scenario | n | low_tclad | high_tclad | rwst_full | rwst_empty |
|---|---:|---:|---:|---:|---:|
| FirstData | 9860 | 0.366 | 0.076 | 0.384 | 0.123 |
| PSV | 4984 | 0.079 | 0.191 | 0.122 | 0.201 |
| SBO | 9040 | 0.628 | 0.065 | 0.875 | 0.012 |
| SDS | 4535 | 0.127 | 0.081 | 0.051 | 0.209 |
| Seal | 6355 | 0.132 | 0.081 | 0.306 | 0.067 |

## 시간특징 차이 (low vs high)
- t_peak_TCLAD 평균: low 0.0000, high 0.1582
- recov_ratio_TCLAD 평균: low 0.2136, high 0.9637
- min_RWST 평균: low 6.5993, high 3.7884
- early_slope_TCLAD 평균: low -0.0835, high -0.0502
- t_peak_PPS 평균: low 0.0191, high 0.0568

## 입력 연관성
- 이산차원 MI (nats, 상위): low_tclad -> dim20 0.056, dim16 0.046, dim10 0.0077; rwst_full -> dim20 0.0849, dim16 0.0705, dim10 0.0059
- 상관(특수값 제외, max_TCLAD): dim16 0.310, dim09 -0.222, dim02 0.138, dim22 0.129, dim00 0.128, dim21 -0.121

## 확산모델 관점 시사점
- 시나리오별 레짐 비율 차이가 커서 조건 입력에 시나리오 ID를 명시적으로 포함 필요
- 특수값(-1/1000)은 별도 마스크/카테고리 처리 필요
- high_tclad는 p99 임계값(2911.0) 동률 값 포함으로 3,172건
- 시간특징 차이가 있어 레짐별 평가 지표(peak 위치, 회복률)를 함께 사용 필요

## 관련 로그
- D:\Diffusion_test\docs\experiments_log.md (2026-01-10, Bifurcation 특징 추출/레짐 분리)
- D:\Diffusion_test\docs\experiments_log.md (2026-01-10, Bifurcation 추가분석)
