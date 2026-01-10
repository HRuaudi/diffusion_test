# Diffusion_test 프로젝트 계획

## 목표
원자력 발전소 시뮬레이션 데이터(tspred)에 대한 Conditional Diffusion Model 실험
- **핵심 연구 질문**: 입력 조건에 따른 시계열 출력의 bifurcation 패턴 분석

## 데이터
- **소스**: `data/raw/tspred_v2_new.h5`
- **입력**: 24D 사고 조건 벡터 (AC전원 시간, 플래그 등)
- **출력**: 9D × T 가변 길이 시계열 (물리량 변화)
- **샘플 수**: 34,774개 (5개 시나리오)
- **상세 정보**: `data/README.md` 참조

## 개발 방식
Vibe Coding (Claude, Codex 활용)
```
sandbox/ → tests/ 통과 → src/
(AI 생성)   (확정 기준)    (프로덕션)
```

## 실험 단위
날짜/세션별 관리
```
experiments/
└── 2025-01-09/
    ├── session_01/
    └── session_02/
```

## 마일스톤

### Phase 1: 기반 구축 (현재)
- [x] 프로젝트 구조 설정
- [x] 데이터 복사 및 검증
- [ ] DataLoader 구현
- [ ] 기본 Diffusion 모델 구현

### Phase 2: 학습 파이프라인
- [ ] Training loop 구현
- [ ] Evaluation metrics 정의
- [ ] 첫 번째 학습 실행

### Phase 3: Bifurcation 분석
- [ ] 조건별 샘플링 실험
- [ ] 입력 변화에 따른 출력 분포 시각화
- [ ] 임계점(critical point) 탐색

## 참고 자료
- 원본 프로젝트: `D:/Research/FinetuningTest`
- 데이터 전처리 문서: `data/README.md`
