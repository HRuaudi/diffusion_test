# dataloader.py - Diffusion 모델용 DataLoader
# 정규화된 H5 파일에서 데이터 로드
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# =============================================================================
# 경로 설정
# =============================================================================
DATA_PATH = r"D:\Diffusion_test\data\processed\tspred_v2_normalized.h5"


# =============================================================================
# Dataset 클래스
# =============================================================================
class DiffusionDataset(Dataset):
    """
    Conditional Diffusion 모델용 Dataset

    Parameters
    ----------
    h5_path : str
        정규화된 H5 파일 경로
    scenarios : list, optional
        사용할 시나리오 목록 (None이면 전체)
    max_len : int, optional
        시계열 최대 길이 (패딩/자르기)
    split : str, optional
        'train', 'val', 'test' 중 하나
    split_ratio : tuple
        (train, val, test) 비율, 기본 (0.8, 0.1, 0.1)
    seed : int
        분할 시드
    """

    def __init__(
        self,
        h5_path=DATA_PATH,
        scenarios=None,
        max_len=None,
        split=None,
        split_ratio=(0.8, 0.1, 0.1),
        seed=42
    ):
        self.h5_path = h5_path
        self.max_len = max_len

        # 데이터 인덱스 로드
        self.samples = []  # (scenario, case_id)

        with h5py.File(h5_path, 'r') as hf:
            all_scenarios = [k for k in hf.keys() if not k.startswith('_')]

            if scenarios is None:
                scenarios = all_scenarios

            for scenario in scenarios:
                if scenario in hf:
                    case_ids = list(hf[scenario].keys())
                    for case_id in case_ids:
                        self.samples.append((scenario, case_id))

        # Train/Val/Test 분할
        if split is not None:
            np.random.seed(seed)
            indices = np.random.permutation(len(self.samples))

            n_train = int(len(indices) * split_ratio[0])
            n_val = int(len(indices) * split_ratio[1])

            if split == 'train':
                indices = indices[:n_train]
            elif split == 'val':
                indices = indices[n_train:n_train + n_val]
            elif split == 'test':
                indices = indices[n_train + n_val:]

            self.samples = [self.samples[i] for i in indices]

        # H5 파일 핸들 (lazy loading)
        self._hf = None

    def _get_hf(self):
        """Lazy H5 file handle"""
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, 'r')
        return self._hf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scenario, case_id = self.samples[idx]
        hf = self._get_hf()

        # 데이터 로드
        x = hf[scenario][case_id]['input'][:]   # (47,)
        y = hf[scenario][case_id]['output'][:]  # (T, 9)

        # 시계열 길이 처리
        T = y.shape[0]
        if self.max_len is not None:
            if T > self.max_len:
                # 자르기
                y = y[:self.max_len]
            elif T < self.max_len:
                # 패딩
                pad = np.zeros((self.max_len - T, y.shape[1]), dtype=np.float32)
                y = np.vstack([y, pad])
            T = self.max_len

        # Tensor 변환
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return {
            'condition': x,      # (47,)
            'output': y,         # (T, 9) or (max_len, 9)
            'length': T,         # 원본 길이
            'scenario': scenario,
            'case_id': case_id
        }

    def close(self):
        if self._hf is not None:
            self._hf.close()
            self._hf = None


# =============================================================================
# Collate 함수 (가변 길이 배치용)
# =============================================================================
def collate_variable_length(batch):
    """
    가변 길이 시계열 배치 처리
    패딩하여 동일 길이로 맞춤
    """
    conditions = torch.stack([b['condition'] for b in batch])
    lengths = [b['output'].shape[0] for b in batch]
    max_len = max(lengths)

    # 패딩
    batch_size = len(batch)
    n_vars = batch[0]['output'].shape[1]
    outputs = torch.zeros(batch_size, max_len, n_vars)

    for i, b in enumerate(batch):
        T = b['output'].shape[0]
        outputs[i, :T, :] = b['output']

    # 마스크 생성 (유효한 시점 = 1)
    mask = torch.zeros(batch_size, max_len)
    for i, T in enumerate(lengths):
        mask[i, :T] = 1

    return {
        'condition': conditions,  # (B, 47)
        'output': outputs,        # (B, T_max, 9)
        'mask': mask,             # (B, T_max)
        'lengths': torch.tensor(lengths)
    }


# =============================================================================
# DataLoader 생성 함수
# =============================================================================
def create_dataloaders(
    h5_path=DATA_PATH,
    batch_size=32,
    max_len=None,
    num_workers=0,
    split_ratio=(0.8, 0.1, 0.1),
    seed=42
):
    """
    Train/Val/Test DataLoader 생성

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_dataset = DiffusionDataset(
        h5_path=h5_path,
        max_len=max_len,
        split='train',
        split_ratio=split_ratio,
        seed=seed
    )

    val_dataset = DiffusionDataset(
        h5_path=h5_path,
        max_len=max_len,
        split='val',
        split_ratio=split_ratio,
        seed=seed
    )

    test_dataset = DiffusionDataset(
        h5_path=h5_path,
        max_len=max_len,
        split='test',
        split_ratio=split_ratio,
        seed=seed
    )

    # collate_fn 선택
    collate_fn = collate_variable_length if max_len is None else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DataLoader 테스트")
    print("=" * 70)

    # 1. 데이터셋 기본 테스트
    print("\n[1] Dataset 테스트")
    dataset = DiffusionDataset(split='train')
    print(f"   Train 샘플 수: {len(dataset)}")

    sample = dataset[0]
    print(f"   condition shape: {sample['condition'].shape}")
    print(f"   output shape: {sample['output'].shape}")
    print(f"   scenario: {sample['scenario']}")
    dataset.close()

    # 2. 시계열 길이 분포 확인
    print("\n[2] 시계열 길이 분포")
    lengths = []
    with h5py.File(DATA_PATH, 'r') as hf:
        scenarios = [k for k in hf.keys() if not k.startswith('_')]
        for scenario in scenarios:
            for case_id in list(hf[scenario].keys())[:100]:
                T = hf[scenario][case_id]['output'].shape[0]
                lengths.append(T)

    lengths = np.array(lengths)
    print(f"   샘플 수: {len(lengths)}")
    print(f"   min: {lengths.min()}, max: {lengths.max()}")
    print(f"   mean: {lengths.mean():.1f}, std: {lengths.std():.1f}")

    # 3. DataLoader 테스트 (고정 길이)
    print("\n[3] DataLoader 테스트 (max_len=512)")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=16,
        max_len=512
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # 첫 배치 확인
    batch = next(iter(train_loader))
    print(f"\n   Batch shapes:")
    print(f"   - condition: {batch['condition'].shape}")
    print(f"   - output: {batch['output'].shape}")

    # 4. DataLoader 테스트 (가변 길이)
    print("\n[4] DataLoader 테스트 (가변 길이)")
    train_loader_var, _, _ = create_dataloaders(
        batch_size=8,
        max_len=None
    )

    batch = next(iter(train_loader_var))
    print(f"   Batch shapes:")
    print(f"   - condition: {batch['condition'].shape}")
    print(f"   - output: {batch['output'].shape}")
    print(f"   - mask: {batch['mask'].shape}")
    print(f"   - lengths: {batch['lengths'].tolist()}")

    # 5. 시나리오별 분포
    print("\n[5] 시나리오별 샘플 수")
    with h5py.File(DATA_PATH, 'r') as hf:
        scenarios = [k for k in hf.keys() if not k.startswith('_')]
        for scenario in scenarios:
            n = len(hf[scenario].keys())
            print(f"   {scenario}: {n}")

    print("\n" + "=" * 70)
    print("DataLoader 테스트 완료")
    print("=" * 70)
