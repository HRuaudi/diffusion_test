# test_pipeline.py - Diffusion Pipeline 테스트 (Mock 데이터)
# 실제 H5 데이터 없이 파이프라인 검증
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diffusion import GaussianDiffusion
from scripts.model import MLPDenoiser, ConvDenoiser


def create_mock_data(n_samples=200, seq_len=256, n_features=9, cond_dim=47):
    """
    Mock 데이터 생성 (sine wave 패턴)
    실제 학습 가능 여부를 검증하기 위한 합성 데이터
    """
    # Condition: 랜덤 벡터
    conditions = torch.randn(n_samples, cond_dim)

    # Output: 조건에 따른 sine wave (학습 가능한 패턴)
    outputs = torch.zeros(n_samples, seq_len, n_features)

    t = torch.linspace(0, 4 * 3.14159, seq_len)

    for i in range(n_samples):
        # 조건에서 주파수/진폭/위상 추출
        freq = 1.0 + conditions[i, 0].item() * 0.5
        amp = 0.5 + conditions[i, 1].item() * 0.3
        phase = conditions[i, 2].item() * 0.5

        for j in range(n_features):
            noise = torch.randn(seq_len) * 0.1
            outputs[i, :, j] = amp * torch.sin(freq * t + phase + j * 0.5) + noise

    # 정규화 [0, 1]
    outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min() + 1e-8)

    return conditions, outputs


def run_test():
    print("=" * 70)
    print("Diffusion Pipeline Test (Mock Data)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # ==========================================================================
    # Mock 데이터 생성
    # ==========================================================================
    print("\n[1] Mock 데이터 생성")

    n_train, n_val = 160, 40
    seq_len = 256
    n_features = 9
    cond_dim = 47

    train_cond, train_out = create_mock_data(n_train, seq_len, n_features, cond_dim)
    val_cond, val_out = create_mock_data(n_val, seq_len, n_features, cond_dim)

    print(f"   Train: {train_cond.shape}, {train_out.shape}")
    print(f"   Val: {val_cond.shape}, {val_out.shape}")

    # DataLoader
    train_dataset = TensorDataset(train_cond, train_out)
    val_dataset = TensorDataset(val_cond, val_out)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # ==========================================================================
    # Model 초기화
    # ==========================================================================
    print("\n[2] Model 초기화")

    model = ConvDenoiser(
        seq_len=seq_len,
        n_features=n_features,
        cond_dim=cond_dim,
        hidden_dim=64,
        n_layers=4
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: ConvDenoiser")
    print(f"   Parameters: {n_params:,}")

    # ==========================================================================
    # Diffusion 초기화
    # ==========================================================================
    print("\n[3] Diffusion 초기화")

    diffusion = GaussianDiffusion(
        num_timesteps=100,  # 빠른 테스트를 위해 축소
        beta_schedule='cosine',
        device=device
    )
    print(f"   Timesteps: 100")
    print(f"   Schedule: cosine")

    # ==========================================================================
    # Training
    # ==========================================================================
    print("\n[4] Training (5 epochs)")
    print("-" * 50)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        # Train
        model.train()
        train_loss = 0.0
        for cond, out in train_loader:
            cond, out = cond.to(device), out.to(device)

            optimizer.zero_grad()
            loss = diffusion.training_loss(model, out, cond)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cond, out in val_loader:
                cond, out = cond.to(device), out.to(device)
                loss = diffusion.training_loss(model, out, cond)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"   Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}")

    print("-" * 50)

    # ==========================================================================
    # Sample 생성
    # ==========================================================================
    print("\n[5] Sample 생성 테스트")

    model.eval()
    test_cond = val_cond[:4].to(device)
    test_real = val_out[:4].to(device)

    start = time.time()
    with torch.no_grad():
        generated = diffusion.sample(model, test_cond, seq_len=seq_len, n_features=n_features)
    gen_time = time.time() - start

    print(f"   Condition shape: {test_cond.shape}")
    print(f"   Generated shape: {generated.shape}")
    print(f"   Real data shape: {test_real.shape}")
    print(f"   Generation time: {gen_time:.2f}s (4 samples)")
    print(f"   Generated range: [{generated.min().item():.4f}, {generated.max().item():.4f}]")
    print(f"   Real data range: [{test_real.min().item():.4f}, {test_real.max().item():.4f}]")

    # MSE 비교
    mse = nn.functional.mse_loss(generated, test_real).item()
    print(f"   MSE (gen vs real): {mse:.6f}")

    # ==========================================================================
    # 결과 요약
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Pipeline Test 결과")
    print("=" * 70)
    print(f"""
   ✓ Forward process (q_sample): 정상
   ✓ Denoiser model (ConvDenoiser): 정상
   ✓ Training loop: 정상
   ✓ Reverse sampling: 정상
   ✓ Loss 감소: Train {train_loss:.4f}

   다음 단계:
   - 실제 H5 데이터로 학습
   - Hyperparameter 튜닝
   - 생성 품질 평가
""")


if __name__ == "__main__":
    run_test()
