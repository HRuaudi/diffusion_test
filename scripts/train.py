# train.py - Conditional Diffusion Training Loop
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diffusion import GaussianDiffusion
from scripts.model import MLPDenoiser, ConvDenoiser
from scripts.dataloader import create_dataloaders

# =============================================================================
# 경로 설정
# =============================================================================
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tspred_v2_normalized.h5"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Training 함수
# =============================================================================
def train_epoch(model, diffusion, train_loader, optimizer, device, epoch):
    """Single training epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        condition = batch['condition'].to(device)  # (B, 47)
        output = batch['output'].to(device)        # (B, T, 9)

        optimizer.zero_grad()

        # Diffusion training loss
        loss = diffusion.training_loss(model, output, condition)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # 진행 상황 출력
        if (batch_idx + 1) % 50 == 0:
            print(f"   Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.6f}")

    avg_loss = total_loss / n_batches
    return avg_loss


def validate(model, diffusion, val_loader, device):
    """Validation loop"""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            condition = batch['condition'].to(device)
            output = batch['output'].to(device)

            loss = diffusion.training_loss(model, output, condition)

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


# =============================================================================
# Main Training
# =============================================================================
def main():
    print("=" * 70)
    print("Conditional Diffusion Model Training")
    print("=" * 70)

    # 설정
    config = {
        # Data
        'h5_path': str(DATA_PATH),
        'batch_size': 32,
        'max_len': 512,
        'sample_size': None,  # 전체 데이터 사용, 디버깅시 1000 등으로 설정

        # Model
        'model_type': 'conv',  # 'mlp' or 'conv'
        'hidden_dim': 128,
        'n_layers': 6,

        # Diffusion
        'num_timesteps': 1000,
        'beta_schedule': 'cosine',

        # Training
        'epochs': 100,
        'lr': 1e-4,
        'weight_decay': 1e-4,

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"\n[Config]")
    for k, v in config.items():
        print(f"   {k}: {v}")

    device = config['device']
    print(f"\nDevice: {device}")

    # ==========================================================================
    # DataLoader
    # ==========================================================================
    print("\n[1] DataLoader 생성")
    train_loader, val_loader, test_loader = create_dataloaders(
        h5_path=config['h5_path'],
        batch_size=config['batch_size'],
        max_len=config['max_len'],
        sample_size=config['sample_size']
    )

    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    print(f"   Train batches: {len(train_loader)}")

    # ==========================================================================
    # Model
    # ==========================================================================
    print("\n[2] Model 초기화")

    if config['model_type'] == 'mlp':
        model = MLPDenoiser(
            seq_len=config['max_len'],
            n_features=9,
            cond_dim=47,
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers']
        )
    else:
        model = ConvDenoiser(
            seq_len=config['max_len'],
            n_features=9,
            cond_dim=47,
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers']
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {config['model_type'].upper()}")
    print(f"   Parameters: {n_params:,}")

    # ==========================================================================
    # Diffusion
    # ==========================================================================
    print("\n[3] Diffusion 초기화")
    diffusion = GaussianDiffusion(
        num_timesteps=config['num_timesteps'],
        beta_schedule=config['beta_schedule'],
        device=device
    )
    print(f"   Timesteps: {config['num_timesteps']}")
    print(f"   Schedule: {config['beta_schedule']}")

    # ==========================================================================
    # Optimizer & Scheduler
    # ==========================================================================
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print("\n[4] Training 시작")
    print("-" * 70)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, diffusion, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, diffusion, val_loader, device)
        val_losses.append(val_loss)

        # Scheduler step
        scheduler.step()

        # Time
        epoch_time = time.time() - start_time

        # Logging
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{config['epochs']} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {lr:.2e} | Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = CHECKPOINT_DIR / f"best_model_{config['model_type']}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)
            print(f"   -> Best model saved: {save_path}")

        # Periodic checkpoint
        if epoch % 10 == 0:
            save_path = CHECKPOINT_DIR / f"checkpoint_{config['model_type']}_epoch{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)

    print("-" * 70)
    print(f"\nTraining 완료!")
    print(f"   Best Val Loss: {best_val_loss:.6f}")
    print(f"   Final Train Loss: {train_losses[-1]:.6f}")

    # ==========================================================================
    # Test Evaluation
    # ==========================================================================
    print("\n[5] Test 평가")
    test_loss = validate(model, diffusion, test_loader, device)
    print(f"   Test Loss: {test_loss:.6f}")

    # ==========================================================================
    # Sample Generation Test
    # ==========================================================================
    print("\n[6] Sample 생성 테스트")

    # 첫 배치에서 조건 가져오기
    test_batch = next(iter(test_loader))
    condition = test_batch['condition'][:4].to(device)  # 4개 샘플

    print(f"   Condition shape: {condition.shape}")

    # 생성 (시간 측정)
    start_time = time.time()
    with torch.no_grad():
        generated = diffusion.sample(model, condition, seq_len=config['max_len'])
    gen_time = time.time() - start_time

    print(f"   Generated shape: {generated.shape}")
    print(f"   Generation time: {gen_time:.2f}s (4 samples)")
    print(f"   Value range: [{generated.min().item():.4f}, {generated.max().item():.4f}]")

    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)


# =============================================================================
# Quick Test (1 epoch)
# =============================================================================
def quick_test():
    """1 epoch 빠른 테스트"""
    print("=" * 70)
    print("Quick Test (1 epoch, 100 samples)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # DataLoader (소량)
    print("\n[1] DataLoader")
    train_loader, val_loader, _ = create_dataloaders(
        h5_path=str(DATA_PATH),
        batch_size=16,
        max_len=256,
        sample_size=100
    )
    print(f"   Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Model (작은 크기)
    print("\n[2] Model")
    model = ConvDenoiser(
        seq_len=256,
        n_features=9,
        cond_dim=47,
        hidden_dim=64,
        n_layers=4
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # Diffusion (적은 timesteps)
    diffusion = GaussianDiffusion(
        num_timesteps=100,
        beta_schedule='cosine',
        device=device
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # 1 Epoch
    print("\n[3] Training (1 epoch)")
    train_loss = train_epoch(model, diffusion, train_loader, optimizer, device, 1)
    val_loss = validate(model, diffusion, val_loader, device)

    print(f"\n   Train Loss: {train_loss:.6f}")
    print(f"   Val Loss: {val_loss:.6f}")

    # Sample
    print("\n[4] Sample 생성")
    test_batch = next(iter(val_loader))
    condition = test_batch['condition'][:2].to(device)

    start = time.time()
    with torch.no_grad():
        generated = diffusion.sample(model, condition, seq_len=256)
    gen_time = time.time() - start

    print(f"   Shape: {generated.shape}")
    print(f"   Time: {gen_time:.2f}s")
    print(f"   Range: [{generated.min().item():.4f}, {generated.max().item():.4f}]")

    print("\n" + "=" * 70)
    print("Quick Test 완료")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test (1 epoch, 100 samples)')
    args = parser.parse_args()

    if args.quick:
        quick_test()
    else:
        main()
