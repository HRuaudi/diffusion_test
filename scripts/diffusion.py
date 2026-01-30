# diffusion.py - Diffusion Forward Process
# Beta schedule, noise 추가, sampling
import torch
import torch.nn as nn
import numpy as np


class GaussianDiffusion:
    """
    Gaussian Diffusion Process for Time Series Generation

    Parameters
    ----------
    num_timesteps : int
        Diffusion timesteps (T)
    beta_schedule : str
        'linear' or 'cosine'
    beta_start : float
        Starting beta value (for linear)
    beta_end : float
        Ending beta value (for linear)
    device : str
        'cuda' or 'cpu'
    """

    def __init__(
        self,
        num_timesteps=1000,
        beta_schedule='linear',
        beta_start=1e-4,
        beta_end=0.02,
        device='cuda'
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        # Beta schedule 생성
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas.to(device)

        # Diffusion 상수 계산
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device),
            self.alphas_cumprod[:-1]
        ])

        # q(x_t | x_0) 계산용
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # p(x_{t-1} | x_t) 계산용
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule (improved diffusion)"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: q(x_t | x_0)
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

        Parameters
        ----------
        x_0 : torch.Tensor, shape (B, T, D)
            Original clean data
        t : torch.Tensor, shape (B,)
            Timestep indices
        noise : torch.Tensor, optional
            Pre-sampled noise

        Returns
        -------
        x_t : torch.Tensor, shape (B, T, D)
            Noisy data at timestep t
        noise : torch.Tensor, shape (B, T, D)
            Added noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # 브로드캐스팅을 위해 shape 맞춤
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def p_sample(self, model, x_t, t, condition):
        """
        Reverse process: p(x_{t-1} | x_t)
        Single denoising step

        Parameters
        ----------
        model : nn.Module
            Denoiser network
        x_t : torch.Tensor, shape (B, T, D)
            Noisy data at timestep t
        t : torch.Tensor, shape (B,)
            Current timestep
        condition : torch.Tensor, shape (B, C)
            Conditioning input

        Returns
        -------
        x_t_minus_1 : torch.Tensor
            Denoised data at timestep t-1
        """
        # 모델로 noise 예측
        noise_pred = model(x_t, t, condition)

        # x_{t-1} 계산
        sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None]
        beta = self.betas[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        # Mean prediction
        x_mean = sqrt_recip_alpha * (x_t - beta * noise_pred / sqrt_one_minus_alpha)

        # Add noise (except at t=0)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            posterior_var = self.posterior_variance[t][:, None, None]
            x_t_minus_1 = x_mean + torch.sqrt(posterior_var) * noise
        else:
            x_t_minus_1 = x_mean

        return x_t_minus_1

    @torch.no_grad()
    def sample(self, model, condition, seq_len, n_features=9):
        """
        Full reverse sampling: generate x_0 from noise

        Parameters
        ----------
        model : nn.Module
            Trained denoiser
        condition : torch.Tensor, shape (B, C)
            Conditioning input
        seq_len : int
            Output sequence length
        n_features : int
            Number of output features

        Returns
        -------
        x_0 : torch.Tensor, shape (B, seq_len, n_features)
            Generated time series
        """
        model.eval()
        batch_size = condition.shape[0]
        device = condition.device

        # Start from pure noise
        x = torch.randn(batch_size, seq_len, n_features, device=device)

        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, condition)

        return x

    def training_loss(self, model, x_0, condition, noise=None):
        """
        Training loss: MSE between predicted and actual noise

        Parameters
        ----------
        model : nn.Module
            Denoiser network
        x_0 : torch.Tensor, shape (B, T, D)
            Clean target data
        condition : torch.Tensor, shape (B, C)
            Conditioning input
        noise : torch.Tensor, optional
            Pre-sampled noise

        Returns
        -------
        loss : torch.Tensor
            MSE loss
        """
        batch_size = x_0.shape[0]

        # Random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device)

        # Add noise
        x_t, noise = self.q_sample(x_0, t, noise)

        # Predict noise
        noise_pred = model(x_t, t, condition)

        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Diffusion Forward Process 테스트")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Diffusion 초기화
    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        beta_schedule='linear',
        device=device
    )

    # 테스트 데이터
    batch_size = 4
    seq_len = 512
    n_features = 9

    x_0 = torch.randn(batch_size, seq_len, n_features, device=device)

    print(f"\n[1] Forward Process 테스트")
    print(f"   x_0 shape: {x_0.shape}")

    # 다양한 timestep에서 테스트
    for t_val in [0, 250, 500, 750, 999]:
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
        x_t, noise = diffusion.q_sample(x_0, t)

        # 노이즈 비율 확인
        signal_ratio = diffusion.sqrt_alphas_cumprod[t_val].item()
        noise_ratio = diffusion.sqrt_one_minus_alphas_cumprod[t_val].item()

        print(f"   t={t_val:4d}: signal={signal_ratio:.4f}, noise={noise_ratio:.4f}")

    print(f"\n[2] Beta Schedule 비교")

    linear_diff = GaussianDiffusion(num_timesteps=1000, beta_schedule='linear', device=device)
    cosine_diff = GaussianDiffusion(num_timesteps=1000, beta_schedule='cosine', device=device)

    print(f"   Linear - beta range: [{linear_diff.betas[0]:.6f}, {linear_diff.betas[-1]:.6f}]")
    print(f"   Cosine - beta range: [{cosine_diff.betas[0]:.6f}, {cosine_diff.betas[-1]:.6f}]")

    # Alpha cumprod 비교 (t=500)
    print(f"   Linear - alpha_cumprod[500]: {linear_diff.alphas_cumprod[500]:.4f}")
    print(f"   Cosine - alpha_cumprod[500]: {cosine_diff.alphas_cumprod[500]:.4f}")

    print(f"\n[3] Noise Schedule 시각화 데이터")
    t_steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    print("   t   | Linear SNR | Cosine SNR")
    print("   ----|------------|------------")
    for t in t_steps:
        linear_snr = linear_diff.sqrt_alphas_cumprod[t] / linear_diff.sqrt_one_minus_alphas_cumprod[t]
        cosine_snr = cosine_diff.sqrt_alphas_cumprod[t] / cosine_diff.sqrt_one_minus_alphas_cumprod[t]
        print(f"   {t:4d}| {linear_snr.item():10.4f} | {cosine_snr.item():10.4f}")

    print("\n" + "=" * 70)
    print("Diffusion Forward Process 테스트 완료")
    print("=" * 70)
