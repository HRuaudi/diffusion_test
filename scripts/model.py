# model.py - Denoiser Models for Conditional Diffusion
import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Timestep embedding using sinusoidal functions"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class MLPDenoiser(nn.Module):
    """
    Simple MLP-based Denoiser for Conditional Diffusion

    Architecture:
    - Condition encoder: condition (47) -> hidden
    - Time encoder: timestep -> hidden
    - Noise predictor: [x_t, cond, time] -> noise prediction

    Parameters
    ----------
    seq_len : int
        Fixed sequence length
    n_features : int
        Number of output features (9)
    cond_dim : int
        Condition dimension (47)
    hidden_dim : int
        Hidden layer dimension
    time_dim : int
        Timestep embedding dimension
    n_layers : int
        Number of hidden layers
    """

    def __init__(
        self,
        seq_len=512,
        n_features=9,
        cond_dim=47,
        hidden_dim=512,
        time_dim=128,
        n_layers=4
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.flat_dim = seq_len * n_features

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Condition encoder
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Main network: [flattened x_t, time_embed, cond_embed] -> noise
        input_dim = self.flat_dim + hidden_dim * 2

        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ])
        layers.append(nn.Linear(hidden_dim, self.flat_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x_t, t, condition):
        """
        Parameters
        ----------
        x_t : torch.Tensor, shape (B, T, D)
            Noisy input at timestep t
        t : torch.Tensor, shape (B,)
            Diffusion timestep
        condition : torch.Tensor, shape (B, C)
            Conditioning input

        Returns
        -------
        noise_pred : torch.Tensor, shape (B, T, D)
            Predicted noise
        """
        B = x_t.shape[0]

        # Flatten x_t
        x_flat = x_t.view(B, -1)  # (B, T*D)

        # Embeddings
        t_emb = self.time_embed(t.float())       # (B, hidden)
        c_emb = self.cond_encoder(condition)      # (B, hidden)

        # Concatenate
        h = torch.cat([x_flat, t_emb, c_emb], dim=-1)

        # Predict noise
        noise_flat = self.network(h)  # (B, T*D)
        noise_pred = noise_flat.view(B, self.seq_len, self.n_features)

        return noise_pred


class ConvDenoiser(nn.Module):
    """
    1D Convolutional Denoiser for better temporal modeling

    Architecture:
    - Time + Condition embedding -> broadcast to sequence
    - Stack of 1D Conv layers with residual connections
    """

    def __init__(
        self,
        seq_len=512,
        n_features=9,
        cond_dim=47,
        hidden_dim=128,
        time_dim=64,
        n_layers=6
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU()
        )

        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU()
        )

        # Input projection: [n_features + hidden*2] -> hidden
        self.input_proj = nn.Conv1d(n_features + hidden_dim * 2, hidden_dim, 1)

        # Conv layers with residual
        self.conv_layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 4)
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, 3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(8, hidden_dim),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.GELU()
                )
            )

        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, n_features, 1)

    def forward(self, x_t, t, condition):
        """
        Parameters
        ----------
        x_t : torch.Tensor, shape (B, T, D)
        t : torch.Tensor, shape (B,)
        condition : torch.Tensor, shape (B, C)

        Returns
        -------
        noise_pred : torch.Tensor, shape (B, T, D)
        """
        B, T, D = x_t.shape

        # Embeddings
        t_emb = self.time_embed(t.float())       # (B, hidden)
        c_emb = self.cond_embed(condition)        # (B, hidden)

        # Broadcast to sequence length
        t_emb = t_emb[:, :, None].expand(-1, -1, T)  # (B, hidden, T)
        c_emb = c_emb[:, :, None].expand(-1, -1, T)  # (B, hidden, T)

        # Concatenate with input
        x = x_t.permute(0, 2, 1)  # (B, D, T)
        x = torch.cat([x, t_emb, c_emb], dim=1)  # (B, D+hidden*2, T)

        # Input projection
        h = self.input_proj(x)  # (B, hidden, T)

        # Conv layers with residual
        for conv in self.conv_layers:
            h = h + conv(h)

        # Output projection
        out = self.output_proj(h)  # (B, D, T)
        out = out.permute(0, 2, 1)  # (B, T, D)

        return out


# =============================================================================
# 테스트
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Denoiser Model 테스트")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 테스트 데이터
    batch_size = 4
    seq_len = 512
    n_features = 9
    cond_dim = 47

    x_t = torch.randn(batch_size, seq_len, n_features, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    condition = torch.randn(batch_size, cond_dim, device=device)

    print(f"\n[1] MLP Denoiser 테스트")
    mlp_model = MLPDenoiser(
        seq_len=seq_len,
        n_features=n_features,
        cond_dim=cond_dim,
        hidden_dim=256,
        n_layers=3
    ).to(device)

    n_params = sum(p.numel() for p in mlp_model.parameters())
    print(f"   Parameters: {n_params:,}")

    noise_pred = mlp_model(x_t, t, condition)
    print(f"   Input shape: {x_t.shape}")
    print(f"   Output shape: {noise_pred.shape}")

    print(f"\n[2] Conv Denoiser 테스트")
    conv_model = ConvDenoiser(
        seq_len=seq_len,
        n_features=n_features,
        cond_dim=cond_dim,
        hidden_dim=128,
        n_layers=6
    ).to(device)

    n_params = sum(p.numel() for p in conv_model.parameters())
    print(f"   Parameters: {n_params:,}")

    noise_pred = conv_model(x_t, t, condition)
    print(f"   Input shape: {x_t.shape}")
    print(f"   Output shape: {noise_pred.shape}")

    print(f"\n[3] 메모리 사용량 (Conv)")
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        for _ in range(10):
            noise_pred = conv_model(x_t, t, condition)
            loss = noise_pred.mean()
            loss.backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"   Peak memory: {peak_mem:.1f} MB")
    else:
        print("   (CUDA not available)")

    print("\n" + "=" * 70)
    print("Denoiser Model 테스트 완료")
    print("=" * 70)
