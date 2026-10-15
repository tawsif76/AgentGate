import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentGateConfig:
    def __init__(
        self,
        vocab_size=32_000,
        context_length=2_048,
        n_layers=24,
        n_heads=10,
        hidden_size=640,
        intermediate_size=1_792,
        rope_theta=10_000.0,
        rms_norm_eps=1e-5,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps

        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")

        self.head_dim = hidden_size // n_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")


def rotate_half(x):
    """[x1, x2] -> [-x2, x1], splitting the last dimension in half."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10_000.0):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len

        # RoPE operates on 2D coordinate pairs, so frequencies cover half the head dimension
        half_dim = dim // 2
        # exponents: [0, 1/half_dim, 2/half_dim, ..., (half_dim-1)/half_dim] -> shape: (half_dim,)
        exponents = torch.arange(half_dim, dtype=torch.float32) / half_dim
        # inv_freq: theta^(-2i / dim) decay factor per frequency pair -> shape: (half_dim,)
        inv_freq = 1.0 / (theta**exponents)

        # positions: [0, 1, 2, ..., max_seq_len - 1] -> shape: (max_seq_len,)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # freqs = positions outer-product inv_freq -> shape: (max_seq_len, half_dim)
        freqs = torch.outer(positions, inv_freq)
        # Duplicate frequencies across the feature dimension to match full dim -> shape: (max_seq_len, dim)
        angles = torch.cat((freqs, freqs), dim=-1)

        # Cache precomputed cos and sin tables as non-persistent buffers (saved in memory, not model weights)
        # shape of cached tensors: (max_seq_len, dim)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, q, k):
        # q and k shape: (B, H, T, D)
        # where B = batch_size, H = n_heads, T = seq_len, D = head_dim
        seq_len = q.shape[-2]  # Extracts T

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}"
            )

        # Slice the cache up to current sequence length T:
        # self.cos_cached[:seq_len] has shape: (T, D)
        # [None, None, :, :] reshapes to: (1, 1, T, D) for broadcasting across batch (B) and heads (H)
        cos = self.cos_cached[:seq_len].to(dtype=q.dtype)[None, None, :, :]
        sin = self.sin_cached[:seq_len].to(dtype=q.dtype)[None, None, :, :]

        # Both returned q and k retain their original shape: (B, H, T, D)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        # Fused projection producing Q, K, V for all heads in a single matmul
        # Weight shape: (3 * hidden_size, hidden_size)
        self.qkv_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=False
        )

        self.rope = RotaryPositionalEmbedding(
            config.head_dim, max_seq_len=config.context_length, theta=config.rope_theta
        )
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Inference-only instrumentation; no new parameters or checkpoint keys.
        self.inspect_attention = False
        self.last_attn_weights = None

    def forward(self, x):
        # Input x shape: (B, T, hidden_size)
        B, T, C = x.shape

        # Single fused projection: (B, T, hidden_size) -> (B, T, 3 * hidden_size)
        qkv = self.qkv_proj(x)

        # Split into q, k, v, each of shape: (B, T, hidden_size)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape into heads and move head dim next to batch:
        # .view(B, T, n_heads, head_dim) -> shape: (B, T, H, D)
        # .transpose(1, 2)               -> shape: (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Rotate Q and K by positional angles; broadcasting across B and H
        # Output shapes remain identical: (B, H, T, D)
        q, k = self.rope(q, k)

        # PyTorch scaled dot-product attention.
        # Uses an optimized CUDA backend when available.
        # Computes: softmax((Q @ K.T) / sqrt(D) + causal_mask) @ V
        # out shape: (B, H, T, D)
        if self.inspect_attention:
            # Match the causal SDPA operation, but expose attention probabilities.
            scores = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
            causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
            weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
            self.last_attn_weights = weights.detach()
            out = weights @ v
        else:
            self.last_attn_weights = None
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

        # Unpack tensor dimensions
        B, H, T, D = out.shape

        # Transpose heads and time:
        # out.transpose(1, 2)            -> shape: (B, T, H, D)
        # .contiguous()                  -> ensures memory layout is contiguous after swapping axes
        # .view(B, T, H * D)             -> flattens heads: (B, T, hidden_size)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)

        # Final projection: (B, T, hidden_size) -> (B, T, hidden_size)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x):
        # Input x shape: (B, T, hidden_size)
        # self.gate_proj(x):                 (B, T, intermediate_size)
        # F.silu(self.gate_proj(x)):          (B, T, intermediate_size)
        # self.up_proj(x):                   (B, T, intermediate_size)
        # Element-wise product (*):          (B, T, intermediate_size)
        # self.down_proj(...):               (B, T, hidden_size)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Pre-attention normalization: scales along hidden_size dimension
        # weight shape: (hidden_size,)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MultiHeadAttention(config)

        # Pre-feedforward normalization: scales along hidden_size dimension
        # weight shape: (hidden_size,)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x):
        # Input x shape: (B, T, hidden_size)

        # Pre-LN Self-Attention with residual connection:
        # self.attn_norm(x):                 (B, T, hidden_size)
        # self.attn(...):                    (B, T, hidden_size)
        # x + ...:                           (B, T, hidden_size)
        x = x + self.attn(self.attn_norm(x))

        # Pre-LN SwiGLU Feedforward with residual connection:
        # self.ffn_norm(x):                  (B, T, hidden_size)
        # self.ffn(...):                     (B, T, hidden_size)
        # x + ...:                           (B, T, hidden_size)
        x = x + self.ffn(self.ffn_norm(x))

        # Output shape: (B, T, hidden_size)
        return x


class AgentGate(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        self.config = config if config is not None else AgentGateConfig()

        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )

        # Sequence of n_layers transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )

        # Final layer normalization applied to transformer trunk output
        # weight shape: (hidden_size,)
        self.norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # Language modeling head mapping hidden states to vocabulary logits
        # Weight shape: (vocab_size, hidden_size)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

        self.apply(self._init_weights)

        # Scaled standard deviation for residual projection layers at initialization
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)

        for block in self.blocks:
            # block.attn.proj.weight shape:     (hidden_size, hidden_size)
            # block.ffn.down_proj.weight shape: (hidden_size, intermediate_size)
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=residual_std)

        # Weight tying: share parameters between token embedding and classification head
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None):
        # tokens shape: (B, T) where B = batch_size, T = seq_len
        # targets shape (if provided): (B, T)
        _, T = tokens.shape

        if T > self.config.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds context length "
                f"{self.config.context_length}"
            )

        # Embedding lookup: (B, T) -> (B, T, hidden_size)
        x = self.token_embedding(tokens)

        # Pass representations through all N transformer blocks
        # x maintains shape: (B, T, hidden_size) throughout
        for block in self.blocks:
            x = block(x)

        # Final RMSNorm: (B, T, hidden_size) -> (B, T, hidden_size)
        x = self.norm(x)

        # Project to vocabulary dimension:
        # logits shape: (B, T, vocab_size)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            # Flatten predictions and ground-truth tokens across batch and time:
            # logits.reshape(-1, self.config.vocab_size) -> shape: (B * T, vocab_size)
            # targets.reshape(-1)                         -> shape: (B * T,)
            # loss is a scalar tensor                     -> shape: ()
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1)
            )

        # logits shape: (B, T, vocab_size)
        # loss shape:   () if targets is not None else None
        return logits, loss

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=0.0, top_k=None):
        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.context_length :]

            logits, _ = self(context)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            else:
                logits = logits / temperature

                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat((tokens, next_token), dim=1)

        return tokens


if __name__ == "__main__":
    model = AgentGate()
    print(model.num_parameters())
