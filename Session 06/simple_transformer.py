import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Simple positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SimpleAttention(nn.Module):
    """Scaled dot-product attention."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v), attn

class SimpleTransformerBlock(nn.Module):
    """Single transformer block with self-attention and FFN."""
    def __init__(self, d_model, n_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.attn = SimpleAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, attn_weights = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights

class SimpleTransformer(nn.Module):
    """Minimal decoder-only transformer for sequence modeling."""
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads, d_model*4, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        attn_weights = []
        for block in self.blocks:
            x, attn = block(x, mask)
            attn_weights.append(attn)
        x = self.norm(x)
        logits = self.head(x)
        return logits, attn_weights

# Example usage
if __name__ == "__main__":
    model = SimpleTransformer(vocab_size=10000, d_model=256, n_layers=4)
    x = torch.randint(0, 10000, (2, 10))  # batch_size=2, seq_len=10
    logits, attns = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print("Model ready for training!")
