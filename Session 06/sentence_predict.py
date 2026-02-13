import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

# === MODEL ===
class PositionalEncoding(nn.Module):
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
    def __init__(self, d_model, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
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
        B, T, C = x.shape
        
        q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        attn_out, attn_weights = SimpleAttention(self.d_k)(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.w_o(attn_out)
        
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4, max_len=128, dropout=0.1):
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
        self.seq_len = max_len

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

# === DATA ===
text = """
To be, or not to be: that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them? To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action.--Soft you now!
The fair Ophelia! Nymph, in thy orisons
Be all my sins remember'd.
""" * 5  # 5x repetition

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

class TextDataset(Dataset):
    def __init__(self, text, seq_len=64):
        self.data = []
        self.seq_len = seq_len
        for i in range(0, len(text) - seq_len, 3):
            seq = text[i:i+seq_len]
            target = text[i+1:i+seq_len+1]
            self.data.append((seq, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return (torch.tensor([char_to_idx[c] for c in seq], dtype=torch.long),
                torch.tensor([char_to_idx[c] for c in target], dtype=torch.long))

# === TRAINING ===
dataset = TextDataset(text)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SimpleTransformer(vocab_size)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

print(f"Vocab: {vocab_size}, Dataset: {len(dataset)} samples")
model.train()

for epoch in range(200):
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

# === GENERATION  ===
def generate_text(model, seed_text, length=100, temperature=0.85):
    model.eval()
    chars = [char_to_idx[c] for c in seed_text]
    generated = list(seed_text)
    seq_len = model.seq_len
    
    with torch.no_grad():
        for _ in range(length):
            context_list = chars[-seq_len:]
            if len(context_list) < seq_len:
                # Pad LEFT with zeros (beginning of sequence)
                padding = [0] * (seq_len - len(context_list))
                context_list = padding + context_list
            
            context = torch.tensor([context_list], dtype=torch.long)  # Shape: (1, seq_len)
            
            logits, _ = model(context)
            logits = logits[0, -1, :] / temperature  # Last position
            probs = F.softmax(logits, dim=-1)
            
            # Simple top-k sampling
            k = min(20, probs.size(0))
            top_k_probs, top_k_indices = torch.topk(probs, k)
            next_idx = torch.multinomial(top_k_probs, 1).item()
            next_char = idx_to_char[top_k_indices[next_idx].item()]
            
            generated.append(next_char)
            chars.append(top_k_indices[next_idx].item())
    
    return ''.join(generated)

print("\n" + "="*70)
print("SHAKESPEARE GENERATION:")
print(generate_text(model, "To be", 100))
print(generate_text(model, "What dreams", 100))
