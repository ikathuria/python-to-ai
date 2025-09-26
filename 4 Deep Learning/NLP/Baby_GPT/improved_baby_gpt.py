import torch
import numpy as np
import torch.nn as nn
import os
import json
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, Any


class BabyGPTConfig:
    """Configuration class for Baby GPT model"""

    def __init__(
        self,
        vocab_size: int = 65,
        block_size: int = 40,
        n_embd: int = 512,
        n_head: int = 8,
        n_layer: int = 6,
        dropout: float = 0.2,
        batch_size: int = 64,
        learning_rate: float = 0.0003,
        max_iters: int = 6000,
        eval_interval: int = 500,
        eval_iters: int = 300,
        seed: int = 256
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.seed = seed

        # Validation
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"


class TextProcessor:
    """Handles text encoding/decoding and vocabulary management"""

    def __init__(self, text: str):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        """Encode string to list of integers"""
        try:
            return [self.stoi[c] for c in s]
        except KeyError as e:
            raise ValueError(f"Character {e} not in vocabulary")

    def decode(self, tokens: list[int]) -> str:
        """Decode list of integers to string"""
        try:
            return ''.join(self.itos[i] for i in tokens)
        except KeyError as e:
            raise ValueError(f"Token {e} not in vocabulary")

    def save_vocab(self, path: str):
        """Save vocabulary to file"""
        vocab_data = {
            'chars': self.chars,
            'stoi': self.stoi,
            'itos': {str(k): v for k, v in self.itos.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2)

    @classmethod
    def load_vocab(cls, path: str):
        """Load vocabulary from file"""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        processor = cls.__new__(cls)
        processor.chars = vocab_data['chars']
        processor.vocab_size = len(processor.chars)
        processor.stoi = vocab_data['stoi']
        processor.itos = {int(k): v for k, v in vocab_data['itos'].items()}
        return processor


class DataLoader:
    """Handles data loading and batching"""

    def __init__(self, data: torch.Tensor, config: BabyGPTConfig, device: str):
        self.config = config
        self.device = device

        # Split data
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        print(f"Dataset size: {len(data):,} tokens")
        print(f"Training set: {len(self.train_data):,} tokens")
        print(f"Validation set: {len(self.val_data):,} tokens")

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data"""
        data = self.train_data if split == "train" else self.val_data

        if len(data) <= self.config.block_size:
            raise ValueError(
                f"Data too small for block_size {self.config.block_size}")

        ix = torch.randint(len(data) - self.config.block_size,
                           (self.config.batch_size,))

        x = torch.stack([data[i:i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.config.block_size]
                        for i in ix])

        return x.to(self.device), y.to(self.device)


class Head(nn.Module):
    """Single attention head"""

    def __init__(self, config: BabyGPTConfig):
        super().__init__()
        self.head_size = config.n_embd // config.n_head

        self.key = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, self.head_size, bias=False)

        self.register_buffer(
            'tril',
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

        self.dropout = nn.Dropout(config.dropout)
        self.scale = self.head_size ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, E = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Attention weights
        wei = q @ k.transpose(-2, -1) * self.scale  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Weighted aggregation
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""

    def __init__(self, config: BabyGPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config)
                                   for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed forward network"""

    def __init__(self, config: BabyGPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # Changed from ReLU to GELU for better performance
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block with pre-norm"""

    def __init__(self, config: BabyGPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))  # Pre-norm
        x = x + self.ffwd(self.ln2(x))
        return x


class BabyGPT(nn.Module):
    """Baby GPT model"""

    def __init__(self, config: BabyGPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding_table = nn.Embedding(
            config.vocab_size, config.n_embd)
        self.pos_emb_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model has {n_params:,} parameters")

    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape

        # Embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.pos_emb_table(torch.arange(
            T, device=idx.device))  # (T, n_embd)

        x = tok_emb + pos_emb  # (B, T, n_embd)

        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate tokens with optional temperature and top-k sampling"""
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to block_size
                idx_cond = idx[:, -self.config.block_size:]

                # Get predictions
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature  # (B, vocab_size)

                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                # Append
                idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx

    def save_checkpoint(self, path: str, optimizer: torch.optim.Optimizer, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config.__dict__,
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        config = BabyGPTConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint


class Trainer:
    """Training manager"""

    def __init__(self, model: BabyGPT, dataloader: DataLoader, config: BabyGPTConfig, device: str):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate)

        # Track training metrics
        self.train_losses = []
        self.val_losses = []

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Estimate loss on train and validation sets"""
        out = {}
        self.model.eval()

        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.dataloader.get_batch(split)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()

        self.model.train()
        return out

    def train(self, save_checkpoints: bool = True, checkpoint_dir: str = "checkpoints"):
        """Train the model"""
        if save_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)

        print("Starting training...")

        for iter_num in range(self.config.max_iters):
            # Evaluation
            if iter_num % self.config.eval_interval == 0:
                losses = self.estimate_loss()
                print(
                    f"Step {iter_num:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                self.train_losses.append(losses['train'])
                self.val_losses.append(losses['val'])

                # Save checkpoint
                if save_checkpoints and iter_num > 0:
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f"checkpoint_{iter_num}.pt")
                    self.model.save_checkpoint(
                        checkpoint_path, self.optimizer, iter_num, losses['train'])

            # Training step
            xb, yb = self.dataloader.get_batch('train')
            _, loss = self.model(xb, yb)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        print("Training completed!")


def main():
    """Main function to run the training"""
    # Set device and seed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Configuration
    config = BabyGPTConfig(seed=256)
    torch.manual_seed(config.seed)

    # Load and process text
    input_file = './data/movie_data.txt'

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found")

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Text length: {len(text):,} characters")

    # Process text
    processor = TextProcessor(text)
    config.vocab_size = processor.vocab_size

    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Unique characters: {''.join(processor.chars)}")

    # Save vocabulary
    processor.save_vocab("vocab.json")

    # Prepare data
    data = torch.tensor(processor.encode(text), dtype=torch.long)
    dataloader = DataLoader(data, config, device)

    # Create model and trainer
    model = BabyGPT(config).to(device)
    trainer = Trainer(model, dataloader, config, device)

    # Train
    trainer.train()

    # Generate some text
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("="*50)

    # Start with random context
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(
        context, max_new_tokens=500, temperature=0.8, top_k=100)
    print(processor.decode(generated[0].tolist()))

    # Start with a prompt
    prompt = "Where is Huck?"
    context = torch.tensor(processor.encode(
        prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(
        context, max_new_tokens=500, temperature=0.8, top_k=100)
    print(f"\nPrompt: {prompt}")
    print(processor.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
