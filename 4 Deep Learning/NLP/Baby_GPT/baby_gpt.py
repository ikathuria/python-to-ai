import torch
import numpy as np
import torch.nn as nn

from torch.nn import functional as F


torch.manual_seed(256)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 256  # N tokens in sequence
batch_size = 64
max_iters = 6000
eval_interval = 500
learning_rate = 0.0003
eval_iters = 300
vocab_size = 88  # 65

# every id for a given token is embedded to vector of this size
n_embd = 512
n_head = 8  # 8 attention heads
n_layer = 6  # 6 eoncoder layers
dropout = 0.2


text = ''

input_file2 = 'HuckFinn.txt'

with open(input_file2, 'r', encoding='utf-8') as f:
    text = f.read()


print("length of data in letter or characters")
len(text)


list(set(text))


the_chars = sorted(list(set(text)))

vocab_size = len(the_chars)  # 65

print(len(the_chars))

print(''.join(the_chars))

# The printed oputput
# !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz


stoi = {ch: i for i, ch in enumerate(the_chars)}
itos = {i: ch for i, ch in enumerate(the_chars)}


print(stoi)
print(itos)


def encode(s): return [stoi[c] for c in s]


encode("bahh")


def decode(l): return ''.join(itos[i] for i in l)


decode([55, 54, 61, 61])


data = torch.tensor(encode(text), dtype=torch.long)

print(data)


n = int(0.9*len(data))

train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+1+block_size] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y


temp_batch_size = 4
temp_block_size = 16

# select random starting points for the 4 sentences
ix = torch.randint(
    len(data) - block_size,
    (temp_batch_size,)
)

print(ix)


for index_temp in ix:
    print(data[index_temp])


x = torch.stack(
    [data[i: i + temp_block_size] for i in ix]

)

y = torch.stack(
    [data[i+1: i+1 + temp_block_size] for i in ix]
)

print(x)
print(y)


@torch.no_grad()  # for efficient processing
def estimate_loss():
    out = {}
    model.eval()  # set to no training
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # back to training
    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)  # [512, 64]
        self.query = nn.Linear(n_embd, head_size, bias=False)  # [512, 64]
        self.value = nn.Linear(n_embd, head_size, bias=False)  # [512, 64]

        tril_def = torch.tril(torch.ones(block_size, block_size))  # [40, 40]

        self.register_buffer(
            'tril',
            tril_def
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, E = x.shape  # [batch_size, 40, 512]

        k = self.key(x)  # k = (B, T, 64)
        q = self.query(x)  # q = (B, T, 64)

        E2 = 64  # I think this is 64 and not 512
        # (B, T, E) @ (B, E, T)  -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * E2 ** -0.5

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform weighted aggregation of values

        v = self.value(x)  # x = (B, 40, E)
        out = wei @ v  # (B, T, T) @ (B, T, 64) -> (B, T, 64)

        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):  # 512

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # [512, 4*512]
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # [4*512, 512]
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):  # (8, 64)
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # 512, 512
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):

    def __init__(self, n_embd, n_head):  # (512, 8)
        super().__init__()
        head_size = n_embd // n_head  # 64
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)  # 512
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd)  # [65, 512]
        self.pos_emb_table = nn.Embedding(block_size, n_embd)  # [block, 512]

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_ffw_head = nn.Linear(
            n_embd, vocab_size)  # [512, 65] # FFW Layer

    def forward(self, idx, targets=None):
        B, T = idx.shape  # (Batch, 40)
        # ids and targets are both (B, T) tensors of integers

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_emb_table(torch.arange(T, device=device))

        x = tok_emb + pos_emb  # [B, T, E] or [64, 40, 512]

        # This is the architecture
        x = self.blocks(x)  # (B, T, E)
        x = self.ln_f(x)  # (B, T, E)   ## norm
        logits = self.lm_ffw_head(x)  # [B, 40, 65]

        if targets is None:
            loss = None
        else:
            B, T, E = logits.shape
            logits = logits.view(B*T, E)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):  # idx is (B, T)
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)  # get preds
            logits = logits[:, -1, :]  # focus on last one (B, E)
            probs = F.softmax(logits, dim=-1)  # (B, E) get probs
            idx_next = torch.multinomial(
                probs, num_samples=1)  # (B, 1) selected
            # (B, T+1) append sample to running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = GPTModel()

m = model.to(device)

optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)


for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    # eval the loss
    logits, loss = m(xb, yb)

    optimizer.zero_grad(set_to_none=True)  # zero out
    loss.backward()
    optimizer.step()


# Starting token  id_sos = 0
sos_context = torch.zeros((1, 1),  dtype=torch.long, device=device)

generated_text = m.generate(sos_context, max_new_tokens=500)[0].tolist()

print(decode(generated_text))


sos_context = torch.ones((1, 1),  dtype=torch.long, device=device)

generated_text = m.generate(sos_context, max_new_tokens=500)[0].tolist()

print(decode(generated_text))


new_lst = encode("Where is Huck?")


new_np = np.array(new_lst)
new_np


new_context = torch.tensor(new_np, dtype=torch.long, device=device)


new_context = new_context.view((1, -1))
new_context


generated_text = m.generate(new_context, max_new_tokens=500)[0].tolist()

print(decode(generated_text))

new_context.shape


sos_context_tmp = torch.ones((1, 1),  dtype=torch.long, device=device)
sos_context_tmp.shape
