"""
AlphaZero chess training script. No CLI flags needed.
Usage: uv run train.py
"""

import os
import random
import time
import shutil
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from prepare import make_dataloader, evaluate_loss, ts, TIME_BUDGET


# ---------------------------------------------------------------------------
# ChessNet architecture
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 22, 8, 8)
        return F.relu(self.bn1(self.conv1(s)))


class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class OutBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Value head
        self.conv  = nn.Conv2d(256, 1, kernel_size=1)
        self.bn    = nn.BatchNorm2d(1)
        self.fc1   = nn.Linear(8 * 8, 64)
        self.fc2   = nn.Linear(64, 1)
        # Policy head
        self.conv1      = nn.Conv2d(256, 128, kernel_size=1)
        self.bn1        = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc         = nn.Linear(8 * 8 * 128, 8 * 8 * 73)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))
        v = v.view(-1, 8 * 8)
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))
        p = F.relu(self.bn1(self.conv1(s)))
        p = p.view(-1, 8 * 8 * 128)
        p = self.logsoftmax(self.fc(p)).exp()
        return p, v


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        return self.outblock(s)


class AlphaLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error  = (value - y_value) ** 2
        policy_error = torch.sum((-policy * (1e-6 + y_policy.float()).float().log()), 1)
        return (value_error.view(-1).float() + policy_error).mean()


def init_weights(net):
    """
    Initialise weights for ChessNet:
      - Conv2d: Kaiming normal (He init), suited for ReLU activations
      - BatchNorm2d: weight=1, bias=0 (standard identity initialisation)
      - Linear (hidden): Kaiming normal for ReLU
      - Linear (value output, fc2): Xavier uniform, suited for tanh output
    """
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if name.endswith('fc2'):  # value head output → tanh
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

ROOT_DIR     = '/home/owensr/chess'
TRAIN_DIR    = os.path.join(ROOT_DIR, 'data', 'trainOld')
VALIDATE_DIR = os.path.join(ROOT_DIR, 'data', 'validate')
MODEL_DIR    = os.path.join(ROOT_DIR, 'data', 'model_data')

LR           = 0.0003
BATCH_SIZE   = 256

RUN_ID       = 1     # included in saved model filename

# ---------------------------------------------------------------------------
# Setup: device, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()

random.seed(42)
torch.manual_seed(42)

cuda   = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(f"[{ts()}] Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""), flush=True)

net = ChessNet().to(device)
init_weights(net)
print(f"[{ts()}] Weights initialised (Kaiming/Xavier).", flush=True)

if cuda:
    print(f"[{ts()}] Compiling model with torch.compile()...", flush=True)
    net = torch.compile(net)
    print(f"[{ts()}] Compiled.", flush=True)

num_params = sum(p.numel() for p in net.parameters())
print(f"[{ts()}] Parameters: {num_params/1e6:.1f}M", flush=True)

criterion  = AlphaLoss().to(device)
optimizer  = optim.Adam(net.parameters(), lr=LR)
train_iter = make_dataloader(TRAIN_DIR, BATCH_SIZE, pin_memory=cuda)

state, policy, value = next(train_iter)  # prefetch first batch

print(f"[{ts()}] Time budget: {TIME_BUDGET}s  batch_size: {BATCH_SIZE}  lr: {LR}", flush=True)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training  = time.time()
total_training_time = 0.0
total_records     = 0
roll_9            = 7.0
roll_99           = 7.0
step              = 0

net.train()

while True:
    t0 = time.time()

    state  = state.to(device,  dtype=torch.float32)
    policy = policy.to(device, dtype=torch.float32)
    value  = value.to(device,  dtype=torch.float32)

    optimizer.zero_grad()
    policy_pred, value_pred = net(state)
    loss = criterion(value_pred[:, 0], value, policy_pred, policy)
    loss.backward()
    optimizer.step()

    loss_f = loss.item()
    roll_9  = 0.9  * roll_9  + 0.1  * loss_f
    roll_99 = 0.99 * roll_99 + 0.01 * loss_f
    total_records += state.size(0)

    state, policy, value = next(train_iter)  # prefetch next batch

    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    pct_done  = 100 * min(total_training_time / TIME_BUDGET, 1.0)
    remaining = max(0, TIME_BUDGET - total_training_time)
    print(f"\r[{ts()}] step {step:05d} ({pct_done:.1f}%) | loss: {loss_f:.4f} | l9: {roll_9:.4f} | l99: {roll_99:.4f} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r

# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------

net.eval()
out_file = os.path.join(MODEL_DIR,
    f"model_{RUN_ID}_loss{roll_99:.2f}_{datetime.datetime.today().strftime('%Y-%m-%d-%H%M%S')}.gz")
torch.save({'state_dict': net.state_dict()}, out_file)
shutil.copy(out_file, os.path.join(MODEL_DIR, "latest.gz"))
print(f"[{ts()}] Saved to {out_file}", flush=True)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print(f"[{ts()}] Evaluating on validation set...", flush=True)
val_loss = evaluate_loss(net, device, BATCH_SIZE, VALIDATE_DIR)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

t_end = time.time()
records_per_sec = int(total_records / total_training_time) if total_training_time > 0 else 0
peak_vram_mb    = torch.cuda.max_memory_allocated() / 1024 / 1024 if cuda else 0

print("---")
print(f"val_loss:         {val_loss:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"records_per_sec:  {records_per_sec}")
print(f"total_records_M:  {total_records / 1e6:.3f}")
print(f"num_batches:      {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"device:           {device}")
