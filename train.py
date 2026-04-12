"""
AlphaZero chess training script. No CLI flags needed.
Usage: uv run train.py
"""

import os
import math
import random
import time
import shutil
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from prepare import make_dataloader, evaluate_loss, ts, TIME_BUDGET, EVAL_SAMPLES, EVAL_BATCH_SIZE


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
    def __init__(self, planes=256, bottleneck=64):
        super().__init__()
        self.conv1 = nn.Conv2d(planes, bottleneck, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(bottleneck)
        self.conv2 = nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(bottleneck)
        self.conv3 = nn.Conv2d(bottleneck, planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
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
        self.conv1      = nn.Conv2d(256, 64, kernel_size=1)
        self.bn1        = nn.BatchNorm2d(64)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc         = nn.Linear(8 * 8 * 64, 8 * 8 * 73)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))
        v = v.view(-1, 8 * 8)
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))
        p = F.relu(self.bn1(self.conv1(s)))
        p = p.view(-1, 8 * 8 * 64)
        p = self.logsoftmax(self.fc(p)).exp()
        return p, v


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock()
        for block in range(7):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(7):
            s = getattr(self, "res_%i" % block)(s)
        return self.outblock(s)


class AlphaLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error  = (value - y_value) ** 2
        policy_error = torch.sum((-policy * (1e-6 + y_policy.float()).float().log()), 1)
        return (value_error.view(-1).float() + policy_error).mean()


def save_weight_stats(net, path):
    """
    Analyse the trained model weights and write a summary to path.

    Per parameter:
      mean, std, min, max, L2 norm, % near-zero (|w| < 1e-3),
      and gradient mean/std from the last backward pass (if available).

    Flags layers that may indicate training problems:
      - std < 0.001  (weights collapsed / not learning)
      - std > 1.0    (weights exploded)
      - near_zero%  > 50  (mostly dead)
      - grad_std > 0.1    (large gradient variance)
    """
    # Access underlying module if torch.compile() was used
    module = getattr(net, '_orig_mod', net)

    rows   = []
    issues = []

    for name, param in module.named_parameters():
        w    = param.detach().float()
        mean = w.mean().item()
        std  = w.std().item()
        wmin = w.min().item()
        wmax = w.max().item()
        l2   = w.norm(2).item()
        nz   = (w.abs() < 1e-3).float().mean().item() * 100

        grad_mean = grad_std = float('nan')
        if param.grad is not None:
            g         = param.grad.detach().float()
            grad_mean = g.mean().item()
            grad_std  = g.std().item()

        rows.append((name, tuple(param.shape), mean, std, wmin, wmax, l2, nz, grad_mean, grad_std))

        flags = []
        if std < 0.001:                       flags.append('collapsed')
        if std > 1.0:                         flags.append('exploded')
        if nz > 50:                           flags.append(f'dead({nz:.0f}%)')
        if not math.isnan(grad_std) and grad_std > 0.1:
                                              flags.append(f'large_grad_std({grad_std:.3f})')
        if flags:
            issues.append((name, flags))

    with open(path, 'w') as f:
        f.write(f"Weight statistics\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Header
        f.write(f"{'Name':<45} {'Shape':<20} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9} "
                f"{'L2':>9} {'Dead%':>6} {'GradMean':>10} {'GradStd':>10}\n")
        f.write("-" * 145 + "\n")

        for name, shape, mean, std, wmin, wmax, l2, nz, gm, gs in rows:
            shape_str = str(shape)
            gm_str = f"{gm:10.4f}" if not math.isnan(gm) else f"{'n/a':>10}"
            gs_str = f"{gs:10.4f}" if not math.isnan(gs) else f"{'n/a':>10}"
            f.write(f"{name:<45} {shape_str:<20} {mean:9.4f} {std:9.4f} {wmin:9.4f} {wmax:9.4f} "
                    f"{l2:9.2f} {nz:6.1f} {gm_str} {gs_str}\n")

        # Summary by layer type
        def layer_rows(keyword):
            return [r for r in rows if keyword in r[0]]

        f.write("\n=== Summary by layer type ===\n")
        for label, keyword in [("Conv weights", ".conv"), ("BN weights", ".bn"),
                                ("Linear weights", ".fc")]:
            subset = layer_rows(keyword)
            if subset:
                stds = [r[3] for r in subset]
                l2s  = [r[6] for r in subset]
                f.write(f"  {label:<20}  n={len(subset):3d}  mean_std={sum(stds)/len(stds):.4f}  "
                        f"mean_l2={sum(l2s)/len(l2s):.2f}\n")

        # Flagged layers
        f.write("\n=== Flagged layers ===\n")
        if issues:
            for name, flags in issues:
                f.write(f"  {name:<45}  {', '.join(flags)}\n")
        else:
            f.write("  None\n")

    print(f"[{ts()}] Weight stats written to {path}", flush=True)


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

WARMUP_STEPS = 5
T_MAX        = 360  # approximate total steps for cosine decay (matches observed 359 steps)

criterion  = AlphaLoss().to(device)
optimizer  = optim.AdamW(net.parameters(), lr=LR, weight_decay=0.001)
scheduler  = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
    optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_STEPS),
    optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX - WARMUP_STEPS, eta_min=LR * 0.1),
], milestones=[WARMUP_STEPS])
train_iter = make_dataloader(TRAIN_DIR, BATCH_SIZE, pin_memory=cuda)

state, policy, value = next(train_iter)  # prefetch first batch

print(f"[{ts()}] Time budget: {TIME_BUDGET}s  batch_size: {BATCH_SIZE}  lr: {LR}  eval_samples: {EVAL_SAMPLES}  eval_batch_size: {EVAL_BATCH_SIZE}", flush=True)

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
    scheduler.step()

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
# Weight statistics
# ---------------------------------------------------------------------------

save_weight_stats(net, "weight_stats.txt")

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
