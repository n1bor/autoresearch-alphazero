"""
Data loading and evaluation for AlphaZero chess training.
Imported by train.py — do not modify evaluate_loss.
"""

import os
import math
import random
import time
import datetime
import itertools
from concurrent.futures import ThreadPoolExecutor
import compress_pickle as pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class board_data(Dataset):
    """Single-file dataset: array of (board, policy, value) triples."""
    def __init__(self, dataset):  # dataset = np.array of (s, p, v)
        self.X        = dataset[:, 0]
        self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].transpose(2, 0, 1), self.y_p[idx], self.y_v[idx]


def _load_file(path):
    """Load a single .gz file and return numpy array, or None on error."""
    with open(path, 'rb') as fo:
        try:
            data = pickle.load(fo)
        except EOFError:
            return path, None
    return path, np.array(data, dtype="object")


class board_data_all(IterableDataset):
    """Infinite dataset: cycles through all .gz files in a directory forever.
    Loads up to `prefetch` files in parallel using a thread pool."""
    def __init__(self, directory, seed=42, prefetch=5):
        super().__init__()
        self.directory = directory
        self.rng = random.Random(seed)
        self.prefetch = prefetch

    def generate(self):
        files = sorted(os.path.join(self.directory, f)
                       for f in os.listdir(self.directory) if f.endswith('.gz'))
        assert files, f"No .gz files found in {self.directory}"
        epoch = 0
        with ThreadPoolExecutor(max_workers=self.prefetch) as executor:
            while True:
                epoch += 1
                shuffled = list(files)
                self.rng.shuffle(shuffled)
                print(f"\n[{ts()}][loader] epoch {epoch} — {len(shuffled)} files", flush=True)

                file_iter = iter(shuffled)
                # Seed the window with up to `prefetch` concurrent futures
                pending = [(f, executor.submit(_load_file, f))
                           for f in itertools.islice(file_iter, self.prefetch)]

                while pending:
                    file_path, future = pending.pop(0)
                    # Immediately queue the next file so loading overlaps with yielding
                    try:
                        next_f = next(file_iter)
                        pending.append((next_f, executor.submit(_load_file, next_f)))
                    except StopIteration:
                        pass

                    print(f"[{ts()}][loader] loading {os.path.basename(file_path)}", flush=True)
                    _, data = future.result()
                    if data is None:
                        print(f"[{ts()}][loader] EOFError in {file_path}, skipping", flush=True)
                        continue
                    print(f"[{ts()}][loader] loaded — {len(data)} records", flush=True)
                    file_loader = iter(DataLoader(board_data(data), shuffle=False, pin_memory=False))
                    while True:
                        item = next(file_loader, None)
                        if item is None:
                            break
                        yield (torch.squeeze(item[0]),
                               torch.squeeze(item[1]),
                               torch.squeeze(item[2]))

    def __iter__(self):
        return iter(self.generate())


def make_dataloader(train_path, batch_size, pin_memory=False):
    """Returns an infinite iterator yielding (state, policy, value) batches."""
    dataset = board_data_all(train_path)
    loader  = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=pin_memory)
    return iter(loader)


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — fixed metric)
# ---------------------------------------------------------------------------

TIME_BUDGET      = 1200  # training time in seconds (wall clock, excluding startup/compilation)
EVAL_SAMPLES     = 200_000 if torch.cuda.is_available() else 5_000
EVAL_BATCH_SIZE  = 2_000  # larger = faster eval; does not affect loss values

@torch.no_grad()
def evaluate_loss(net, device, batch_size, validate_path):  # batch_size unused; eval uses EVAL_BATCH_SIZE
    """
    Average AlphaLoss over EVAL_SAMPLES samples drawn from the validation set.
    Loss = MSE(value) + cross_entropy(policy).
    Returns NaN if no validation data is available.
    """
    files = sorted(os.path.join(validate_path, f)
                   for f in os.listdir(validate_path) if f.endswith('.gz'))
    if not files:
        print(f"[{ts()}][eval] no .gz files in {validate_path}, skipping", flush=True)
        return float('nan')

    random.Random(42).shuffle(files)
    print(f"[{ts()}][eval] evaluating ({EVAL_SAMPLES} samples)...", flush=True)

    net.eval()
    total_loss    = 0.0
    total_batches = 0
    total_samples = 0
    t_eval_start  = time.time()

    for file in files:
        if total_samples >= EVAL_SAMPLES:
            break
        with open(file, 'rb') as fo:
            try:
                data = pickle.load(fo)
            except EOFError:
                continue
        data   = np.array(data, dtype="object")
        loader = DataLoader(board_data(data), batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=False)
        for state, policy, value in loader:
            if total_samples >= EVAL_SAMPLES:
                break
            state  = state.to(device,  dtype=torch.float32)
            policy = policy.to(device, dtype=torch.float32)
            value  = value.to(device,  dtype=torch.float32)
            policy_pred, value_pred = net(state)
            value_error  = (value - value_pred[:, 0]) ** 2
            policy_error = torch.sum((-policy * (1e-6 + policy_pred.float()).float().log()), 1)
            total_loss    += (value_error.view(-1).float() + policy_error).mean().item()
            total_batches += 1
            total_samples += state.size(0)

    elapsed  = time.time() - t_eval_start
    val_loss = total_loss / total_batches if total_batches > 0 else float('nan')
    print(f"[{ts()}][eval] complete — val_loss: {val_loss:.4f} ({total_samples} samples, {total_batches} batches in {elapsed:.0f}s)", flush=True)
    return val_loss


# ---------------------------------------------------------------------------
# Main (smoke test — checks data is readable)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.expanduser("~"), "chess", "data", "trainOld")

    print(f"Checking data at {path}")
    files = [f for f in os.listdir(path) if f.endswith('.gz')]
    print(f"Found {len(files)} .gz files")
    if files:
        with open(os.path.join(path, files[0]), 'rb') as fo:
            data = pickle.load(fo)
        data = np.array(data, dtype="object")
        board, policy, value = data[0]
        print(f"Sample — board: {board.shape}, policy: {policy.shape}, value: {value}")
        print("Data OK.")
