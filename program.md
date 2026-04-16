# autoresearch-alphazero

This is an experiment to have the LLM do its own research on AlphaZero chess training.

## Introduction

There are 6 stages to follow. These are:

1. Setup - does the initial setup of the experiment.
2. Baseline - this is to run a baseline test of the model and get initial statistics.

These steps are done in a loop continuously.
3. Analysis and planning - in this step you should analyse the data in weights_stats.txt and write a plan in plan.md of the changes you will make in this loop.
4. Coding - update train.py with the changes proposed in the plan.md file.
5. Run the experiment - this is where you run the train.py script.
6. Read out the results - extract the results from run.log.
7. Update git. Either accept the change or revert the commit if it did not help.
 

## 1. Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr11`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current branch.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed dataloader, evaluation, and data utilities. Do not modify.
   - `train.py` — the file you modify. Network architecture, optimizer, training loop.
4. **Verify data exists**: Check that `/home/owensr/chess/data/trainOld/` contains `.gz` training files and `/home/owensr/chess/data/validate/` contains `.gz` validation files. Check that `/home/owensr/chess/data/model_data/random.gz` exists.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, proceed to Baseline.

## 2. Baseline

Run a baseline execution of the script:
```
uv run train.py > run.log 2>&1
```
Then read out the results and update `results.tsv` with the baseline results.


## 3. Analysis and planning


**The goal is simple: get the lowest val_loss.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. Some increase is acceptable for meaningful val_loss gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_loss improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_loss improvement from deleting code? Definitely keep.


In this step you will look in `run.log` for any errors or warnings.
You will look in `weights_stats.txt` for statistics on the final weights of the model.
You will then create a new `plan_<commitID>.md` file and populate this with you reasoning as to how we could reduce the val_loss by updating the code.
Also include ideas in this document for how the weights_stats.txt file could be enhanced to improve the data you have to do the analysis next time through the loop.
Only include one idea at a time in this plan. Always the one you think is the best. You can try other ideas in later loops.

## 4. Coding

**What you CAN do:**
- Make changes to train.py to implement the fix. Everything is fair game: network architecture, optimizer, hyperparameters, batch size, number of residual blocks, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation harness, dataloader, and data utilities.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_loss` function in `prepare.py` is the ground truth metric.

## 5. Run the experiment

Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)

**Timeout**: Each experiment should take ~6 minutes total (5 min training + 1 min eval + startup overhead). If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

## 6. Read out the results

Read out the results: `grep "^val_loss:\|^peak_vram_mb:" run.log`

If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.

Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)

## 7. Update git

If val_loss improved (lower), you "advance" the branch, keeping the git commit

else If val_loss is equal or worse, you git reset back to where you started

# Other Information

## Output format

Once the script finishes it prints a summary like this:

```
---
val_loss:         7.433287
training_seconds: 304.0
total_seconds:    454.2
peak_vram_mb:     0.0
records_per_sec:  37
total_records_M:  0.012
num_batches:      43
num_params_M:     60.8
device:           cpu
```

You can extract the key metric from the log file:

```
grep "^val_loss:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
date commit	val_loss	memory_gb	status	description
```

1. date experment finished
2. git commit hash (short, 7 chars)
3. val_loss achieved (e.g. 7.433287) — use 0.000000 for crashes
4. peak memory in GB, round to .1f (e.g. 1.2 — divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
date    commit	val_loss	memory_gb	status	description
Thu Apr 16 14:43:13 CEST 2026   bde8fa3	7.433287	0.0	keep	baseline
Thu Apr 16 14:43:13 CEST 2026   b2c3d4e	7.312000	0.1	keep	reduce residual blocks to 10
Thu Apr 16 14:43:13 CEST 2026  c3d4e5f	7.501000	0.0	discard	switch to SGD optimizer
Thu Apr 16 14:43:13 CEST 2026   d4e5f6g	0.000000	0.0	crash	double conv filters (OOM)
```
