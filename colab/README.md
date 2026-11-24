# Colab long-run and dashboard import guide

This folder contains `train_long.ipynb`, a notebook for running long GA
training jobs on Google Colab, and the steps for bringing the result back to
the local machine so `dashboard.html` shows the new run.

## 1. Get the patched code into Colab

The notebook assumes the repository contains the vectorised sensor path, the
resume-enabled runner and the `record_showcase` import option. Choose one:

- Push this working tree to your GitHub fork, then use the clone cell in the
  notebook. The repository URL in `pyproject.toml` is the current upstream.
- Or zip the project directory to your Google Drive, for example
  `/content/drive/MyDrive/Deep/project.zip`, and use the unzip cell.

Do not paste secrets, API keys or personal data into the notebook. Mounting
Drive gives the notebook access to your Drive files.

## 2. Run in Colab

1. Open `train_long.ipynb` in Colab and set Runtime > Change runtime type >
   Hardware accelerator to GPU if one is available.
2. Run the setup cells in order: mount Drive, get code, install dependencies,
   write config, recalibrate.
3. Run the training cell. It writes three things to the Drive run directory:
   - `state.npz` - full population, RNG and history state, every 50 generations
   - `history.csv` - one row per generation, appended in chunks
   - `best.npz` - the final best genome, in the format `neural.checkpoint`
     reads, with the generation count in its metadata
4. If the runtime disconnects, re-run the setup cells and then run the resume
   cell. It reloads the checkpoint and continues from the next generation.
   Do not run the first training cell again after a disconnect.

The free Colab tier can terminate a notebook and has a runtime ceiling of
about 12 hours. Checkpointing is what allows a longer total run: each chunk
costs at most one `save_interval` of recomputation.

### Config knobs

`evolution.generations` is the total target. `evolution.evaluation_steps`
and `population.size` dominate runtime, so reduce them if you need more
generations in the same wall-clock budget. Whenever either changes,
recalibrate `configs/calibration.json` with the cell in the notebook:
fitness scales were measured for a specific evaluation regime and the runner
refuses to invent constants.

For multi-seed data, run one notebook per seed with a different run directory
and a different fixed seed. Each run produces its own checkpoint and CSV.
Avoid running distributed workers on the free tier; sequential chunked runs
stay within Colab's rules.

## 3. Bring the result to the local machine

Copy these files out of the Drive run directory into the local repository:

```
experiments/EXP-HEADLESS-COLAB/
    stats_summary.json   <- rename of summary.json from Colab
    config.json          <- colab_config.json from Colab
    history.csv          <- optional, the raw per-generation rows
    best.npz             <- the exported best genome
```

`EXP-HEADLESS-COLAB` is discovered by the dashboard under the existing
"Headless smoke" family, so no dashboard code change is needed to list the
run. `stats_summary.json` is read directly by `webdash/build_static.py`.

## 4. Update the dashboard replay

The replay embedded in `dashboard.html` comes from
`experiments/EXP-SHOWCASE/recording.json`. To record the imported Colab brain
there, without retraining locally:

```bash
python experiments/record_showcase.py \
  --solo-only \
  --solo-genome experiments/EXP-HEADLESS-COLAB/best.npz \
  --solo-config experiments/EXP-HEADLESS-COLAB/config.json \
  --steps 150 \
  --solo-agents 10 \
  --seed 1 \
  --output experiments/EXP-SHOWCASE/recording.json

python webdash/build_static.py --experiments experiments --output dashboard.html
```

`--solo-only` skips the predator/prey training. `--solo-genome` loads the
exported `.npz` and replays it as the evolved solo brain; the untrained brain
is still generated from the same seed for the side-by-side comparison. The
generation label shown on the page comes from the checkpoint metadata.

If you want to keep the existing two-scenario showcase, omit `--solo-only`
and accept the local predator/prey training cost.

Open `dashboard.html` in a browser. It is self-contained and needs no server.
