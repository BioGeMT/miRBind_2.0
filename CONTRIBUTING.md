# Contributing to miRBind 2.0

## Repository structure

- `code/` — plain, reusable scripts: encoders, model definitions, metrics, plotting. Each script should be runnable on its own.
- `analysis/` — one folder per analysis (e.g. *training Random Forest on K-mer encoding of Manakov 1:1*). Each folder contains a master script that reproduces the analysis end-to-end. Use the folder as a playground for hyperparameter search or architecture experiments, but the master script should run only the final pipeline. Intermediate outputs (encoded datasets, prediction files) can live here locally — don't commit them. Keep these scripts thin: they should orchestrate code from `code/`, not duplicate it. Add a `requirements.txt` or `environment.yaml`.
- `data/` — placeholder. Locally holds the datasets; on GitHub it only ships the download scripts.
- `models/` — trained model checkpoints. Mostly populated via download scripts.

## Workflow

Create a new branch, work there, clean it up, then open a PR against `main`.

### Adding a shared utility

Put it in `code/`. Make it a plain Python script that runs on its own and is reusable.

### Creating a new analysis

Create a folder under `analysis/` with a descriptive name. Experiment freely, but end with a master script that runs the final pipeline with fixed settings and produces a single consistent result.

### Changing a shared utility, dataset, or model

If you change something in `code/` (or in a shared dataset / model), find all usages in `analysis/` and either update them, rerun them, or flag them (e.g. open an issue) so downstream results don't silently go stale.

## Adding a model to the leaderboard

1. Push the code to reproduce and evaluate the model to GitHub.
2. Create a folder for the trained model in [Google Drive](https://drive.google.com/drive/folders/1IH7_CjxWW7Q0dKEFJY3L3yo4B2WWxJh2?usp=drive_link) and upload it.
3. Add a row to the leaderboard in [README.md](README.md) with metrics, model link, and code link.