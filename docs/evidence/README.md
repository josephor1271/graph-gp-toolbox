# Compact measured evidence

These are outputs from the new repository, not legacy benchmark results.
Each main study has 108 fits: 18 datasets × two optimizer starts × three methods.
The two amplitude studies use the same datasets and splits.

All fit rows, numerical sweeps, summaries, configurations, versions and the first
dataset's exact likelihood landscape are retained. Trajectories here include only
the first dataset and first initialization (all three methods), sufficient to
regenerate the three published figures. Full trajectories, every synthetic array
and prediction array remain in local gitignored results/ and are reproducible
with the experiment commands. The compact bundles are not the full raw archive.

To regenerate figures from this evidence bundle without retraining:

```bash
python experiments/01_heat_hyperparameter_recovery.py --plot-only --output docs/evidence/heat_recovery
python experiments/01_heat_hyperparameter_recovery.py --plot-only --output docs/evidence/free_amplitude
```

Plot-only writes figures into the selected evidence directory; the published
copies are under docs/figures/. Fit summaries choose the best exact-NLL start
per method and dataset. Starts are not counted as independent GP draws.

The optimizer audit retains its 12 fit rows and configuration here. Its complete
1,000-step trajectories remain in local results/optimizer_audit/ and are
reproducible with experiments/01_optimizer_audit.py.

CSV line endings are normalized to LF. One upstream PyTorch build-machine path
in an audit deprecation warning is replaced with <upstream-pytorch-build>; the
warning text and all numerical data are otherwise retained.
