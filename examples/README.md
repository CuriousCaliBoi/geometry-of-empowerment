# Explore the doorway experiment

Run from the repository root with the locked environment:

```bash
uv sync --locked --extra dev
uv run python examples/explore_mutual_information.py --out figures/mi-lab
uv run python make_all_figures.py 4 5 6
```

The example uses the upstream occupancy LP and Blahut-Arimoto solver, with
500 sampled reward directions, seed 0, and a start at row 4, column 3 of the
8-by-8 central-door grid. Coordinates are zero-based. Change `--gamma`, `--row`,
`--col`, or `--directions` to rerun it. The figure commands can take several
minutes on their first run.

Outputs include the full skill channel and optimized prior in `mi-details.json`,
a plot of the mixture and four illustrative conditional distributions, and
`doorway-skills.gif`. The four displayed skills target different parts of the
grid; the reported MI uses the entire discovered skill set. The GIF contains
sample trajectories, whereas each occupancy averages over a geometrically
distributed observation time and trajectory randomness.

The calculation checks both equivalent expressions:

```text
I(Z; S+) = H(S+) - E_z H(S+ | Z=z)
        = sum_z prior[z] KL(channel[z] || mixture)
mixture = prior @ channel
```

All values use natural logarithms and are reported in nats; divide by `ln(2)`
for bits. Random LP directions can miss occupancy vertices, so the optimized
finite-channel MI is a lower bound on potential empowerment.

For editable experiments and plots, open `mutual_information_lab.ipynb`:

```bash
uv run --with jupyterlab jupyter lab examples/mutual_information_lab.ipynb
```

The notebook recomputes its results using the same source code. Its final
experiment mixes every skill with the shared occupancy distribution, preserving
the overall future-state distribution while removing information about the
chosen skill. The skill prior is held fixed during that comparison.

This repository implements finite MDP geometry and paper figures. It does not
include the Craftax training implementation from the companion
[Scaling Empowered Agents](https://scaling-empowerment.github.io/) work.
