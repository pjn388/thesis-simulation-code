# Thesis supporting code

Supporting code used to generate simulations, analyse results and produce figures
for a thesis investigating dent and gouge defects in steel pipelines.

The Python workflows use CalculiX to model denting, indenter withdrawal and
internal pressurisation. Units are mm, N and MPa.

## Contents

- `final/sym/`: simulation package, case inputs, solver build scripts and tests.
- `final/results/`: numerical verification, analysis and comparison scripts.
- `final/thesis/`: figure-generation scripts and generated thesis figures.
- `artifacts/` and `final/presentation/`: additional simulation-rendering scripts.

## Getting started

Requires Python 3.11, uv and a CalculiX executable. Set `CCX_COMMAND` to its path.
Gmsh also requires libGLU on Linux.

```bash
cd final/sym
uv sync
uv run pipe-sim run cases/elastic_intact.json --out runs/my-elastic-run
uv run pipe-sim benchmark runs/my-elastic-run
uv run python -m pytest tests -q
```

Dependencies are declared in `final/sym/pyproject.toml`. Use a new output directory
for each run.

## Scope

Generated thesis figures are included. Papers, thesis text, historical simulation
data, compiled binaries, caches and lock files are excluded. Recreating historical
figures requires the corresponding data separately.