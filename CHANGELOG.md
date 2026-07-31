# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/); this project uses a
manual `__version__` string in [`version.py`](version.py).
## [1.1.0]

### Added
- **Imperial Unit System** support for material properties, bolts, and calculation inputs.
- **Custom Material fatigue endurance limit ($S_e$)** default derived dynamically from $S_{ut}$.
- **Fatigue check gate** now properly gates the `recommend_bolt` workflow.
- **Combined/Governing FoS columns** in FE import PDF summary tables.

### Fixed
- Fixed **Python 3.12+ import crash** due to eager TypedDict evaluation.
- Fixed **zero-shear torque** not being correctly handled in the analysis logic.
- Corrected **Imperial Stress Area constant** from $\pi/4$ to $\pi/4 \times (1 - 0.9743/n)^2$.
- Fixed **7/8" hex key size** missing mapping.
- Clamped **member force line to 0** in PDF joint diagrams.
- Fixed various theoretical descriptions and equation block references in the engineering manual.


## [1.0.0]

### Added
- **Combined tension–shear** (elliptic interaction) check on the governing bolt of
  a bolt group, shared with the FE-import per-bolt check.
- **Fine-pitch sweep** in the size/grade selector — every size, grade and available
  thread pitch (coarse and fine / UNF) is evaluated.
- **Higher-fidelity thread stripping**: pitch-dependent thread shear areas and a
  differential bolt-vs-nut stripping check (whichever thread strips first governs).
- **VDI 2230 load-introduction factor** `n` refining the effective load-sharing.
- **Neutral-axis (prying)** tension model option for eccentric bolt groups, in
  addition to the centroidal-elastic model.
- **Scatter-band design flow**: min-clamp checks (separation, slip) evaluated at the
  low end of the preload-scatter band and yield/overload at the high end.
- **Sensitivity view** (governing FoS vs preload) and **Monte-Carlo** propagation of
  the preload scatter into an FoS distribution.
- Application **version** surfaced in the UI footer and PDF, and a persistent
  "verify against certified data" disclaimer.
- Packaging & tooling: `LICENSE`, `Dockerfile` (uv-based build), `pyproject.toml`, a
  `uv` lockfile (`uv.lock`) and `uv run` workflow, plus a pinned `constraints.txt`.
- Test suites for `charts`, `analysis`, `project_io`, plus worked-example validation.

### Changed
- Split the monolithic app into `mechanics` / `analysis` / `charts` / `project_io` /
  `fe_import` / `report` / `theory` / `app`, fully type-checked (mypy) and linted.
- Colour-blind-safe (Okabe–Ito) pass/fail palette.
- Reverted an experimental numba dependency (net negative for this workload: ~3 s
  cold-start for no warm-path gain); the bolt-group/FE paths stay vectorised with numpy.

### Fixed
- Custom-material editor add/delete reliability (single source of truth).
- Friction-condition handling no longer silently falls back on an unknown key.
- Dependency pins corrected so a clean install resolves.
