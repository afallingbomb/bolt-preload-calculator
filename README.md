# Bolt Preload & Joint Analysis Calculator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bolt-preload-calculator-g6wrqlutxwitcsaygyj6dw.streamlit.app/)

A Streamlit web application for analysing bolted joints in **metric or imperial**
units: recommended preload, tightening torque, joint constant ($C$), thermal
load, fatigue life, joint separation, thread stripping and **eccentric
bolt-group loading**.

## Features
- **Bolt & joint configuration**: metric/imperial bolt sizes, a **thread
  series / pitch** selector (coarse **and fine** for metric; **UNC / UNF / UNEF**
  for inch — the same 60° profile, so a finer pitch raises the stress area and
  proof load and propagates through every result), bolt types, bolt materials and a
  multi-layer clamped stack from a built-in material database of **~100 engineering
  materials** (steels, stainless, tool steels, cast irons, aluminium, titanium,
  copper alloys, nickel/superalloys, polymers, composites, …) with strengths,
  moduli, CTE and a **recorded data source** for every entry, extensible with
  **user-defined materials**.
- **Preload & torque**: target preload from proof load, nut-factor torque
  ($T = K F d$), an **expected preload-scatter band** based on the chosen
  tightening method, a **von Mises tightening-stress** check (axial + thread
  torsion vs. yield) and a VDI 2230 **embedment / relaxation** preload loss.
- **Stiffness**: two-section bolt stiffness and Rötscher / VDI 2230 frustum-cone
  member stiffness in series, giving the joint constant $C$.
- **Thermal**: differential-expansion preload change between assembly and
  operating temperature.
- **Fatigue**: selectable criterion — **Goodman, Gerber, ASME-elliptic,
  Soderberg, SWT, Morrow**, or the bolt-specific **VDI 2230** endurance limit
  (threads rolled before/after heat treatment). The mean-stress criteria are
  evaluated along the correct **preloaded-bolt load line** (Shigley Ch. 8-12) so
  the constant preload mean stress is not double-counted. A **Haigh diagram**
  plots the operating point against every failure locus, with a side-by-side
  factor-of-safety comparison across all criteria.
- **Bolt group / pattern**: distribute axial load, an overturning moment and
  eccentric in-plane shear over a rectangular grid, bolt circle or custom
  pattern (elastic method). The **governing bolt** drives the tension checks; a
  **slip-critical** check compares the friction capacity ($\mu n_s F_i$) to the
  bolt shear, and a **combined tension–shear** (elliptic interaction) check is
  applied to the governing bolt — all with a colour-mapped pattern plot.
- **Safety checks**: crushing, fatigue, joint separation, bolt proof (yield),
  thread stripping, joint slip, combined tension–shear, and tightening overload —
  with a recommended minimum thread engagement length.
- **Fastener tools** (own tab, reusing the current joint): reverse
  **torque → preload** (with scatter band), **angle / turn-of-nut** control from
  the joint stiffness, a **bolt-length & thread-in-grip** helper (ISO 888 metric /
  ASME B18.2.1 inch), a
  **size / grade selector** that sweeps the catalogue — every size, grade **and
  thread pitch** (coarse and fine / UNF) — for the smallest bolt meeting your
  factor-of-safety targets, and a **reference-dimension** lookup
  (across-flats, hex-key, clearance hole, tap drill).
- **Multi-bolt**: share an external load across an arbitrary number of bolts, or
  use the full bolt-group analysis for eccentric loads.
- **FE results import (CSV)**: a self-contained tab that reads per-bolt results from
  an external finite-element run (in **SI units**; `axial_force_max/min_N` are the
  **total** bolt tension) and reports per-bolt **proof / fatigue / shear / combined**
  factors of safety with a PASS/FAIL verdict. It adds **population graphics**
  (min-FOS histogram, governing-check breakdown, worst-bolts ranking, normalised
  Haigh and tension–shear scatters, FOS-by-check box plots and a cumulative
  distribution), a downloadable **100-bolt sample CSV**, and its **own separate
  report** (annotated CSV + a PDF with the graphics dashboard) that does not affect
  the other tabs' exports.
- **Export**: JSON, a Markdown report, and a **PDF report** (with the joint
  diagram, bolt-pattern plot and Haigh diagram embedded) — all in the selected
  display units.
- **Save / load projects**: download the full input set as JSON and reload it
  later to restore the entire setup (including custom materials and bolt group).
- **Unit auto-convert**: switching between metric and imperial converts the
  values already entered (lengths, forces, temperatures, moments).
- **Transparency**: a collapsible panel lists every equation and key assumption,
  referenced to Shigley Ch. 8 and VDI 2230.

## Material data & sources

Every material in the built-in database stores a `source` tag alongside its
properties (`Syc`, `E`, `CTE`; bolts also `Sp`, `Sy`, `Sut`, `Se`). These are
**nominal, room-temperature, condition/temper-specific** values (the condition is
in the material name) compiled from the standard references below, and they
**must be verified against certified data for design use**. `Syc` is the
compressive yield — the tensile yield for ductile metals, or the compressive
strength for brittle materials (cast irons, concrete). The full list with sources
is shown in-app under **🧪 Custom Materials → 📖 Built-in materials (reference)**,
and the canonical record is the `JOINT_MATERIALS` / `BOLT_MATERIALS_*` tables in
[`mechanics.py`](mechanics.py).

Source tags: **ASM** (ASM Metals Reference Book / Handbook), **MMPDS** (MMPDS-2023,
metallic aerospace materials), **MatWeb**, **CDA** (Copper Development Assoc.),
**SpecialM/Haynes** (superalloy datasheets), **Mfr** (manufacturer datasheets),
**ASTM/ISO/EN …** (the cited standard), **ETB** (Engineering ToolBox / handbooks);
bolt strengths per **ISO 898-1 / SAE J429** with endurance `Se` per **Shigley
Table 8-17**.

## Architecture

The code is split by responsibility; calculations run **internally in SI units**
(mm, N, MPa, °C, N·mm) and convert to/from the selected display units only at the
input and output edges.

| Module | Responsibility |
|---|---|
| [`mechanics.py`](mechanics.py) | Pure engineering core: reference data, preload/stiffness/thermal/fatigue/separation, bolt-group and FE evaluation. No UI; the bolt-group and FE-import paths are vectorised with `numpy`. |
| [`analysis.py`](analysis.py) | Pure presentation logic derived from a result: factor-of-safety formatting and the design findings (warnings/errors). No Streamlit. |
| [`charts.py`](charts.py) | Matplotlib (PDF) and Altair (interactive) figure builders. No Streamlit. |
| [`project_io.py`](project_io.py) | Project save/load (JSON), unit auto-conversion, and the custom-material data helpers; operates on Streamlit `session_state`. |
| [`fe_import.py`](fe_import.py) | The self-contained FE-results import tab and its CSV/PDF report. |
| [`report.py`](report.py) | ReportLab PDF builders (analysis report, FE report, theory manual). |
| [`theory.py`](theory.py) | Single source for the in-app theory manual, rendered both on screen and to PDF. |
| [`app.py`](app.py) | Streamlit page: sidebar, tabs, and the orchestration that wires the above together. |

`benchmark.py` is a standalone micro-benchmark (`python benchmark.py`) for the hot
paths; it is not imported by the app or the tests.

## Running Locally

Copy the project folder to the target machine, then use **[uv](https://docs.astral.sh/uv/)**
(recommended) or plain pip.

### uv (fast, reproducible)
`uv` installs the exact locked dependency set from `pyproject.toml` + `uv.lock` and launches the
app — it creates the virtual environment for you:
```bash
uv run streamlit run app.py
```

### pip
Requires Python 3.12+ (the numpy 2.x / pandas 3.x stack).
```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt   # -c pins the tested versions
streamlit run app.py
```

### Docker
```bash
docker build -t bolt-calc .
docker run -p 8501:8501 bolt-calc
```

## Running on Streamlit Community Cloud

This repository is structured to be seamlessly deployed to Streamlit Community Cloud.
1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click "New app".
3. Select the repository, set the main file path to `app.py`, and hit "Deploy"!

## Development
The dev tools (pytest, flake8, mypy) are a `dev` dependency group. With **uv** they are
installed automatically:
```bash
uv run pytest --cov=.
uv run flake8 .
uv run mypy .
```
With **pip**, install them first (`pip install -r requirements-dev.txt`) and run the same
commands with `python -m` instead of `uv run`. Tool configuration lives in `setup.cfg`
(flake8, mypy) and `pyproject.toml` (pytest).
