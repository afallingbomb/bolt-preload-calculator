from matplotlib.figure import Figure

from charts import (
    make_joint_diagram, make_cross_section_diagram, make_haigh_diagram,
    alt_joint_diagram, alt_bolt_group_chart, _fe_results_df, fig_to_png,
)
from mechanics import calculate_preload, BOLT_MATERIALS_METRIC

_LAYERS = [
    {"Material": "Steel (Mild)", "thickness": 10.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6},
    {"Material": "Steel (Mild)", "thickness": 10.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6},
]


def _res():
    return calculate_preload(
        d=10.0, p=1.5, bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"], layers=_LAYERS,
        bolt_type="Hex Head", use_washer=True, is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)",
        external_load_max=10000.0, external_load_min=0.0)


def test_make_joint_diagram_degenerate_returns_none():
    assert make_joint_diagram(0.0, 1.0, 1000.0, 0.0, 0.1, 1.0, 1.0, "N", "mm") is None
    assert make_joint_diagram(float('inf'), 1.0, 1000.0, 0.0, 0.1, 1.0, 1.0, "N", "mm") is None


def test_make_joint_diagram_valid_returns_png_figure():
    fig = make_joint_diagram(3.0e5, 2.0e6, 20000.0, 5000.0, 0.13, 1.0, 1.0, "N", "mm")
    assert isinstance(fig, Figure)
    assert fig_to_png(fig).startswith(b"\x89PNG")


def test_make_cross_section_degenerate_and_valid():
    assert make_cross_section_diagram([], 10.0, 15.0, True, 1.0, "mm") is None
    assert make_cross_section_diagram(
        [{"Material": "X", "thickness": 0.0}], 10.0, 15.0, True, 1.0, "mm") is None
    assert isinstance(make_cross_section_diagram(_LAYERS, 10.0, 15.0, True, 1.0, "mm"), Figure)


def test_make_haigh_diagram_valid_and_degenerate():
    res = _res()
    assert isinstance(make_haigh_diagram(res, 1.0, "MPa"), Figure)
    res_no_se = dict(res)
    res_no_se["endurance_Se_MPa"] = 0.0       # no endurance -> no diagram
    assert make_haigh_diagram(res_no_se, 1.0, "MPa") is None


def test_alt_joint_diagram_degenerate_and_valid():
    assert alt_joint_diagram(0.0, 1.0, 1000.0, 0.0, 0.1, 1.0, 1.0, "N", "mm") is None
    assert alt_joint_diagram(3.0e5, 2.0e6, 20000.0, 0.0, 0.13, 1.0, 1.0, "N", "mm") is not None


def test_alt_bolt_group_chart_empty_and_valid():
    assert alt_bolt_group_chart([], [], None, -1, 1.0, 1.0, "mm", "N") is None
    ch = alt_bolt_group_chart([(0.0, 50.0), (0.0, -50.0)], [1000.0, -500.0],
                              None, 0, 1.0, 1.0, "mm", "N")
    assert ch is not None


def test_fe_results_df_adds_derived_columns():
    rows = [{"bolt_id": "A", "passes": True, "sigma_m_MPa": 100.0, "Sut_MPa": 800.0,
             "sigma_a_MPa": 20.0, "Se_MPa": 129.0, "shear_max_N": 1000.0,
             "stress_area_mm2": 58.0, "sigma_max_MPa": 150.0, "Sp_MPa": 600.0,
             "Sy_MPa": 640.0, "min_fos": 1.5, "governing": "Proof"}]
    df = _fe_results_df(rows)
    assert df["Result"].iloc[0] == "PASS"
    for col in ("sm_n", "sa_n", "tau", "tn", "vn"):
        assert col in df.columns
