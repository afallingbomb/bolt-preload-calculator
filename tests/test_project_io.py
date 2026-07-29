import pytest

import project_io


@pytest.fixture
def ss(monkeypatch):
    """Replace Streamlit's session_state with a plain dict for the duration of a test.

    project_io only uses item access / .get / `in` / del on session_state, so a dict
    is a faithful stand-in and keeps these tests free of a Streamlit runtime."""
    state: dict = {}
    monkeypatch.setattr(project_io.st, "session_state", state)
    return state


# --- parse_custom_materials (pure) ---------------------------------------------

def test_parse_custom_materials_valid_and_unit_scaling():
    joint, bolt = project_io.parse_custom_materials(
        [{"Name": "MyAlu", "Syc (MPa)": 200.0, "E (GPa)": 70.0, "CTE (µm/m·°C)": 23.0}],
        [{"Name": "MyBolt", "Sp (MPa)": 600.0, "Sy (MPa)": 640.0, "Sut (MPa)": 800.0,
          "Se (MPa)": 129.0, "E (GPa)": 200.0, "CTE (µm/m·°C)": 11.5}])
    assert joint["MyAlu"]["Syc"] == pytest.approx(200.0)
    assert joint["MyAlu"]["E"] == pytest.approx(70000.0)        # GPa -> MPa
    assert joint["MyAlu"]["CTE"] == pytest.approx(23.0e-6)      # µm/m·°C -> 1/°C
    assert bolt["MyBolt"]["Sp"] == pytest.approx(600.0)
    assert bolt["MyBolt"]["Sy"] == pytest.approx(640.0)
    assert bolt["MyBolt"]["E"] == pytest.approx(200000.0)


def test_parse_custom_materials_ignores_incomplete_rows():
    joint, bolt = project_io.parse_custom_materials(
        [{"Name": "", "Syc (MPa)": 200, "E (GPa)": 70},        # no name
         {"Name": "NoE", "Syc (MPa)": 200, "E (GPa)": 0},      # no modulus
         {"Name": "NoSyc", "Syc (MPa)": 0, "E (GPa)": 70}],    # no strength
        [{"Name": "NoSe", "Sp (MPa)": 600, "Sut (MPa)": 800, "Se (MPa)": 0, "E (GPa)": 200}])
    assert joint == {}
    assert bolt == {}


def test_parse_custom_materials_bolt_defaults():
    # Sy and CTE omitted -> Sy defaults to Sp/0.9 and CTE to 11.5e-6.
    _, bolt = project_io.parse_custom_materials(
        [], [{"Name": "B", "Sp (MPa)": 900, "Sut (MPa)": 1000, "Se (MPa)": 150, "E (GPa)": 200}])
    assert bolt["B"]["Sy"] == pytest.approx(900 / 0.9)
    assert bolt["B"]["CTE"] == pytest.approx(11.5e-6)


# --- _coerce_scalar (pure) -----------------------------------------------------

def test_coerce_scalar_types():
    assert project_io._coerce_scalar("use_washer", True) is True
    assert project_io._coerce_scalar("use_washer", 0) is False
    assert project_io._coerce_scalar("num_bolts", 4.6) == 5            # int(round(...))
    assert project_io._coerce_scalar("required_fos", 2) == pytest.approx(2.0)
    assert isinstance(project_io._coerce_scalar("required_fos", 2), float)
    assert project_io._coerce_scalar("bolt_size", "M10") == "M10"      # passthrough


# --- convert_inputs (session_state) --------------------------------------------

def test_convert_inputs_metric_imperial_roundtrip(ss):
    ss.update({
        "thread_engagement": 15.0, "g_px": 50.0,           # lengths
        "ext_max": 1000.0, "g_shear": 250.0,               # forces
        "g_moment_max": 20.0,                              # moment
        "temp_assembly": 20.0, "temp_operating": 120.0,    # temperatures
        "layers": [{"Thickness": 10.0}],
        "group_table": [{"X": 5.0, "Y": -5.0}],
    })
    project_io.convert_inputs(to_metric=False)   # metric -> imperial
    # An intermediate value really did change units (sanity check the conversion ran).
    assert ss["thread_engagement"] == pytest.approx(15.0 / 25.4, abs=1e-4)
    project_io.convert_inputs(to_metric=True)    # imperial -> metric (round trip)

    assert ss["thread_engagement"] == pytest.approx(15.0, abs=1e-3)
    assert ss["g_px"] == pytest.approx(50.0, abs=1e-3)
    assert ss["ext_max"] == pytest.approx(1000.0, abs=1e-2)
    assert ss["g_shear"] == pytest.approx(250.0, abs=1e-3)
    assert ss["g_moment_max"] == pytest.approx(20.0, abs=1e-3)
    assert ss["temp_assembly"] == pytest.approx(20.0, abs=1e-2)
    assert ss["temp_operating"] == pytest.approx(120.0, abs=1e-2)
    assert ss["layers"][0]["Thickness"] == pytest.approx(10.0, abs=1e-3)
    assert ss["group_table"][0]["X"] == pytest.approx(5.0, abs=1e-3)
    assert ss["group_table"][0]["Y"] == pytest.approx(-5.0, abs=1e-3)


# --- build_project / apply_project round-trip ----------------------------------

def test_build_apply_project_roundtrip(ss):
    ss.update({
        "units": "Metric (mm, N, MPa, °C)", "bolt_size": "M10", "use_washer": True,
        "num_bolts": 3, "required_fos": 1.5, "ext_max": 500.0,
        "layers": [{"Material": "Steel (Mild)", "Thickness": 10.0}],
        "custom_joint_rows": [{"Name": "X", "Syc (MPa)": 250.0, "E (GPa)": 200.0, "CTE (µm/m·°C)": 12.0}],
        "_not_an_input": "ignored",
    })
    proj = project_io.build_project()
    assert proj["schema"] == project_io.PROJECT_SCHEMA
    assert "_not_an_input" not in proj["scalars"]      # only INPUT_KEYS are saved

    ss.clear()
    project_io.apply_project(proj)
    assert ss["bolt_size"] == "M10"
    assert ss["use_washer"] is True
    assert ss["num_bolts"] == 3
    assert ss["required_fos"] == pytest.approx(1.5)
    assert ss["ext_max"] == pytest.approx(500.0)
    assert ss["layers"][0]["Material"] == "Steel (Mild)"
    assert ss["custom_joint_rows"][0]["Name"] == "X"


def test_apply_project_rejects_bad_schema(ss):
    with pytest.raises(ValueError):
        project_io.apply_project({"schema": "something-else", "scalars": {}})
    with pytest.raises(ValueError):
        project_io.apply_project({"not": "a project"})


def test_apply_project_skips_uncoercible_scalar(ss):
    # num_bolts must be int-coercible; a non-numeric value is skipped, not crashed.
    proj = {"schema": project_io.PROJECT_SCHEMA,
            "scalars": {"num_bolts": "not-a-number", "bolt_size": "M12"}}
    project_io.apply_project(proj)
    assert ss["bolt_size"] == "M12"
    assert "num_bolts" not in ss          # left at its widget default
