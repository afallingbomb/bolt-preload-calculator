from streamlit.testing.v1 import AppTest


def test_theory_pdf_builds_from_blocks():
    """The standalone theory-manual PDF must render every block without error."""
    from report import build_theory_pdf
    from theory import THEORY_BLOCKS, THEORY_TITLE
    pdf = build_theory_pdf(THEORY_TITLE, THEORY_BLOCKS)
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 20000          # a real multi-page document, not an empty shell


def test_every_theory_equation_typesets():
    """Every display equation must render with mathtext (no monospace fallback), so
    the exported PDF stays consistently typeset."""
    from report import _eq_png
    from theory import THEORY_BLOCKS
    fallbacks = [c for kind, c in THEORY_BLOCKS if kind == "eq" and _eq_png(c) is None]
    assert not fallbacks, f"{len(fallbacks)} equations fell back to text: {fallbacks[:2]}"


def test_app_loads_and_calculates():
    """Test that the app loads and calculates default values without errors."""
    at = AppTest.from_file("app.py")
    at.run(timeout=15)  # cold start imports matplotlib + pandas + reportlab

    assert not at.exception
    # Should be at least 4 metrics in the main row
    assert len(at.metric) >= 4


def test_bolt_group_mode_renders_without_error():
    """Enabling the eccentric bolt-group analysis (shear + eccentricity + moment)
    must exercise the interactive bolt-group chart and the matplotlib PDF figure
    (with shear-vector arrows) without raising."""
    at = AppTest.from_file("app.py")
    at.run(timeout=20)
    assert not at.exception

    at.session_state["use_group"] = True
    at.session_state["pattern_type"] = "Rectangular grid"
    at.session_state["g_shear"] = 1500.0
    at.session_state["g_ecc"] = 60.0
    at.session_state["g_axial_max"] = 8000.0
    at.session_state["g_moment_max"] = 50.0
    at.run(timeout=20)

    assert not at.exception


def test_fe_import_sample_path_renders():
    """The FE Import tab must parse the built-in sample, evaluate every bolt, and
    build the results table and the separate CSV/PDF report without error."""
    at = AppTest.from_file("app.py")
    at.run(timeout=20)
    assert not at.exception

    at.session_state["fe_use_sample"] = True
    at.run(timeout=20)

    assert not at.exception



def test_imperial_unit_switch():
    """D3: Test Imperial unit switch and rendering."""
    at = AppTest.from_file("app.py")
    at.run(timeout=15)
    
    # Toggle to Imperial
    at.radio(key="units").set_value("Imperial (in, lbf, psi, °F)").run()
    
    assert not at.exception
    # Verify standard imperial threads exist
    assert at.selectbox(key="bolt_size").value == '1/2'
    
    # Switch back to metric
    at.radio(key="units").set_value("Metric (mm, N, MPa, °C)").run()
    assert not at.exception
    assert at.selectbox(key="bolt_size").value == 'M8'

def test_optional_toggles():
    """D4: Test that the app runs successfully with all optional toggles enabled."""
    at = AppTest.from_file("app.py")
    at.run(timeout=15)
    
    at.checkbox(key="use_washer").set_value(True).run()
    at.toggle(key="use_thermal").set_value(True).run()
    at.number_input(key="temp_assembly").set_value(20.0).run()
    at.number_input(key="temp_operating").set_value(150.0).run()
    at.number_input(key="embedment_um").set_value(5.0).run()
    
    at.toggle(key="use_fatigue").set_value(True).run()
    
    at.toggle(key="use_thread").set_value(True).run()
    
    assert not at.exception
