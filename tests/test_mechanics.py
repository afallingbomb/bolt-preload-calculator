import math
import pytest
from mechanics import (
    calculate_stress_area, calculate_bearing_area,
    calculate_preload, calculate_layer_stiffness,
    rectangular_pattern, circular_pattern, analyze_bolt_group,
    fatigue_factor_of_safety, vdi2230_endurance_amplitude,
    all_fatigue_factors, tightening_von_mises_stress,
    preload_from_torque, preload_from_yield_percent, tightening_angle, standard_thread_length,
    recommend_bolt_length, grip_thread_engagement, bolt_hardware_reference,
    recommend_bolt, bolt_member_forces, clamp_load_budget,
    thread_series_options, thread_designation,
    evaluate_fe_bolt, evaluate_fe_rows,
    BOLT_MATERIALS_METRIC, BOLT_MATERIALS_IMPERIAL,
    BOLT_SIZES_METRIC, STANDARD_BOLT_LENGTHS_METRIC_MM,
    BOLT_THREAD_SERIES_METRIC, BOLT_THREAD_SERIES_IMPERIAL,
    JOINT_MATERIALS, BOLT_SIZES_IMPERIAL, FATIGUE_CRITERIA
)


def _steel_layers():
    """A simple two-layer steel stack for the fastener-tool tests."""
    return [
        {"Material": "Steel (Mild)", "thickness": 10.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6},
        {"Material": "Steel (Mild)", "thickness": 10.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6},
    ]

def test_material_database_completeness_and_sources():
    # The joint-material database should be large and every entry must carry the
    # numeric properties plus a recorded data source.
    assert len(JOINT_MATERIALS) >= 100
    for name, props in JOINT_MATERIALS.items():
        for key in ("Syc", "E", "CTE", "source"):
            assert key in props, f"{name} missing '{key}'"
        assert props["E"] > 0 and props["Syc"] > 0 and props["CTE"] > 0
        assert isinstance(props["source"], str) and props["source"].strip()
    # Bolt grades must also record a source.
    for db in (BOLT_MATERIALS_METRIC, BOLT_MATERIALS_IMPERIAL):
        for name, props in db.items():
            for key in ("Sp", "Sy", "Sut", "Se", "E", "CTE", "source"):
                assert key in props, f"{name} missing '{key}'"
            assert isinstance(props["source"], str) and props["source"].strip()
    # Materials the test-suite and UI depend on must remain present.
    for required in ("Steel (Mild)", "Steel 4140", "Aluminum (6061-T6)"):
        assert required in JOINT_MATERIALS


def test_calculate_stress_area():
    area = calculate_stress_area(8.0, 1.25)
    assert 36.0 < area < 37.0

def test_calculate_bearing_area():
    # M10 Hex Head -> dw = 15
    ab = calculate_bearing_area(10.0, 15.0)
    assert 81.0 < ab < 82.0

def test_calculate_layer_stiffness():
    k = calculate_layer_stiffness(200000, 10.0, 15.0, 10.0)
    assert k > 0

def test_calculate_preload_basic():
    layers = [
        {"Material": "Steel (Mild)", "thickness": 10.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6},
        {"Material": "Steel (Mild)", "thickness": 10.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}
    ]
    result = calculate_preload(
        d=10.0, p=1.5,
        bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"],
        layers=layers,
        bolt_type="Hex Head",
        use_washer=False,
        is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)"
    )
    
    assert 57.0 < result["tensile_stress_area_mm2"] < 59.0
    assert 34000 < result["proof_load_N"] < 35500
    assert result["crushing_warning_material"] == "Layer 1"
    assert result["recommended_preload_N"] < result["target_preload_N"]
    assert result["kb_N_mm"] > 0
    assert result["km_N_mm"] > 0
    assert 0 < result["joint_constant_C"] < 1.0

def test_thermal_expansion():
    # Aluminum joint expands more than steel bolt -> preload increases
    layers = [
        {"Material": "Aluminum (6061-T6)", "thickness": 50.0, "Syc": 275, "E": 69000, "CTE": 23.6e-6}
    ]
    result = calculate_preload(
        d=10.0, p=1.5,
        bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"], # CTE 11.5e-6
        layers=layers,
        bolt_type="Hex Head",
        use_washer=True,
        is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)",
        temp_assembly=20.0,
        temp_operating=120.0 # Heating up
    )
    
    # Delta F should be positive
    assert result["thermal_delta_F_N"] > 0
    assert result["operating_preload_N"] > result["recommended_preload_N"]

def test_fatigue_and_stripping():
    layers = [
        {"Material": "Steel (Mild)", "thickness": 20.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}
    ]
    result = calculate_preload(
        d=10.0, p=1.5,
        bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"],
        layers=layers,
        bolt_type="Hex Head",
        use_washer=False,
        is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)",
        external_load_max=10000.0,
        external_load_min=0.0,
        thread_engagement_length=15.0,
        internal_thread_material_props=JOINT_MATERIALS["Aluminum (6061-T6)"]
    )
    
    # Fatigue checks
    assert result["fatigue_sigma_a_MPa"] > 0
    assert result["fatigue_sigma_m_MPa"] > 0
    assert result["fatigue_fos"] > 0
    
    # Shear stripping checks
    assert result["thread_shear_fos"] > 0

def test_fatigue_infinite_without_alternating_load():
    # With no cyclic external load there is no fatigue concern -> infinite FOS.
    layers = [
        {"Material": "Steel (Mild)", "thickness": 20.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}
    ]
    result = calculate_preload(
        d=10.0, p=1.5,
        bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"],
        layers=layers,
        bolt_type="Hex Head",
        use_washer=True,
        is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)",
        external_load_max=0.0,
        external_load_min=0.0,
    )
    assert result["fatigue_fos"] == float("inf")
    assert result["fatigue_sigma_a_MPa"] == 0.0


def test_fatigue_loadline_from_preload_point():
    # Closed-form check of the Shigley preloaded-bolt Goodman FOS:
    #   n_f = Se (Sut - sigma_i) / (Sut*sigma_a + Se*(sigma_m - sigma_i))
    layers = [
        {"Material": "Steel (Mild)", "thickness": 20.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}
    ]
    result = calculate_preload(
        d=10.0, p=1.5,
        bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"],
        layers=layers,
        bolt_type="Hex Head",
        use_washer=True,
        is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)",
        external_load_max=10000.0,
        external_load_min=0.0,
    )
    Se, Sut = 129.0, 800.0
    s_a = result["fatigue_sigma_a_MPa"]
    s_m = result["fatigue_sigma_m_MPa"]
    s_i = result["preload_stress_MPa"]
    expected = (Se * (Sut - s_i)) / (Sut * s_a + Se * (s_m - s_i))
    assert result["fatigue_fos"] == pytest.approx(expected, rel=1e-6)
    # The proper load line is less conservative than naive Goodman-from-origin.
    naive = 1.0 / (s_a / Se + s_m / Sut)
    assert result["fatigue_fos"] > naive


def test_fatigue_factor_goodman_matches_closed_form():
    # The helper's Goodman branch must equal Shigley Eq. 8-38.
    Se, Sut, Sp, Sy = 129.0, 800.0, 600.0, 640.0
    sa, sm, si = 40.0, 200.0, 150.0
    expected = (Se * (Sut - si)) / (Sut * sa + Se * (sm - si))
    assert fatigue_factor_of_safety("Goodman", sa, sm, si, Se, Sut, Sp, Sy) == pytest.approx(expected)


def test_fatigue_factor_no_alternating_is_infinite():
    for crit in ("Goodman", "Gerber", "ASME-elliptic", "Soderberg"):
        assert fatigue_factor_of_safety(crit, 0.0, 450.0, 450.0, 129, 800, 600, 640) == float("inf")


def test_fatigue_factor_gerber_on_locus():
    # A point sitting exactly on the Gerber locus with zero preload must give n_f = 1.
    Se, Sut = 100.0, 500.0
    sa, sm = 36.0, 400.0          # 36/100 + (400/500)^2 = 0.36 + 0.64 = 1.0
    n = fatigue_factor_of_safety("Gerber", sa, sm, 0.0, Se, Sut, 400.0, 415.0)
    assert n == pytest.approx(1.0, rel=1e-6)


def test_fatigue_criteria_relative_conservatism():
    # For a representative preloaded joint the classic ordering holds:
    #   Soderberg <= Goodman <= ASME-elliptic <= Gerber.
    layers = [
        {"Material": "Steel (Mild)", "thickness": 20.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}
    ]

    def fos(crit):
        return calculate_preload(
            d=10.0, p=1.5, bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"],
            layers=layers, bolt_type="Hex Head", use_washer=True, is_permanent=False,
            friction_condition="Dry / as-received (K=0.20)",
            external_load_max=15000.0, external_load_min=0.0,
            fatigue_criterion=crit)["fatigue_fos"]

    g, ge, a, s = fos("Goodman"), fos("Gerber"), fos("ASME-elliptic"), fos("Soderberg")
    assert all(x > 0 for x in (g, ge, a, s))
    assert s <= g <= a <= ge


def test_fatigue_swt_on_locus():
    # sigma_ar = sqrt(sigma_max * sigma_a) = Se with zero preload -> n_f = 1.
    Se = 100.0
    sa, sm = 50.0, 150.0          # sigma_max = 200; sqrt(200*50) = 100 = Se
    n = fatigue_factor_of_safety("SWT", sa, sm, 0.0, Se, 800, 600, 640)
    assert n == pytest.approx(1.0, rel=1e-6)


def test_fatigue_morrow_less_conservative_than_goodman():
    Se, Sut, Sp, Sy = 129.0, 800.0, 600.0, 640.0
    sa, sm, si = 30.0, 480.0, 450.0
    g = fatigue_factor_of_safety("Goodman", sa, sm, si, Se, Sut, Sp, Sy)
    m = fatigue_factor_of_safety("Morrow", sa, sm, si, Se, Sut, Sp, Sy)
    assert m > g > 0


def test_vdi2230_endurance_amplitude_formula():
    # M10 rolled before HT: 0.85 * (150/10 + 45) = 0.85 * 60 = 51 MPa.
    asv = vdi2230_endurance_amplitude(10.0, False, 0.0, 1.0)
    assert asv == pytest.approx(51.0, rel=1e-6)
    # Rolled after HT with negligible mean force -> factor ~2 -> twice as high.
    asg = vdi2230_endurance_amplitude(10.0, True, 0.0, 1000.0)
    assert asg == pytest.approx(102.0, rel=1e-6)
    assert asg > asv


def test_vdi2230_criterion_integration():
    layers = [
        {"Material": "Steel (Mild)", "thickness": 20.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}
    ]
    kw = dict(
        d=10.0, p=1.5, bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"],
        layers=layers, bolt_type="Hex Head", use_washer=True, is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)",
        external_load_max=12000.0, external_load_min=0.0)
    before = calculate_preload(fatigue_criterion="VDI 2230, rolled before HT", **kw)
    after = calculate_preload(fatigue_criterion="VDI 2230, rolled after HT", **kw)
    assert before["fatigue_fos"] > 0
    assert before["fatigue_criterion"] == "VDI 2230, rolled before HT"
    # Rolling after heat treatment is never weaker.
    assert after["fatigue_fos"] >= before["fatigue_fos"]


def test_tightening_von_mises_exceeds_axial():
    # The combined (axial + torsion) stress is always >= the pure axial stress.
    At = 58.0
    preload = 26000.0
    axial = preload / At
    vm = tightening_von_mises_stress(preload, At, d=10.0, p=1.5, nut_factor=0.20)
    assert vm > axial
    # With a frictionless thread (low K) there is little torsion -> close to axial.
    vm_low = tightening_von_mises_stress(preload, At, d=10.0, p=1.5, nut_factor=0.02)
    assert vm_low == pytest.approx(axial, rel=0.05)


def test_embedment_reduces_operating_preload():
    layers = [
        {"Material": "Steel (Mild)", "thickness": 20.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}
    ]
    common = dict(
        d=10.0, p=1.5, bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"],
        layers=layers, bolt_type="Hex Head", use_washer=True, is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)")
    base = calculate_preload(embedment_um=0.0, **common)
    embed = calculate_preload(embedment_um=10.0, **common)
    assert embed["embedment_loss_N"] > 0
    assert embed["operating_preload_N"] == pytest.approx(
        base["operating_preload_N"] - embed["embedment_loss_N"], rel=1e-9)


def test_all_fatigue_factors_covers_every_criterion():
    factors = all_fatigue_factors(20.0, 470.0, 450.0, 129, 800, 600, 640,
                                  d=10.0, bolt_mean_force=16000.0, yield_force=37000.0)
    assert set(factors.keys()) == set(FATIGUE_CRITERIA)
    assert all(v > 0 for v in factors.values())


def test_separation_and_proof():
    layers = [
        {"Material": "Steel (Mild)", "thickness": 20.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}
    ]
    result = calculate_preload(
        d=12.0, p=1.75,
        bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"],
        layers=layers,
        bolt_type="Hex Head",
        use_washer=True,
        is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)",
        external_load_max=5000.0,
        external_load_min=0.0,
    )
    # Separation load = Fi / (1 - C) and must exceed the operating preload.
    assert result["separation_load_N"] == pytest.approx(
        result["operating_preload_N"] / (1.0 - result["joint_constant_C"]), rel=1e-6)
    assert result["separation_load_N"] > result["operating_preload_N"]
    assert result["separation_fos"] > 1.0
    # Proof FOS = proof load / max bolt force.
    assert result["proof_fos"] == pytest.approx(
        result["proof_load_N"] / result["max_bolt_force_N"], rel=1e-6)


def test_required_engagement_develops_proof_load():
    # At exactly the required engagement, the internal threads carry the bolt
    # proof load, i.e. the stripping FOS evaluated against Fp equals 1.0.
    layers = [
        {"Material": "Aluminum (6061-T6)", "thickness": 20.0, "Syc": 275, "E": 69000, "CTE": 23.6e-6}
    ]
    result = calculate_preload(
        d=10.0, p=1.5,
        bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"],
        layers=layers,
        bolt_type="Hex Head",
        use_washer=True,
        is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)",
        thread_engagement_length=0.0,
        internal_thread_material_props=JOINT_MATERIALS["Aluminum (6061-T6)"],
    )
    Le = result["required_engagement_mm"]
    assert Le > 0
    d, p = 10.0, 1.5
    Ssy_int = 0.577 * JOINT_MATERIALS["Aluminum (6061-T6)"]["Syc"]
    Ssy_bolt = 0.577 * BOLT_MATERIALS_METRIC["Grade 8.8"]["Sy"]
    cap_int = Ssy_int * 0.875 * math.pi * d                       # internal (nut) capacity per mm
    cap_bolt = Ssy_bolt * 0.75 * math.pi * (d - 1.0825 * p)       # bolt-thread capacity per mm
    # The soft aluminium tapped hole strips before the Grade 8.8 bolt threads.
    assert cap_int < cap_bolt
    # At the required engagement the governing (internal) threads develop the proof load.
    assert cap_int * Le == pytest.approx(result["proof_load_N"], rel=1e-6)


def test_load_intro_factor_reduces_bolt_load_share():
    layers = [{"Material": "Steel (Mild)", "thickness": 20.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}]
    kw = dict(d=10.0, p=1.5, bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"], layers=layers,
              bolt_type="Hex Head", use_washer=True, is_permanent=False,
              friction_condition="Dry / as-received (K=0.20)",
              external_load_max=10000.0, external_load_min=0.0)
    full = calculate_preload(load_intro_factor=1.0, **kw)
    reduced = calculate_preload(load_intro_factor=0.5, **kw)
    # A smaller load-introduction factor -> the bolt sees less of the external load
    # (better for bolt fatigue), but the members are relieved more, so the joint
    # separates at a lower external load (a more conservative separation margin).
    assert reduced["max_bolt_force_N"] < full["max_bolt_force_N"]
    assert reduced["separation_load_N"] < full["separation_load_N"]
    # The geometric joint constant is unchanged (n scales load-sharing, not stiffness).
    assert reduced["joint_constant_C"] == pytest.approx(full["joint_constant_C"])


def test_imperial_bolt_size_is_metric_equivalent():
    # The imperial size table stores MILLIMETRE equivalents, so the app must use
    # them as-is (no 25.4x conversion). A 1/2-13 bolt is 0.5 in = 12.7 mm and has
    # a tensile stress area near 0.14 in^2 (~92.7 mm^2). A double conversion would
    # blow this up by 25.4^2, so guard against that regression.
    d_mm, p_mm = BOLT_SIZES_IMPERIAL["1/2"]
    assert d_mm == pytest.approx(12.7, rel=1e-3)        # 0.5 in
    assert p_mm == pytest.approx(25.4 / 13.0, rel=1e-3)  # 13 TPI (UNC default)
    At = calculate_stress_area(d_mm, p_mm)
    assert 90.0 < At < 95.0                              # mm^2
    assert 0.135 < At / 645.16 < 0.150                  # in^2


def test_calculate_preload_infinite_stiffness():
    layers_zero = [
        {"Material": "Steel", "thickness": 0.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}
    ]
    result_zero = calculate_preload(
        d=10.0, p=1.5,
        bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"],
        layers=layers_zero,
        bolt_type="Hex Head",
        use_washer=False,
        is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)"
    )
    assert result_zero["km_N_mm"] == 0.0


def test_rectangular_pattern_centroid_and_count():
    coords = rectangular_pattern(rows=2, cols=3, pitch_x=50.0, pitch_y=40.0)
    assert len(coords) == 6
    cx = sum(x for x, _ in coords) / len(coords)
    cy = sum(y for _, y in coords) / len(coords)
    assert cx == pytest.approx(0.0)
    assert cy == pytest.approx(0.0)
    xs = sorted({round(x, 6) for x, _ in coords})
    assert xs == pytest.approx([-50.0, 0.0, 50.0])


def test_circular_pattern_radius_and_count():
    coords = circular_pattern(n=4, bolt_circle_dia=100.0)
    assert len(coords) == 4
    for x, y in coords:
        assert math.hypot(x, y) == pytest.approx(50.0)


def test_bolt_group_pure_axial_shares_equally():
    coords = rectangular_pattern(2, 2, 50.0, 50.0)
    res = analyze_bolt_group(coords, axial_load=4000.0)
    assert res["governing_tension_N"] == pytest.approx(1000.0)
    assert all(t == pytest.approx(1000.0) for t in res["tensions_N"])
    assert res["governing_shear_N"] == pytest.approx(0.0)


def test_bolt_group_moment_tension_distribution():
    # Two bolts at y = +/-50 mm, moment 1000 N*mm about the x-axis.
    # Sum d^2 = 2*50^2 = 5000; tension = M*d/Sum d^2 = 1000*50/5000 = 10 N.
    coords = [(0.0, 50.0), (0.0, -50.0)]
    res = analyze_bolt_group(coords, moment=1000.0, moment_axis="x")
    assert res["sum_distance_sq_mm2"] == pytest.approx(5000.0)
    assert res["governing_tension_N"] == pytest.approx(10.0)
    assert sorted(res["tensions_N"]) == pytest.approx([-10.0, 10.0])


def test_bolt_group_moment_not_reactable_when_collinear_with_axis():
    # All bolts on the bending axis (same y) cannot react a moment about x.
    coords = [(-50.0, 0.0), (50.0, 0.0)]
    res = analyze_bolt_group(coords, axial_load=2000.0, moment=1000.0, moment_axis="x")
    assert res["moment_reactable"] is False
    assert res["governing_tension_N"] == pytest.approx(1000.0)  # axial share only


def test_bolt_group_torsional_shear():
    # Four bolts at +/-50 in x and y; in-plane shear 4000 N at eccentricity 100 mm.
    # J = sum(x^2+y^2) = 4*(50^2+50^2) = 20000 mm^2; T = 4000*100 = 4e5 N*mm.
    # Torsional shear at r=70.71 mm: T*r/J = 4e5*70.71/20000 = 1414.2 N, at 45 deg.
    # Direct shear V/N = 1000 N along +y. Worst bolt resultant ~ 2236 N.
    coords = rectangular_pattern(2, 2, 100.0, 100.0)
    res = analyze_bolt_group(coords, shear_load=4000.0, shear_eccentricity=100.0)
    assert res["polar_moment_mm2"] == pytest.approx(20000.0)
    assert res["governing_shear_N"] == pytest.approx(2236.07, rel=1e-3)


# =============================================================================
# Fastener tools
# =============================================================================

def test_preload_from_torque_inverts_torque_relation():
    # T = K F d (d in metres): 100 N*m, K=0.2, d=12 mm -> F = 100/(0.2*0.012) = 41,667 N.
    F = preload_from_torque(100.0, 0.20, 12.0)
    assert F == pytest.approx(100.0 * 1000.0 / (0.20 * 12.0))
    assert F == pytest.approx(41666.67, rel=1e-4)
    # Consistent with the forward torque computed by calculate_preload.
    res = calculate_preload(
        d=12.0, p=1.75, bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"],
        layers=_steel_layers(), bolt_type="Hex Head", use_washer=True,
        is_permanent=False, friction_condition="Dry / as-received (K=0.20)")
    back = preload_from_torque(res["torque_Nm"], 0.20, 12.0)
    assert back == pytest.approx(res["recommended_preload_N"], rel=1e-6)


def test_preload_from_torque_degenerate_inputs():
    assert preload_from_torque(50.0, 0.0, 10.0) == 0.0
    assert preload_from_torque(50.0, 0.2, 0.0) == 0.0


def test_tightening_angle_matches_series_deflection():
    kb, km, p, F = 400000.0, 600000.0, 1.5, 30000.0
    delta = F * (1.0 / kb + 1.0 / km)        # series-spring deflection (mm)
    expected = 360.0 * delta / p
    assert tightening_angle(F, kb, km, p) == pytest.approx(expected)
    # A snug preload reduces the remaining rotation, linearly.
    half = tightening_angle(F, kb, km, p, snug_preload=F / 2.0)
    assert half == pytest.approx(expected / 2.0)
    # Degenerate stiffness / pitch -> no angle.
    assert tightening_angle(F, 0.0, km, p) == 0.0
    assert tightening_angle(F, kb, km, 0.0) == 0.0
    assert tightening_angle(F, float('inf'), km, p) == 0.0


def test_standard_thread_length_iso888_bands():
    assert standard_thread_length(10.0, 100.0) == pytest.approx(26.0)   # 2d+6, L<=125
    assert standard_thread_length(10.0, 125.0) == pytest.approx(26.0)
    assert standard_thread_length(10.0, 150.0) == pytest.approx(32.0)   # 2d+12, 125<L<=200
    assert standard_thread_length(10.0, 250.0) == pytest.approx(45.0)   # 2d+25, L>200


def test_recommend_bolt_length_rounds_up_to_standard():
    l_min, rec = recommend_bolt_length(20.0, 12.0, STANDARD_BOLT_LENGTHS_METRIC_MM)
    assert l_min == pytest.approx(32.0)
    assert rec == 35.0                                   # next preferred length >= 32
    # An exact standard length is accepted as-is.
    _, exact = recommend_bolt_length(20.0, 20.0, STANDARD_BOLT_LENGTHS_METRIC_MM)
    assert exact == 40.0
    # Nothing long enough -> None.
    l_min2, none_rec = recommend_bolt_length(500.0, 50.0, STANDARD_BOLT_LENGTHS_METRIC_MM)
    assert l_min2 == pytest.approx(550.0)
    assert none_rec is None


def test_grip_thread_engagement_flags_threads_in_grip():
    # M10, L=30 mm -> b = 2*10+6 = 26, shank = 4 mm. Grip 20 mm -> threads in grip.
    eng = grip_thread_engagement(10.0, 30.0, 20.0)
    assert eng["thread_length_mm"] == pytest.approx(26.0)
    assert eng["shank_length_mm"] == pytest.approx(4.0)
    assert eng["threads_in_grip"] is True
    # A long bolt with a short grip -> shank spans the grip.
    eng2 = grip_thread_engagement(10.0, 100.0, 20.0)   # b=26, shank=74 > 20
    assert eng2["threads_in_grip"] is False


def test_bolt_hardware_reference_known_and_fallback():
    hw = bolt_hardware_reference("M8", 8.0, 1.25)
    assert hw["hex_af_mm"] == pytest.approx(13.0)
    assert hw["socket_af_mm"] == pytest.approx(6.0)
    assert hw["clearance_hole_mm"] == pytest.approx(9.0)
    assert hw["tap_drill_mm"] == pytest.approx(6.75)     # d - p
    # Unknown size: wrench data is None, clearance falls back to the 1.1 d rule.
    unknown = bolt_hardware_reference("M99", 99.0, 4.0)
    assert unknown["hex_af_mm"] is None
    assert unknown["socket_af_mm"] is None
    assert unknown["clearance_hole_mm"] == pytest.approx(round(99.0 * 1.1, 2))
    assert unknown["tap_drill_mm"] == pytest.approx(95.0)


def test_recommend_bolt_finds_smallest_passing():
    # Static external load of 20 kN per bolt. The preload is 75% of proof, so the
    # proof FoS ceiling is ~1.33; require 1.2 plus a separation margin.
    rec = recommend_bolt(
        BOLT_SIZES_METRIC, BOLT_MATERIALS_METRIC, _steel_layers(),
        bolt_type="Hex Head", use_washer=True, is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)",
        external_load_max=20000.0, external_load_min=20000.0,
        target_proof_fos=1.2, target_separation_fos=1.1)
    assert rec["found"] is True
    best = rec["best"]
    assert best is not None
    assert best["proof_fos"] >= 1.2
    assert best["separation_fos"] >= 1.1
    # Candidates are ranked by ascending stress area (smallest/lightest first).
    areas = [c["stress_area_mm2"] for c in rec["candidates"]]
    assert areas == sorted(areas)
    # The best is the smallest-area passing candidate.
    assert best["stress_area_mm2"] == pytest.approx(min(areas))


def test_recommend_bolt_none_when_load_excessive():
    # An enormous load no catalogue bolt can carry -> nothing passes the proof gate.
    rec = recommend_bolt(
        BOLT_SIZES_METRIC, BOLT_MATERIALS_METRIC, _steel_layers(),
        bolt_type="Hex Head", use_washer=True, is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)",
        external_load_max=5.0e6, external_load_min=5.0e6,
        target_proof_fos=1.2, target_separation_fos=1.1)
    assert rec["found"] is False
    assert rec["best"] is None
    assert rec["candidates"] == []


def test_recommend_bolt_larger_load_needs_bigger_bolt():
    common = dict(
        sizes=BOLT_SIZES_METRIC, materials={"Grade 8.8": BOLT_MATERIALS_METRIC["Grade 8.8"]},
        layers=_steel_layers(), bolt_type="Hex Head", use_washer=True, is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)", target_proof_fos=1.2)
    light = recommend_bolt(external_load_max=5000.0, external_load_min=5000.0, **common)
    heavy = recommend_bolt(external_load_max=40000.0, external_load_min=40000.0, **common)
    assert light["found"] and heavy["found"]
    assert heavy["best"]["stress_area_mm2"] >= light["best"]["stress_area_mm2"]


# =============================================================================
# Visualization support (force-vs-load, clamp budget, shear vectors)
# =============================================================================

def test_bolt_member_forces_before_and_after_separation():
    Fi, C = 10000.0, 0.3
    # No load -> both equal the preload.
    assert bolt_member_forces(Fi, C, 0.0) == pytest.approx((Fi, Fi))
    # Below separation (P_sep = Fi/(1-C) = 14285.7): linear sharing.
    fb, fm = bolt_member_forces(Fi, C, 2000.0)
    assert fb == pytest.approx(Fi + C * 2000.0)
    assert fm == pytest.approx(Fi - (1.0 - C) * 2000.0)
    assert fm > 0.0
    # Above separation: member force pinned at 0, bolt carries the whole load.
    fb2, fm2 = bolt_member_forces(Fi, C, 20000.0)
    assert fm2 == 0.0
    assert fb2 == pytest.approx(20000.0)
    # Compression stays on the linear branch (no separation).
    fb3, fm3 = bolt_member_forces(Fi, C, -2000.0)
    assert fb3 == pytest.approx(Fi - C * 2000.0)
    assert fm3 == pytest.approx(Fi + (1.0 - C) * 2000.0)


def test_clamp_load_budget_steps_and_residual():
    steps = clamp_load_budget(installation=10000.0, embedment_loss=500.0,
                              thermal_delta_F=300.0, C=0.3, external_load_max=2000.0)
    labels = [s["label"] for s in steps]
    assert labels == ["Installation", "Embedment", "Thermal", "Ext. load relief", "Residual clamp"]
    assert steps[0]["kind"] == "start" and steps[-1]["kind"] == "total"
    # Running cumulative is consistent with the signed deltas.
    assert steps[0]["cumulative"] == pytest.approx(10000.0)
    assert steps[1]["cumulative"] == pytest.approx(9500.0)
    assert steps[2]["cumulative"] == pytest.approx(9800.0)
    assert steps[3]["delta"] == pytest.approx(-(1.0 - 0.3) * 2000.0)
    # Residual == operating preload minus external relief == F_m at P_max.
    residual = 10000.0 - 500.0 + 300.0 - (1.0 - 0.3) * 2000.0
    assert steps[-1]["cumulative"] == pytest.approx(residual)
    # No external load -> no relief step value.
    no_load = clamp_load_budget(10000.0, 500.0, 300.0, 0.3, 0.0)
    assert no_load[3]["delta"] == pytest.approx(0.0)


def test_bolt_group_shear_vectors_match_magnitudes():
    coords = rectangular_pattern(2, 2, 100.0, 100.0)
    res = analyze_bolt_group(coords, shear_load=4000.0, shear_eccentricity=100.0)
    vecs = res["shear_vectors_N"]
    assert len(vecs) == len(coords)
    # Each stored magnitude equals the hypot of its (Fx, Fy) components.
    for (vx, vy), mag in zip(vecs, res["shears_N"]):
        assert math.hypot(vx, vy) == pytest.approx(mag)
    assert res["governing_shear_N"] == pytest.approx(max(math.hypot(vx, vy) for vx, vy in vecs))
    # Empty pattern still exposes the key.
    assert analyze_bolt_group([])["shear_vectors_N"] == []


# =============================================================================
# Thread series (coarse / fine / UNC / UNF / UNEF)
# =============================================================================

def test_fine_pitch_increases_stress_area_and_proof_load():
    # Same nominal diameter, finer pitch -> larger tensile-stress area, higher proof.
    coarse = calculate_stress_area(12.0, 1.75)   # M12 coarse
    fine = calculate_stress_area(12.0, 1.25)     # M12 fine
    assert fine > coarse
    layers = _steel_layers()
    common = dict(d=12.0, bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"], layers=layers,
                  bolt_type="Hex Head", use_washer=True, is_permanent=False,
                  friction_condition="Dry / as-received (K=0.20)")
    res_coarse = calculate_preload(p=1.75, **common)
    res_fine = calculate_preload(p=1.25, **common)
    assert res_fine["tensile_stress_area_mm2"] > res_coarse["tensile_stress_area_mm2"]
    assert res_fine["proof_load_N"] > res_coarse["proof_load_N"]


def test_thread_series_tables_consistency():
    # Every size in the size tables must have a series list whose first (default)
    # pitch matches the coarse pitch stored in the size tuple.
    for sizes, series in ((BOLT_SIZES_METRIC, BOLT_THREAD_SERIES_METRIC),
                          (BOLT_SIZES_IMPERIAL, BOLT_THREAD_SERIES_IMPERIAL)):
        assert set(sizes.keys()) == set(series.keys())
        for name, (_d, coarse_p) in sizes.items():
            opts = series[name]
            assert len(opts) >= 2                       # at least coarse + one fine
            assert opts[0][1] == pytest.approx(coarse_p)   # first entry is the coarse pitch
            fine_pitches = [p for _lbl, p in opts[1:]]
            assert all(fp < coarse_p for fp in fine_pitches)   # fine pitches are smaller


def test_thread_series_options_and_designation():
    metric_opts = thread_series_options("M12", metric=True)
    assert ("Fine — 1.50 mm", 1.50) in metric_opts
    inch_opts = thread_series_options("1/2", metric=False)
    labels = [lbl for lbl, _ in inch_opts]
    assert any("UNF" in lbl for lbl in labels)
    # Designations.
    assert thread_designation("M12", "Fine — 1.50 mm", True, 1.5) == "M12×1.5"
    assert thread_designation("1/2", "UNF — 20 TPI", False, 1.27) == "1/2-20 UNF"
    # Unknown size -> empty option list.
    assert thread_series_options("M99", metric=True) == []


def test_standard_thread_length_inch_rule():
    # ASME B18.2.1: 2d + 1/4 in (<=6 in) else 2d + 1/2 in.
    d = 12.7                                   # 1/2 in
    assert standard_thread_length(d, 50.0, metric=False) == pytest.approx(2 * d + 6.35)
    assert standard_thread_length(d, 200.0, metric=False) == pytest.approx(2 * d + 12.7)
    # The metric default is unchanged.
    assert standard_thread_length(10.0, 100.0) == pytest.approx(26.0)


def test_grip_thread_engagement_respects_unit_rule():
    # The two conventions apply different thread-length rules, so the engagement
    # check differs by unit system. For this short bolt the inch rule (2d+1/4") gives
    # a slightly longer thread than the metric rule (2d+6 mm), hence a shorter shank.
    metric = grip_thread_engagement(12.7, 60.0, 20.0, metric=True)
    inch = grip_thread_engagement(12.7, 60.0, 20.0, metric=False)
    assert inch["thread_length_mm"] == pytest.approx(2 * 12.7 + 6.35)
    assert metric["thread_length_mm"] == pytest.approx(2 * 12.7 + 6.0)
    assert inch["shank_length_mm"] < metric["shank_length_mm"]
    # At long lengths the rules diverge sharply (metric 2d+25 vs inch 2d+12.7).
    metric_long = grip_thread_engagement(12.7, 250.0, 20.0, metric=True)
    inch_long = grip_thread_engagement(12.7, 250.0, 20.0, metric=False)
    assert metric_long["thread_length_mm"] == pytest.approx(2 * 12.7 + 25.0)
    assert inch_long["thread_length_mm"] == pytest.approx(2 * 12.7 + 12.7)


def test_thermal_same_material():
    # If the joint and bolt are both made of steel (same CTE), the preload should NOT change with temperature.
    layers = [
        {"Material": "Steel (Mild)", "thickness": 50.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}
    ]
    result = calculate_preload(
        d=10.0, p=1.5,
        bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"], # CTE 11.5e-6
        layers=layers,
        bolt_type="Hex Head",
        use_washer=True,
        is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)",
        temp_assembly=20.0,
        temp_operating=120.0
    )
    # Delta F should be exactly 0
    assert result["thermal_delta_F_N"] == pytest.approx(0.0)
    assert result["operating_preload_N"] == pytest.approx(result["recommended_preload_N"])


def test_thermal_infinitely_stiff_joint():
    # If the joint is infinitely stiff, its free expansion MUST be absorbed entirely by the bolt stretching.
    layers = [
        {"Material": "Infinitely Stiff Mat", "thickness": 50.0, "Syc": 250, "E": float('inf'), "CTE": 23.6e-6}
    ]
    result = calculate_preload(
        d=10.0, p=1.5,
        bolt_material_props=BOLT_MATERIALS_METRIC["Grade 8.8"], # CTE 11.5e-6
        layers=layers,
        bolt_type="Hex Head",
        use_washer=True,
        is_permanent=False,
        friction_condition="Dry / as-received (K=0.20)",
        temp_assembly=20.0,
        temp_operating=120.0
    )
    # Delta F should be > 0 because it expands more than the steel bolt
    assert result["thermal_delta_F_N"] > 0.0
    
    # Delta F = delta_deflection * kb
    delta_T = 100.0
    joint_exp = 50.0 * 23.6e-6 * delta_T
    bolt_exp = 50.0 * 11.5e-6 * delta_T
    delta_deflection = joint_exp - bolt_exp
    expected_delta_F = delta_deflection * result["kb_N_mm"]
    assert result["thermal_delta_F_N"] == pytest.approx(expected_delta_F, rel=1e-6)


def test_bolt_hardware_reference_imperial_renamed_key():
    hw = bolt_hardware_reference("1/2", 12.7, 1.9538)
    assert hw["hex_af_mm"] == pytest.approx(19.05)     # 3/4 in wrench
    assert hw["tap_drill_mm"] == pytest.approx(12.7 - 1.9538)


# =============================================================================
# FE results import
# =============================================================================

_G109 = BOLT_MATERIALS_METRIC["Grade 10.9"]


def test_evaluate_fe_bolt_basic_pass():
    ev = evaluate_fe_bolt(12.0, 1.75, _G109["Sp"], _G109["Sy"], _G109["Sut"], _G109["Se"],
                          axial_max=45000.0, axial_min=41000.0, preload=40000.0,
                          shear_max=3000.0, fatigue_criterion="Goodman", target_fos=1.5)
    assert ev["stress_area_mm2"] == pytest.approx(84.27, rel=1e-3)
    # proof FOS = Sp*At / axial_max
    assert ev["proof_fos"] == pytest.approx(_G109["Sp"] * 84.27 / 45000.0, rel=1e-3)
    assert ev["sigma_a_MPa"] > 0 and ev["sigma_m_MPa"] > ev["sigma_a_MPa"]
    assert ev["min_fos"] >= 1.5 and ev["passes"] is True
    # min_fos is the smallest of the four checks
    assert ev["min_fos"] == pytest.approx(min(ev["proof_fos"], ev["fatigue_fos"],
                                              ev["shear_fos"], ev["combined_fos"]))


def test_evaluate_fe_bolt_proof_overload_fails():
    ev = evaluate_fe_bolt(12.0, 1.75, _G109["Sp"], _G109["Sy"], _G109["Sut"], _G109["Se"],
                          axial_max=60000.0, axial_min=40000.0, preload=40000.0,
                          target_fos=1.5)
    assert ev["passes"] is False
    assert ev["proof_fos"] < 1.5          # the tension overload itself is below target
    assert ev["min_fos"] < 1.5


def test_evaluate_fe_bolt_no_alternating_is_infinite_fatigue():
    ev = evaluate_fe_bolt(12.0, 1.75, _G109["Sp"], _G109["Sy"], _G109["Sut"], _G109["Se"],
                          axial_max=30000.0, axial_min=30000.0, preload=20000.0, shear_max=0.0)
    assert ev["sigma_a_MPa"] == pytest.approx(0.0)
    assert ev["fatigue_fos"] == float('inf')


def test_evaluate_fe_bolt_shear_governs():
    ev = evaluate_fe_bolt(12.0, 1.75, _G109["Sp"], _G109["Sy"], _G109["Sut"], _G109["Se"],
                          axial_max=10000.0, axial_min=0.0, shear_max=60000.0, target_fos=1.5)
    assert ev["passes"] is False
    assert ev["min_fos"] < 1.0
    assert ev["governing"] in ("Shear", "Combined")


def test_evaluate_fe_rows_grade_lookup():
    rows = [{"bolt_id": "A", "diameter_mm": 12, "pitch_mm": 1.75, "bolt_grade": "Grade 8.8",
             "axial_force_max_N": 30000, "axial_force_min_N": 25000, "preload_N": 20000}]
    res = evaluate_fe_rows(rows, BOLT_MATERIALS_METRIC, fatigue_criterion="Goodman", target_fos=1.5)
    assert len(res) == 1 and res[0]["error"] == ""
    # Material resolved from the grade DB (Sp = 600).
    assert res[0]["proof_fos"] == pytest.approx(600 * 84.27 / 30000.0, rel=1e-3)
    # Strengths are exposed for the population graphics (Haigh / interaction).
    for key in ("Sp_MPa", "Sy_MPa", "Sut_MPa", "Se_MPa"):
        assert res[0][key] > 0


def test_evaluate_fe_rows_explicit_strengths_and_aliases():
    # Uses header aliases (d, p, axial_max) and explicit strengths (no grade).
    rows = [{"id": "A", "d": 12, "p": 1.75, "proof_MPa": 600, "yield_MPa": 640,
             "ultimate_MPa": 800, "endurance_MPa": 129, "axial_max": 30000, "axial_min": 25000}]
    res = evaluate_fe_rows(rows, {}, target_fos=1.5)
    assert res[0]["error"] == ""
    assert res[0]["proof_fos"] == pytest.approx(600 * 84.27 / 30000.0, rel=1e-3)


def test_evaluate_fe_rows_envelopes_shared_bolt_id():
    rows = [
        {"bolt_id": "B", "diameter_mm": 12, "pitch_mm": 1.75, "bolt_grade": "Grade 10.9",
         "axial_force_max_N": 30000, "axial_force_min_N": 25000, "shear_force_max_N": 2000},
        {"bolt_id": "B", "diameter_mm": 12, "pitch_mm": 1.75, "bolt_grade": "Grade 10.9",
         "axial_force_max_N": 40000, "axial_force_min_N": 20000, "shear_force_max_N": 5000},
    ]
    res = evaluate_fe_rows(rows, BOLT_MATERIALS_METRIC)
    assert len(res) == 1                       # one bolt, two load cases enveloped
    assert res[0]["axial_max_N"] == pytest.approx(40000.0)
    assert res[0]["axial_min_N"] == pytest.approx(20000.0)
    assert res[0]["shear_max_N"] == pytest.approx(5000.0)


def test_evaluate_fe_rows_flags_bad_rows_without_crashing():
    rows = [
        {"bolt_id": "good", "diameter_mm": 12, "pitch_mm": 1.75, "bolt_grade": "Grade 10.9",
         "axial_force_max_N": 30000},
        {"bolt_id": "no_pitch", "diameter_mm": 12, "bolt_grade": "Grade 10.9",
         "axial_force_max_N": 30000},
        {"bolt_id": "no_strength", "diameter_mm": 12, "pitch_mm": 1.75,
         "axial_force_max_N": 30000},
    ]
    res = evaluate_fe_rows(rows, BOLT_MATERIALS_METRIC)
    by_id = {r["bolt_id"]: r for r in res}
    assert by_id["good"]["error"] == ""
    assert "diameter or pitch" in by_id["no_pitch"]["error"]
    assert "strength" in by_id["no_strength"]["error"]


def test_preload_from_yield_percent():
    # 75% of a 600 MPa yield strength bolt with 100 mm^2 area
    # F = (75 / 100) * 600 * 100 = 45000 N
    result = preload_from_yield_percent(75.0, 600.0, 100.0)
    assert result == pytest.approx(45000.0)

    # 100% of 900 MPa, 50 mm^2 area
    result = preload_from_yield_percent(100.0, 900.0, 50.0)
    assert result == pytest.approx(45000.0)


def test_combined_tension_shear_fos():
    from mechanics import combined_tension_shear_fos
    At, Sp, Sy = 58.0, 600.0, 640.0
    shear_allow = 0.577 * Sy
    # Pure tension at proof, or pure shear at the allowable, each give FoS = 1.
    assert combined_tension_shear_fos(Sp * At, 0.0, At, Sp, Sy) == pytest.approx(1.0, rel=1e-6)
    assert combined_tension_shear_fos(0.0, shear_allow * At, At, Sp, Sy) == pytest.approx(1.0, rel=1e-6)
    # Half of each utilisation -> 1/sqrt(0.25 + 0.25) = sqrt(2); more severe than either alone (FoS 2).
    fos = combined_tension_shear_fos(0.5 * Sp * At, 0.5 * shear_allow * At, At, Sp, Sy)
    assert fos == pytest.approx(2.0 ** 0.5, rel=1e-3)
    # No load -> unbounded.
    assert combined_tension_shear_fos(0.0, 0.0, At, Sp, Sy) == float('inf')
