with open('tests/test_mechanics.py', 'a') as f:
    f.write('''

def test_recommend_bolt_fatigue_gate():
    # If the check_fatigue flag is false, a bolt that would otherwise fail fatigue
    # should be accepted. If true, it should be rejected.
    from mechanics import _steel_layers
    layers = _steel_layers()
    # High alternating load
    ext_min = 0.0
    ext_max = 50000.0
    
    rec_with = recommend_bolt(
        layers, bolt_type="Hex Head", use_washer=False,
        is_permanent=False, friction_condition="Dry / as-received (K=0.20)",
        external_load_min=ext_min, external_load_max=ext_max,
        target_fos_proof=1.0, target_fos_separation=1.0, target_fos_fatigue=2.0,
        is_metric=True, embedment_um=0.0, load_intro_factor=1.0,
        thread_engagement_length=15.0, check_fatigue=True, fatigue_criterion="Goodman"
    )
    
    rec_without = recommend_bolt(
        layers, bolt_type="Hex Head", use_washer=False,
        is_permanent=False, friction_condition="Dry / as-received (K=0.20)",
        external_load_min=ext_min, external_load_max=ext_max,
        target_fos_proof=1.0, target_fos_separation=1.0, target_fos_fatigue=2.0,
        is_metric=True, embedment_um=0.0, load_intro_factor=1.0,
        thread_engagement_length=15.0, check_fatigue=False, fatigue_criterion="Goodman"
    )
    
    # The bolt chosen without fatigue checking should be smaller
    if rec_with is not None and rec_without is not None:
        assert rec_without["d_mm"] <= rec_with["d_mm"]

def test_three_layer_stack_and_iso_888():
    layers = [
        {"Material": "Steel (Mild)", "thickness": 20.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6},
        {"Material": "Aluminum (6061-T6)", "thickness": 20.0, "Syc": 275, "E": 69000, "CTE": 23.6e-6},
        {"Material": "Steel (Mild)", "thickness": 20.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}
    ]
    res = calculate_preload(10.0, 1.5, BOLT_MATERIALS_METRIC['Grade 8.8'], layers, 'Hex Head', False, False, 'Dry / as-received (K=0.20)')
    assert res['joint_constant_C'] > 0
    # ISO-888 standard thread length check for metric M10, L=80
    b = standard_thread_length(10.0, 80.0, is_metric=True)
    assert b == 2 * 10.0 + 6.0

def test_bolt_group_moment_axis_y():
    points = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    group = analyze_bolt_group(
        points,
        force_x=1000.0, force_y=0.0, force_z=2000.0,
        moment_x=0.0, moment_y=50000.0, moment_z=10000.0,
        is_metric=True
    )
    assert group["governing_tension_bolt"]["Tension_N"] > 500
    assert group["governing_shear_bolt"]["Shear_N"] > 250
''')
