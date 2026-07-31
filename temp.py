from mechanics import calculate_preload, BOLT_MATERIALS_METRIC
layers = [
    {"Material": "Steel (Mild)", "thickness": 10.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6},
    {"Material": "Steel (Mild)", "thickness": 10.0, "Syc": 250, "E": 200000, "CTE": 11.5e-6}
]
res = calculate_preload(10.0, 1.5, BOLT_MATERIALS_METRIC['Grade 8.8'], layers, 'Hex Head', False, False, 'Dry / as-received (K=0.20)')
print(f'kb: {res["kb_N_mm"]:.3f}, km: {res["km_N_mm"]:.3f}, C: {res["joint_constant_C"]:.4f}')
