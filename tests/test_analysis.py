from analysis import AnalysisContext, DisplayUnits, collect_findings, fos_str


def _result(**over):
    """A minimal result dict carrying just the keys collect_findings reads, all
    set so every check passes by default; override individual keys per test."""
    base = dict(
        crushing_warning_material="",
        proof_load_N=100000.0,
        thread_shear_fos=float('inf'),
        required_engagement_mm=10.0,
        fatigue_fos=float('inf'),
        separation_fos=float('inf'),
        proof_fos=2.0,
        tightening_utilization=0.5,
        tightening_stress_MPa=300.0,
        embedment_loss_N=0.0,
        recommended_preload_N=10000.0,
    )
    base.update(over)
    return base


_UNITS = DisplayUnits(1.0, 1.0, 1.0, "N", "mm", "MPa")


def _findings(res, **over):
    kw = dict(scatter=0.25, preload_disp=10000.0, preload_hi=12000.0,
              required_fos=1.5, fatigue_criterion="Goodman", has_internal=False,
              ext_max_N=0.0, slip_fos=None, combined_fos=None, embedment_um=0.0)
    kw.update(over)
    ctx = AnalysisContext(**kw)
    return collect_findings(res, _UNITS, ctx)


def test_fos_str():
    assert fos_str(float('inf')) == "∞"
    assert fos_str(1.5) == "1.50"
    assert fos_str(2.0) == "2.00"


def test_collect_findings_clean_joint_has_none():
    findings, warnings = _findings(_result())
    assert findings == []
    assert warnings == []


def test_collect_findings_flags_proof_and_fatigue_failures():
    findings, warnings = _findings(_result(proof_fos=0.8, fatigue_fos=0.5))
    severities = {sev for sev, _ in findings}
    assert severities == {"error"}
    msgs = " ".join(m for _, m in findings)
    assert "Bolt Yield Risk" in msgs
    assert "Fatigue Failure Risk" in msgs
    # report_warnings mirror the findings with markdown bold stripped.
    assert len(warnings) == len(findings)
    assert all("**" not in w for w in warnings)


def test_collect_findings_tightening_overload_is_advisory():
    # preload_hi above the proof load is a warning, not an error.
    findings, _ = _findings(_result(proof_load_N=11000.0), preload_hi=12000.0)
    assert ("warn", ) == tuple({sev for sev, _ in findings})
    assert any("Tightening Overload Risk" in m for _, m in findings)


def test_collect_findings_thread_checks_require_internal_material():
    res = _result(thread_shear_fos=0.5)
    # Without an internal thread material the stripping check is skipped entirely.
    assert _findings(res, has_internal=False)[0] == []
    # With one, a sub-1.0 thread-shear FoS is an error.
    findings, _ = _findings(res, has_internal=True)
    assert any(sev == "error" and "Thread Stripping Risk" in m for sev, m in findings)


def test_collect_findings_crushing_is_advisory():
    findings, _ = _findings(_result(crushing_warning_material="Layer 1"))
    assert any(sev == "warn" and "Material Crushing Risk" in m for sev, m in findings)


def test_collect_findings_separation_only_with_external_load():
    res = _result(separation_fos=0.5)
    # No external load -> separation check is not applicable.
    assert _findings(res, ext_max_N=0.0)[0] == []
    findings, _ = _findings(res, ext_max_N=5000.0)
    assert any(sev == "error" and "Joint Separation Risk" in m for sev, m in findings)


def test_collect_findings_slip_failure():
    findings, _ = _findings(_result(), slip_fos=0.5)
    assert any(sev == "error" and "Joint Slip Risk" in m for sev, m in findings)
    # slip_fos None -> no slip finding.
    assert not any("Joint Slip Risk" in m for _, m in _findings(_result(), slip_fos=None)[0])


def test_collect_findings_tightening_yield_vs_high_utilisation():
    over = _findings(_result(tightening_utilization=1.1))[0]
    assert any(sev == "error" and "Tightening Yield Risk" in m for sev, m in over)
    high = _findings(_result(tightening_utilization=0.95))[0]
    assert any(sev == "warn" and "High Tightening Utilization" in m for sev, m in high)


def test_collect_findings_embedment_loss_risk():
    res = _result(embedment_loss_N=10000.0, recommended_preload_N=9000.0)
    findings, _ = _findings(res, embedment_um=5.0)
    assert any(sev == "error" and "Embedment Loss Risk" in m for sev, m in findings)
    # No embedment input -> no finding even if the (stale) loss figure is high.
    assert not any("Embedment Loss Risk" in m for _, m in _findings(res, embedment_um=0.0)[0])


def test_collect_findings_combined_tension_shear():
    findings, _ = _findings(_result(), combined_fos=0.7)
    assert any(sev == "error" and "Combined Tension-Shear Risk" in m for sev, m in findings)
    # None (not applicable) -> no combined finding.
    assert not any("Combined Tension-Shear Risk" in m for _, m in _findings(_result(), combined_fos=None)[0])
