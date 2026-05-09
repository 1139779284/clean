from model_security_gate.t0.stats import required_zero_failure_n, wilson_interval, zero_failure_upper_bound


def test_wilson_interval_bounds():
    out = wilson_interval(1, 10)
    assert out.total == 10
    assert 0.0 <= out.low <= out.rate <= out.high <= 1.0


def test_zero_failure_bound_and_required_n():
    ub = zero_failure_upper_bound(300, confidence=0.95)
    assert 0.0 < ub < 0.02
    assert required_zero_failure_n(0.01, confidence=0.95) > 250
