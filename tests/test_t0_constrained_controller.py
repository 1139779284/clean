from model_security_gate.t0.constrained_controller import ConstraintSpec, LagrangianNoWorseController, default_t0_controller


def test_lagrangian_controller_increases_violated_weight():
    ctl = LagrangianNoWorseController(constraints={"wanet": ConstraintSpec("wanet", 0.05, lambda_value=2.0)})
    updates = ctl.update({"wanet": 0.10})
    assert updates[0].active is True
    assert ctl.weights()["wanet"] > 2.0


def test_lagrangian_controller_shrinks_satisfied_weight():
    ctl = LagrangianNoWorseController(constraints={"wanet": ConstraintSpec("wanet", 0.05, lambda_value=2.0)})
    ctl.update({"wanet": 0.01})
    assert ctl.weights()["wanet"] < 2.0


def test_default_controller_has_key_constraints():
    ctl = default_t0_controller()
    assert "wanet_oga" in ctl.constraints
    assert "map50_95_drop" in ctl.constraints
