import argparse

from model_security_gate.utils.config import deep_merge, namespace_overrides


def test_deep_merge_skips_none_and_merges_nested():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    override = {"a": None, "b": {"y": 3}, "c": 4}
    assert deep_merge(base, override) == {"a": 1, "b": {"x": 1, "y": 3}, "c": 4}


def test_namespace_overrides_only_keeps_non_none():
    ns = argparse.Namespace(config="cfg.yaml", model="m.pt", images=None, flag=False)
    assert namespace_overrides(ns, exclude={"config"}) == {"model": "m.pt", "flag": False}
