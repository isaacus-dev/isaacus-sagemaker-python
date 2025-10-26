from isaacus_sagemaker._router import Router
from isaacus_sagemaker.types import IsaacusSageMakerRuntimeEndpoint


def test_router_requires_endpoints():
    try:
        Router([])
        assert False, "Expected assertion when no endpoints provided"
    except AssertionError:
        pass


def test_router_round_robin_with_model_and_default_mix():
    default = IsaacusSageMakerRuntimeEndpoint(name="default", region="us-east-1")
    # model-specific endpoint for "m1"
    m1a = IsaacusSageMakerRuntimeEndpoint(name="m1-a", region="us-east-1", models=["m1"])

    r = Router([default, m1a])

    # For model "m1" we should alternate between [m1a, default, m1a, default, ...]
    picks = [r.pick("m1").name for _ in range(6)]
    assert picks[:4] == ["m1-a", "default", "m1-a", "default"]

    # For some other model -> only defaults (here: a single one)
    assert r.pick("mX").name == "default"

    # When model is None -> from the defaults RR
    assert r.pick(None).name == "default"


def test_router_validates_endpoint_types_and_names():
    try:
        Router([object()])  # type: ignore[arg-type]
        assert False, "Expected assertion for wrong endpoint types"
    except AssertionError:
        pass

    try:
        Router([IsaacusSageMakerRuntimeEndpoint(name="", region="us-east-1")])
        assert False, "Expected assertion for empty endpoint name"
    except AssertionError:
        pass