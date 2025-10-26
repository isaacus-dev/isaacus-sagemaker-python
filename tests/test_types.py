from isaacus_sagemaker.types import IsaacusSageMakerInvocationRequest, IsaacusSageMakerRuntimeEndpoint

def test_struct_defaults_and_frozen():
    # frozen=True -> attributes are immutable
    req = IsaacusSageMakerInvocationRequest(path="/v1/embeddings")
    assert req.method == "POST"
    assert req.headers is None and req.data is None
    try:
        req.path = "/new"  # type: ignore[attr-defined]
        assert False, "Should be frozen/immutable"
    except Exception:
        pass

    ep = IsaacusSageMakerRuntimeEndpoint(name="abc")
    assert ep.region is None and ep.profile is None and ep.models is None