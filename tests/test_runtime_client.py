import os
import json
import httpx
import msgspec
import pytest

from isaacus_sagemaker.runtime_client import (
    _extract_body,
    _get_model,
    _strip_unproxiable_headers,
    _make_payload,
    _translate_response,
    _build_request,
    IsaacusSageMakerRuntimeHTTPClient,
    AsyncIsaacusSageMakerRuntimeHTTPClient,
)
from isaacus_sagemaker.types import IsaacusSageMakerRuntimeEndpoint


def test_extract_body_accepts_json_and_rejects_others():
    req_ok = httpx.Request(
        method="POST",
        url="https://api.isaacus.com/v1/embeddings",
        headers={"content-type": "application/json"},
        content=b'{"hello":"world"}',
    )
    assert _extract_body(req_ok) == {"hello": "world"}

    # Non-JSON content types are not supported and should raise
    req_bad = httpx.Request(
        method="POST",
        url="https://api.isaacus.com/v1/embeddings",
        headers={"content-type": "text/plain"},
        content=b"nope",
    )
    with pytest.raises(ValueError):
        _extract_body(req_bad)


def test_get_model_from_body():
    assert _get_model({"model": "kanon-2-embedder"}) == "kanon-2-embedder"
    assert _get_model({"not_model": "x"}) is None
    assert _get_model(None) is None  # type: ignore[arg-type]


def test_strip_unproxiable_headers():
    headers = {
        "Authorization": "Bearer abc",
        "Accept": "application/json",
        "Host": "api.isaacus.com",
        "X-Custom": "1",
        "Content-Length": "123",
    }
    out = _strip_unproxiable_headers(headers)
    assert "Authorization" not in out and "Host" not in out and "Content-Length" not in out
    assert out["Accept"] == "application/json"
    assert out["X-Custom"] == "1"


def test_make_payload_is_msgspec_json():
    req = httpx.Request(
        "POST",
        "https://api.isaacus.com/v1/embeddings",
        headers={"content-type": "application/json", "accept": "application/json"},
        content=b'{"model":"m","x":1}',
    )
    body = _extract_body(req)
    payload = _make_payload(req, body)
    decoded = msgspec.json.decode(payload)
    assert decoded["path"] == "/v1/embeddings"
    assert decoded["method"] == "POST"
    assert decoded["data"] == {"model": "m", "x": 1}
    # headers should not contain authorization/host/etc
    hdrs = {k.lower(): v for k, v in decoded["headers"].items()}
    assert "authorization" not in hdrs and "host" not in hdrs
    assert hdrs["accept"] == "application/json"


def test_build_request_signs_and_targets_sagemaker(fake_aws, endpoints):
    client = IsaacusSageMakerRuntimeHTTPClient(endpoints=endpoints, region="ap-southeast-2")
    # ensure env var is injected when absent
    os.environ.pop("ISAACUS_API_KEY", None)
    client2 = IsaacusSageMakerRuntimeHTTPClient(endpoints=endpoints, region="ap-southeast-2")
    assert os.environ.get("ISAACUS_API_KEY") == "ISAACUS_SAGEMAKER_DOES_NOT_NEED_AN_API_KEY"

    # Build an original request
    req = httpx.Request(
        "POST",
        "https://api.isaacus.com/v1/embeddings",
        headers={"content-type": "application/json", "accept": "application/json", "authorization": "Bearer abc"},
        content=b'{"model":"kanon-2-embedder","texts":["a"]}',
    )
    proxied = _build_request(client, req)

    # Target SageMaker invoke endpoint
    assert proxied.url.host.startswith("runtime.sagemaker.") and proxied.url.path.endswith("/invocations")
    assert proxied.headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256 Dummy")
    assert proxied.headers["Content-Type"] == "application/json"
    assert proxied.headers["Accept"] == "application/json"

    # Payload should be the invocation struct JSON
    decoded = msgspec.json.decode(proxied.content)
    assert decoded["path"] == "/v1/embeddings"
    assert decoded["data"]["texts"] == ["a"]
    # ensure unproxiable headers were stripped from the inner payload
    inner_hdrs = {k.lower(): v for k, v in decoded["headers"].items()}
    assert "authorization" not in inner_hdrs


def test_build_request_raises_when_no_endpoint_for_model(fake_aws):
    client = IsaacusSageMakerRuntimeHTTPClient(
        endpoints=[IsaacusSageMakerRuntimeEndpoint(name="only-default", region="us-east-1", models=None)]
    )
    # Ask for a model *explicitly* that is not known and no default endpoints exist for it
    # (we simulate this by constructing a router with only a model-specific endpoint)
    client_only_model = IsaacusSageMakerRuntimeHTTPClient(
        endpoints=[IsaacusSageMakerRuntimeEndpoint(name="model-only", region="us-east-1", models=["m1"])]
    )
    req = httpx.Request(
        "POST",
        "https://api.isaacus.com/v1/embeddings",
        headers={"content-type": "application/json"},
        content=b'{"model":"mX"}',
    )
    with pytest.raises(RuntimeError):
        _build_request(client_only_model, req)

    # sanity: with a default endpoint it should not raise
    _build_request(client, req)


def test_translate_response_passes_through_success_and_maps_wrapped_error():
    # 200 OK passthrough
    ok_resp = httpx.Response(200, json={"ok": True}, request=httpx.Request("GET", "https://x"))
    assert _translate_response(ok_resp) is ok_resp

    # Wrapped SageMaker error -> unwrap to original
    wrapped = httpx.Response(
        502,
        headers={"content-type": "application/json"},
        json={
            "OriginalStatusCode": 404,
            "OriginalMessage": json.dumps({"error": "Not Found"}),
        },
        request=httpx.Request("POST", "https://x"),
    )
    out = _translate_response(wrapped)
    assert out.status_code == 404
    assert out.headers["content-type"] == "application/json"
    assert out.json() == {"error": "Not Found"}


def test_sync_client_send_builds_and_translates(monkeypatch, fake_aws, endpoints):
    # Arrange: capture the request the base class would "send"
    sent = {}

    def fake_super_send(self, req, *args, **kwargs):
        # record what was sent
        sent["req"] = req
        # respond with a wrapped error to exercise translation
        return httpx.Response(
            500,
            headers={"content-type": "application/json"},
            json={
                "OriginalStatusCode": 400,
                "OriginalMessage": json.dumps({"error": "bad request"}),
            },
            request=req,
        )

    # Patch the base httpx.Client.send
    monkeypatch.setattr(httpx.Client, "send", fake_super_send, raising=True)

    client = IsaacusSageMakerRuntimeHTTPClient(endpoints=endpoints, region="ap-southeast-2")

    original = httpx.Request(
        "POST",
        "https://api.isaacus.com/v1/embeddings",
        headers={"content-type": "application/json"},
        content=b'{"model":"kanon-2-embedder","texts":["t"]}',
    )
    resp = client.send(original)
    assert resp.status_code == 400
    assert resp.json() == {"error": "bad request"}

    # Ensure we actually proxied to SageMaker
    assert sent["req"].url.host.startswith("runtime.sagemaker.")


@pytest.mark.asyncio
async def test_async_client_skips_examples_host(monkeypatch, fake_aws, endpoints):
    # If targeting examples.isaacus.com we should *not* proxy or sign
    async def fake_async_send(self, req, *args, **kwargs):
        return httpx.Response(200, json={"ok": True}, request=req)

    # Guard: if _build_request is accidentally called, raise
    def boom(*args, **kwargs):  # pragma: no cover - ensures we don't proxy
        raise AssertionError("Should not have proxied requests to examples.isaacus.com")

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_async_send, raising=True)

    # Also guard against accidental proxy
    import isaacus_sagemaker.runtime_client as rc
    monkeypatch.setattr(rc, "_build_request", boom, raising=True)

    client = AsyncIsaacusSageMakerRuntimeHTTPClient(endpoints=endpoints, region="ap-southeast-2")

    original = httpx.Request(
        "GET",
        "https://examples.isaacus.com/v1/echo",
    )
    resp = await client.send(original)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_get_creds_cached_by_profile(fake_aws, endpoints):
    import isaacus_sagemaker.runtime_client as rc

    # Construct a client with a *default* profile and one endpoint overriding it
    client = IsaacusSageMakerRuntimeHTTPClient(endpoints=endpoints, profile="dev", region="ap-southeast-2")

    # Two different requests should only construct a single boto3.Session per distinct profile key
    req1 = httpx.Request(
        "POST",
        "https://api.isaacus.com/v1/embeddings",
        headers={"content-type": "application/json"},
        content=b'{"model":"kanon-2-embedder"}',
    )
    req2 = httpx.Request(
        "POST",
        "https://api.isaacus.com/v1/embeddings",
        headers={"content-type": "application/json"},
        content=b'{"model":"kanon-2-embedder"}',
    )
    _ = _build_request(client, req1)
    _ = _build_request(client, req2)

    # Only one session constructed for the same profile
    assert fake_aws["Boto3Session"].constructed == 1
