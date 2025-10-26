import os
import sys
import types
import pathlib
from typing import Any
import pytest

# Ensure `src` (editable dev layout) is importable when running from the repo root
try:
    import isaacus_sagemaker  # noqa: F401
except Exception:  # pragma: no cover - only runs in local dev
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if src.exists():
        sys.path.insert(0, str(src))

# Lightweight fakes for AWS signing and boto3 session ------------------------------------

class _DummyAWSRequest:
    """
    Drop-in-enough replacement for botocore.awsrequest.AWSRequest used in tests.
    Only the attributes accessed by the package code are implemented.
    """
    def __init__(self, *, method: str, url: str, data: bytes, headers: dict[str, str]):
        self.method = method
        self.url = url
        # start with a copy so the signer can mutate
        self.headers = dict(headers)
        self.data = data


class _DummySigV4Auth:
    """
    Very small stand-in for botocore.auth.SigV4Auth that simply stamps a header.
    """
    def __init__(self, credentials: Any, service_name: str, region_name: str) -> None:
        # Record what we were given for assertions if needed
        self.credentials = credentials
        self.service_name = service_name
        self.region_name = region_name

    # mimic the API of botocore.auth.SigV4Auth
    def add_auth(self, aws_request: _DummyAWSRequest) -> None:
        # Stamp an Authorization header so tests can assert it was "signed"
        aws_request.headers["Authorization"] = f"AWS4-HMAC-SHA256 Dummy region={self.region_name} service={self.service_name}"


class _DummyCreds:
    def __init__(self, access_key="AKID", secret_key="SECRET", token="TOKEN"):
        self.access_key = access_key
        self.secret_key = secret_key
        self.token = token


class _DummyBoto3Session:
    # class-level counter to let tests assert we only build a session once per profile
    constructed = 0

    def __init__(self, **kwargs: Any) -> None:
        _DummyBoto3Session.constructed += 1
        # allow explicit region override; otherwise default
        self.region_name = kwargs.get("region_name") or "us-east-1"
        self.profile_name = kwargs.get("profile_name")

    def get_credentials(self):
        return _DummyCreds()


@pytest.fixture
def fake_aws(monkeypatch):
    """
    Monkeypatch AWSRequest, SigV4Auth and boto3.Session to predictable dummies.
    """
    import isaacus_sagemaker.runtime_client as rc

    monkeypatch.setattr(rc, "AWSRequest", _DummyAWSRequest, raising=True)
    monkeypatch.setattr(rc, "SigV4Auth", _DummySigV4Auth, raising=True)

    # Patch boto3.Session used by the runtime client
    import boto3
    monkeypatch.setattr(boto3, "Session", _DummyBoto3Session, raising=True)

    # Reset counter before each test that uses this fixture
    _DummyBoto3Session.constructed = 0

    return {
        "AWSRequest": _DummyAWSRequest,
        "SigV4Auth": _DummySigV4Auth,
        "Boto3Session": _DummyBoto3Session,
    }


@pytest.fixture
def endpoints():
    from isaacus_sagemaker.types import IsaacusSageMakerRuntimeEndpoint

    return [
        # default endpoint (serves all models)
        IsaacusSageMakerRuntimeEndpoint(name="default-endpoint", region="ap-southeast-2"),
        # model-specific endpoint
        IsaacusSageMakerRuntimeEndpoint(name="embed-endpoint", region="ap-southeast-2", models=["kanon-2-embedder"]),
    ]

