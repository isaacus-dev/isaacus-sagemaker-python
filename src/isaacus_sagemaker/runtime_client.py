from __future__ import annotations

import os

from typing import Any, Dict, Union, Mapping, Optional, Sequence

import boto3
import httpx
import orjson
import msgspec
import botocore.credentials

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from .types import IsaacusSageMakerRuntimeEndpoint, IsaacusSageMakerInvocationRequest
from ._router import Router

_UNPROXIABLE_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "authorization",
}


def _extract_body(message: Union[httpx.Request, httpx.Response]) -> Any:
    content_type = message.headers.get("content-type", "")

    if "application/json" in content_type:
        body = orjson.loads(message.content)

    else:
        body = message.content

        if body:
            raise ValueError("Only message bodies with the content type `application/json` are supported.")

    return body


def _get_model(body: Any) -> Optional[str]:
    if isinstance(body, dict) and "model" in body:
        return body["model"]

    return None


def _strip_unproxiable_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _UNPROXIABLE_HEADERS}


def _make_payload(
    req: httpx.Request,
    body: Any,
) -> bytes:
    invocation = IsaacusSageMakerInvocationRequest(
        path=req.url.path,
        method=req.method,
        headers=_strip_unproxiable_headers(req.headers),
        data=body,
    )
    payload = msgspec.json.encode(invocation)

    return payload


def _translate_response(resp: httpx.Response) -> httpx.Response:
    # Pass through non-error responses as-is.
    if resp.status_code < 400:
        return resp

    # Attempt to parse the response body as JSON otherwise return the original response.
    content_type = resp.headers.get("content-type", "")
    payload: Optional[Dict] = None

    if "application/json" in content_type:
        payload = orjson.loads(resp.content)

    if not isinstance(payload, dict):
        return resp

    # Attempt to extract the original status code and message from the payload.
    original_status = payload.get("OriginalStatusCode")

    if not original_status:
        return resp

    original_message = payload.get("OriginalMessage")

    if not original_message:
        return resp

    # Attempt to parse the original message as JSON.
    try:
        original_message = orjson.loads(original_message)

    except Exception:
        return resp

    # Build the translated response.
    headers = resp.headers.copy()
    headers["content-type"] = "application/json"

    return httpx.Response(
        status_code=int(original_status),
        headers=headers,
        json=original_message,
        request=resp.request,
    )


def _get_creds(
    endpoint: IsaacusSageMakerRuntimeEndpoint,
    creds: Dict[Optional[str], botocore.credentials.Credentials],
    default_profile: Optional[str] = None,
    session_kwargs: Optional[Dict[str, Any]] = None,
) -> botocore.credentials.Credentials:
    profile = endpoint.profile or default_profile

    if profile not in creds:
        session_kwargs = dict(session_kwargs or {})
        
        if profile is not None:
            session_kwargs["profile_name"] = profile

        session = boto3.Session(**session_kwargs)
        endpoint_creds = session.get_credentials()

        if endpoint_creds is None:
            raise RuntimeError(f"Could not retrieve AWS credentials for profile '{profile}'.")

        creds[profile] = endpoint_creds

    return creds[profile]


def _build_request(
    self: IsaacusSageMakerRuntimeHTTPClient | AsyncIsaacusSageMakerRuntimeHTTPClient,
    req: httpx.Request,
) -> httpx.Request:
    # Extract the request body.
    req_body = _extract_body(req)

    # Determine the model to route the request to if one was specified.
    model = _get_model(req_body)

    # Pick an endpoint to route the request to.
    endpoint = self._router.pick(model)

    if endpoint is None:
        raise RuntimeError(f"No SageMaker endpoints registered for model '{model}'.")

    # Construct the SageMaker invocation payload.
    req_payload = _make_payload(req, req_body)

    # Build the request headers.
    req_headers = {
        "Content-Type": "application/json",
        "Accept": req.headers.get("accept", "application/json"),
    }

    # Construct the URL for the request.
    region = endpoint.region or self._region
    assert region is not None, (
        "AWS region must be specified either at the endpoint-level, client-level or environment-level."
    )
    url = f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint.name}/invocations"

    # Get AWS credentials for the endpoint.
    creds = _get_creds(endpoint, self._creds, default_profile=self._profile, session_kwargs=self._boto_session_kwargs)

    # Sign the request payload with AWS SigV4.
    req_for_sigv4 = AWSRequest(method="POST", url=url, data=req_payload, headers=req_headers)
    SigV4Auth(credentials=creds, service_name="sagemaker", region_name=region).add_auth(req_for_sigv4)
    signed_req_headers = {k: str(v) for k, v in req_for_sigv4.headers.items()}

    # Build the request.
    req = self.build_request(
        method="POST",
        url=url,
        headers=signed_req_headers,
        data=req_payload,
    )

    return req


class IsaacusSageMakerRuntimeHTTPClient(httpx.Client):
    """
    A synchronous Isaacus SDK-compatible HTTP client that proxies requests to SageMaker-deployed Isaacus models through the SageMaker Runtime InvokeEndpoint (`/invocations`) API.

    This client extends `httpx.AsyncClient`.

    Arguments:
        `endpoints` (`Sequence[IsaacusSageMakerRuntimeEndpoint]`): A sequence of SageMaker endpoints to route requests to.
        `region` (`str`, optional): The AWS region where the SageMaker endpoints are deployed. Overriden by any region specified at the endpoint-level. Defaults to `None`, in which case the AWS SDK's default region resolution is used.
        `profile` (`str`, optional): The AWS profile to use when accessing the SageMaker endpoints. Overriden by any profile specified at the endpoint-level. Defaults to `None`, in which case the AWS SDK's default profile resolution is used.
        `boto_session_kwargs` (`Dict[str, Any]`, optional): Additional keyword arguments to pass to the `boto3.Session` constructor when creating the AWS session. Defaults to `None`.
        `**httpx_kwargs` (`Any`): Additional keyword arguments to pass to the `httpx.Client` constructor.
    """

    def __init__(
        self,
        *,
        endpoints: Sequence[IsaacusSageMakerRuntimeEndpoint],
        region: Optional[str] = None,
        profile: Optional[str] = None,
        boto_session_kwargs: Optional[Dict[str, Any]] = None,
        **httpx_kwargs: Any,
    ) -> None:
        super().__init__(**httpx_kwargs)

        self._router = Router(endpoints)

        self._boto_session_kwargs: Dict[str, Any] = boto_session_kwargs or {}
        self._region = (
            region
            or boto3.Session(**self._boto_session_kwargs | ({"profile_name": profile} if profile else {})).region_name
        )
        self._profile = profile
        self._creds: Dict[Optional[str], botocore.credentials.Credentials] = {}

        # If ISAACUS_API_KEY has not been set, set it to `ISAACUS_SAGEMAKER_DOES_NOT_NEED_AN_API_KEY` to avoid errors.
        if "ISAACUS_API_KEY" not in os.environ:
            os.environ["ISAACUS_API_KEY"] = "ISAACUS_SAGEMAKER_DOES_NOT_NEED_AN_API_KEY"

    def send(
        self,
        req: httpx.Request,
        *args: Any,
        **kwargs: Any,
    ) -> httpx.Response:
        # Skip proxying requests to `examples.isaacus.com` (to ensure SDK examples still work as expected).
        if req.url.host.endswith("examples.isaacus.com"):
            return super().send(req, *args, **kwargs)

        # Build the request.
        req = _build_request(self, req)

        # Send the request.
        resp = super().send(req, *args, **kwargs)

        # Translate the response.
        resp = _translate_response(resp)

        return resp


class AsyncIsaacusSageMakerRuntimeHTTPClient(httpx.AsyncClient):
    """
    An asynchronous Isaacus SDK-compatible HTTP client that proxies requests to SageMaker-deployed Isaacus models through the SageMaker Runtime InvokeEndpoint (`/invocations`) API.

    This client extends `httpx.Client`.

    Arguments:
        `endpoints` (`Sequence[IsaacusSageMakerRuntimeEndpoint]`): A sequence of SageMaker endpoints to route requests to.
        `region` (`str`, optional): The AWS region where the SageMaker endpoints are deployed. Overriden by any region specified at the endpoint-level. Defaults to `None`, in which case the AWS SDK's default region resolution is used.
        `profile` (`str`, optional): The AWS profile to use when accessing the SageMaker endpoints. Overriden by any profile specified at the endpoint-level. Defaults to `None`, in which case the AWS SDK's default profile resolution is used.
        `boto_session_kwargs` (`Dict[str, Any]`, optional): Additional keyword arguments to pass to the `boto3.Session` constructor when creating the AWS session. Defaults to `None`.
        `**httpx_kwargs` (`Any`): Additional keyword arguments to pass to the `httpx.Client` constructor.
    """

    def __init__(
        self,
        *,
        endpoints: Sequence[IsaacusSageMakerRuntimeEndpoint],
        region: Optional[str] = None,
        profile: Optional[str] = None,
        boto_session_kwargs: Optional[Dict[str, Any]] = None,
        **httpx_kwargs: Any,
    ) -> None:
        super().__init__(**httpx_kwargs)

        self._router = Router(endpoints)

        self._boto_session_kwargs: Dict[str, Any] = boto_session_kwargs or {}
        self._region = (
            region
            or boto3.Session(**self._boto_session_kwargs | ({"profile_name": profile} if profile else {})).region_name
        )
        self._profile = profile
        self._creds: Dict[Optional[str], botocore.credentials.Credentials] = {}

        # If ISAACUS_API_KEY has not been set, set it to `ISAACUS_SAGEMAKER_DOES_NOT_NEED_AN_API_KEY` to avoid errors.
        if "ISAACUS_API_KEY" not in os.environ:
            os.environ["ISAACUS_API_KEY"] = "ISAACUS_SAGEMAKER_DOES_NOT_NEED_AN_API_KEY"

    async def send(
        self,
        req: httpx.Request,
        *args: Any,
        **kwargs: Any,
    ) -> httpx.Response:
        # Skip proxying requests to `examples.isaacus.com` (to ensure SDK examples still work as expected).
        if req.url.host.endswith("examples.isaacus.com"):
            return await super().send(req, *args, **kwargs)

        # Build the request.
        req = _build_request(self, req)

        # Send the request.
        resp = await super().send(req, *args, **kwargs)

        # Translate the response.
        resp = _translate_response(resp)

        return resp
