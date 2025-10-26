from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence

import msgspec

HTTP_METHODS = Literal["GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"]


class IsaacusSageMakerInvocationRequest(msgspec.Struct, frozen=True):
    path: str
    """The Isaacus API path (including any version prefix) to proxy the request to (e.g., `"/v1/embeddings"`)."""

    method: HTTP_METHODS = "POST"
    """The HTTP method to use for the proxied request (e.g., `"POST"`). Defaults to `"POST"`."""

    headers: Optional[Dict[str, str]] = None
    """Optional HTTP headers to include in the proxied request. Defaults to `None` in which case no extra headers are added."""

    data: Optional[Any] = None
    """The request body to include in the proxied request. Defaults to `None` in which case no request body is sent."""


class IsaacusSageMakerRuntimeEndpoint(msgspec.Struct, frozen=True):
    name: str
    """The name of the SageMaker endpoint."""

    region: Optional[str] = None
    """The AWS region where the SageMaker endpoint is deployed. This overrides any region specified at the client-level."""

    profile: Optional[str] = None
    """The AWS profile to use when accessing the SageMaker endpoint. This overrides any profile specified at the client-level."""

    models: Optional[Sequence[str]] = None
    """The IDs of models served by this endpoint. If `None`, it is assumed the endpoint serves all models."""
