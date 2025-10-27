# Isaacus SageMaker Python integration
<a href="https://pypi.org/project/isaacus-sagemaker/" alt="PyPI Version"><img src="https://img.shields.io/pypi/v/isaacus-sagemaker"></a> <a href="https://github.com/isaacus-dev/isaacus-sagemaker-python/actions/workflows/ci.yml" alt="Build Status"><img src="https://img.shields.io/github/actions/workflow/status/isaacus-dev/isaacus-sagemaker-python/ci.yaml?branch=main"></a>

The Isaacus SageMaker Python integration enables users to interact with private SageMaker deployments of Isaacus legal AI models via the [Isaacus Python SDK](https://github.com/isaacus-dev/isaacus-python).

This integration only requires a single line of code to be added to existing Isaacus API-based applications.

If you're looking for our AWS Marketplace listings, you can find them [here](https://aws.amazon.com/marketplace/seller-profile?id=seller-5e4iuidabgujc). Additionally, we offer a [complete guide](https://docs.isaacus.com/integrations/amazon-sagemaker) on how to deploy and use Isaacus models on SageMaker on our docs.

## Installation 📦
This integration can be installed with `pip`:
```sh
pip install isaacus-sagemaker
```

It also requires the `isaacus` package to be of any use:
```sh
pip install isaacus
```

## Usage 👩‍💻
To use the Isaacus SageMaker integration, import either `IsaacusSageMakerRuntimeHTTPClient` (for synchronous usage) or `AsyncIsaacusSageMakerRuntimeHTTPClient` (for asynchronous usage) from `isaacus_sagemaker` along with `IsaacusSageMakerRuntimeEndpoint` to define available SageMaker endpoints.

Then, create an instance of the `Isaacus` or `AsyncIsaacus` client as you normally would, but also pass your SageMaker HTTP client as the `http_client` parameter.

Below is an example of how you'd do that in practice:

```python
from isaacus import Isaacus, AsyncIsaacus
from isaacus_sagemaker import IsaacusSageMakerRuntimeHTTPClient, AsyncIsaacusSageMakerRuntimeHTTPClient, \
     IsaacusSageMakerRuntimeEndpoint

endpoints = [
    IsaacusSageMakerRuntimeEndpoint(
        name="my-sagemaker-endpoint",
        # region="us-west-2", # Optional, defaults to the client or AWS SDK default region
        # profile="my-aws-profile", # Optional, defaults to the client or AWS SDK default profile
        # models=["kanon-2-embedder"], # Optional, models supported by this endpoint,
        #                              # defaults to all models
    )
]

client = Isaacus(
    http_client=IsaacusSageMakerRuntimeHTTPClient(
        endpoints=endpoints,
        # region="us-west-2", # Optional, defaults to AWS SDK default region
        # profile="my-aws-profile", # Optional, defaults to AWS SDK default profile
        # boto_session_kwargs={"aws_access_key_id": "...",}, # Optional, additional boto3 session kwargs
        # **{}, # Optional, additional httpx.Client kwargs
    )
)

# For asynchronous usage:
aclient = AsyncIsaacus(
    http_client=AsyncIsaacusSageMakerRuntimeHTTPClient(
        endpoints=endpoints,
        # region="us-west-2", # Optional, defaults to AWS SDK default region
        # profile="my-aws-profile", # Optional, defaults to AWS SDK default profile
        # boto_session_kwargs={"aws_access_key_id": "...",}, # Optional, additional boto3 session kwargs
        # **{}, # Optional, additional httpx.AsyncClient kwargs
    )
)
```

Since Isaacus SageMaker deployments are private and hosted within your AWS account, no API key or base URL needs to be provided when constructing Isaacus SDK clients.

Once you've set up your client, no further changes are needed to your existing code.

## API 🧩
### `IsaacusSageMakerRuntimeEndpoint`
```python
class IsaacusSageMakerRuntimeEndpoint(msgspec.Struct, frozen=True):
    name: str
    """The name of the SageMaker endpoint."""

    region: Optional[str] = None
    """The AWS region where the SageMaker endpoint is deployed. This overrides any region specified at the client-level."""

    profile: Optional[str] = None
    """The AWS profile to use when accessing the SageMaker endpoint. This overrides any profile specified at the client-level."""

    models: Optional[Sequence[str]] = None
    """The IDs of models served by this endpoint. If `None`, it is assumed the endpoint serves all models."""
```

### `IsaacusSageMakerRuntimeHTTPClient`
```python
class IsaacusSageMakerRuntimeHTTPClient(httpx.Client):
    """
    A synchronous Isaacus SDK-compatible HTTP client that proxies requests to SageMaker-deployed Isaacus models through the SageMaker Runtime InvokeEndpoint (`/invocations`) API.

    This client extends `httpx.Client`.

    Arguments:
        `endpoints` (`Sequence[IsaacusSageMakerRuntimeEndpoint]`): A sequence of SageMaker endpoints to route requests to.
        `region` (`str`, optional): The AWS region where the SageMaker endpoints are deployed. Overriden by any region specified at the endpoint-level. Defaults to `None`, in which case the AWS SDK's default region resolution is used.
        `profile` (`str`, optional): The AWS profile to use when accessing the SageMaker endpoints. Overriden by any profile specified at the endpoint-level. Defaults to `None`, in which case the AWS SDK's default profile resolution is used.
        `boto_session_kwargs` (`Dict[str, Any]`, optional): Additional keyword arguments to pass to the `boto3.Session` constructor when creating the AWS session. Defaults to `None`.
        `**httpx_kwargs` (`Any`): Additional keyword arguments to pass to the `httpx.Client` constructor.
    """
```

### `AsyncIsaacusSageMakerRuntimeHTTPClient`
```python
class AsyncIsaacusSageMakerRuntimeHTTPClient(httpx.AsyncClient):
    """
    An asynchronous Isaacus SDK-compatible HTTP client that proxies requests to SageMaker-deployed Isaacus models through the SageMaker Runtime InvokeEndpoint (`/invocations`) API.

    This client extends `httpx.AsyncClient`.

    Arguments:
        `endpoints` (`Sequence[IsaacusSageMakerRuntimeEndpoint]`): A sequence of SageMaker endpoints to route requests to.
        `region` (`str`, optional): The AWS region where the SageMaker endpoints are deployed. Overriden by any region specified at the endpoint-level. Defaults to `None`, in which case the AWS SDK's default region resolution is used.
        `profile` (`str`, optional): The AWS profile to use when accessing the SageMaker endpoints. Overriden by any profile specified at the endpoint-level. Defaults to `None`, in which case the AWS SDK's default profile resolution is used.
        `boto_session_kwargs` (`Dict[str, Any]`, optional): Additional keyword arguments to pass to the `boto3.Session` constructor when creating the AWS session. Defaults to `None`.
        `**httpx_kwargs` (`Any`): Additional keyword arguments to pass to the `httpx.Client` constructor.
    """
```

## How it works ⚙️
This integration works by intercepting Isaacus SDK HTTP requests and proxying them to an Isaacus API server through the SageMaker Runtime InvokeEndpoint (`/invocations`) API.

When a request is made through the Isaacus SDK client, the custom HTTP client obtains the path, method, data, and headers from the original request and constructs a new request to the SageMaker endpoint's `/invocations` API that packages the original request details like so:
```json
{
    "path": "/v1/embeddings",
    "method": "POST",
    "headers": {
        "Content-Type": "application/json",
        "Accept": "application/json"
    },
    "data": {
        "model": "kanon-2-embedder",
        "texts": ["This is a confidentiality clause."],
        "task": "retrieval/document"
    }
}
```

The Isaacus API server internally forwards the request to the appropriate internal endpoint and then SageMaker returns the response back to the custom HTTP client. If an error is encountered, SageMaker returns its own error response, which the custom HTTP client translates back into the original error, ensuring maximum compatibility with existing Isaacus SDK-based applications.

## Changelog 🔄
All notable changes to this integration are documented in the [CHANGELOG.md](CHANGELOG.md) file. This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## License 📄
In the spirit of open source, this integration is licensed under the [MIT License](LICENSE).