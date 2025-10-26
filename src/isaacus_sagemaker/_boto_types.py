from typing import IO, Union, Optional, Protocol, TypedDict

from botocore.response import StreamingBody


class InvokeEndpointOutput(TypedDict, total=False):
    Body: StreamingBody
    ContentType: str
    InvokedProductionVariant: str
    CustomAttributes: str


class SageMakerRuntimeLike(Protocol):
    def invoke_endpoint(
        self,
        *,
        EndpointName: str,
        Body: Union[bytes, bytearray, IO[bytes]],
        ContentType: Optional[str] = None,
        Accept: Optional[str] = None,
        CustomAttributes: Optional[str] = None,
        TargetModel: Optional[str] = None,
        TargetVariant: Optional[str] = None,
        TargetContainerHostname: Optional[str] = None,
        InferenceId: Optional[str] = None,
        EnableExplanations: Optional[str] = None,
        InferenceComponentName: Optional[str] = None,
    ) -> InvokeEndpointOutput: ...
