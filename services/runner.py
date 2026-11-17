# services/runner.py
from backends.api_backend import OfficialAPIBackend
from backends.base import InferenceRequest, NormalizedResult
from backends.sdk_backend import LocalSDKBackend


def run_inference(req: InferenceRequest) -> NormalizedResult:
    if req.mode == "api":
        backend = OfficialAPIBackend(token=req.params.get("token"))
    else:
        engine_ui = req.params.get("engine", "Paddle原生")
        engine_map = {
            "Paddle原生": None,
            "ONNX Runtime": "onnxruntime",
            "Transformers": "transformers"
        }
        engine_backend = engine_map.get(engine_ui, None)
        backend = LocalSDKBackend(device=req.device, engine=engine_backend)
        
    return backend.run(req)