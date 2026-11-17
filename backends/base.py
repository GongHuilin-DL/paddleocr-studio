# backends/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

@dataclass
class InferenceRequest:
    task: str            # "ocr" | "doc_parsing"
    mode: str            # "api" | "sdk"
    device: str          # "cpu" | "gpu:0"
    model_version: str
    params: dict = field(default_factory=dict)
    file_path: str = ""

@dataclass
class NormalizedResult:
    task: str
    mode: str
    structured: dict = field(default_factory=dict)
    raw: Any = None
    # 这里可以是单张图片的路径 str，也可以是多张图片路径的 list[str]
    visualization: Optional[Any] = None 
    resources: List[str] = field(default_factory=list)
    error: Optional[str] = None


class PaddleOCRBackend(ABC):
    """所有后端（API / SDK）的统一抽象接口"""
    
    @abstractmethod
    def run(self, req: InferenceRequest) -> NormalizedResult:
        """执行推理并返回归一化结果"""
        pass