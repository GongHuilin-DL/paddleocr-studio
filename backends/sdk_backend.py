# backends/sdk_backend.py
import os
import uuid
import glob

# 尝试添加 cuDNN DLL 路径，防止 Windows 下加载失败
cudnn_path = r"F:\CondaData\envs\paddle_env\Lib\site-packages\nvidia\cudnn\bin"
if os.path.exists(cudnn_path):
    os.add_dll_directory(cudnn_path)

from backends.base import PaddleOCRBackend, InferenceRequest, NormalizedResult


class LocalSDKBackend(PaddleOCRBackend):
    def __init__(self, device: str = "cpu", engine: str | None = None):
        self.device = device
        self.engine = engine

    def _build_perf_kwargs(self, req: InferenceRequest) -> dict:
        """构造性能/精度相关参数"""
        kwargs = {}
        is_paddle = self.engine is None
        if is_paddle:
            if req.params.get("enable_hpi") is not None:
                kwargs["enable_hpi"] = req.params["enable_hpi"]
            if req.params.get("use_tensorrt") is not None:
                kwargs["use_tensorrt"] = req.params["use_tensorrt"]
            if req.params.get("precision"):
                kwargs["precision"] = req.params["precision"]
        if req.params.get("cpu_threads") is not None:
            kwargs["cpu_threads"] = req.params["cpu_threads"]
        return kwargs

    def _create_pipeline_with_fallback(self, pipeline_cls, kwargs, warnings_list):
        """创建 pipeline，若 HPI 缺失则自动降级重试"""
        try:
            return pipeline_cls(**kwargs)
        except RuntimeError as e:
            err_msg = str(e)
            if "ultra-infer" in err_msg or "hpi" in err_msg.lower():
                kwargs.pop("enable_hpi", None)
                kwargs.pop("use_tensorrt", None)
                warnings_list.append("当前环境未安装高性能推理依赖 (ultra-infer)，已自动关闭 HPI 和 TensorRT，回退标准模式。")
                return pipeline_cls(**kwargs)
            raise e

    def run(self, req: InferenceRequest) -> NormalizedResult:
        try:
            os.makedirs("outputs", exist_ok=True)
            warnings_list = []

            # ================= OCR 任务 =================
            if req.task == "ocr":
                from paddleocr import PaddleOCR

                kwargs = dict(
                    use_textline_orientation=req.params.get("use_textline_orientation", True),
                    lang=req.params.get("lang", "ch"),
                    device=self.device,
                    text_rec_score_thresh=req.params.get("text_rec_score_thresh", 0.5),
                    use_doc_orientation_classify=req.params.get("use_doc_orientation_classify", False),
                    use_doc_unwarping=req.params.get("use_doc_unwarping", False),
                )
                if self.engine:
                    kwargs["engine"] = self.engine
                
                kwargs.update(self._build_perf_kwargs(req))
                pipeline = self._create_pipeline_with_fallback(PaddleOCR, kwargs, warnings_list)
                
                output = pipeline.predict(req.file_path)
                lines = []
                vis_paths = []

                for idx, res in enumerate(output):
                    for box, text, score in zip(
                        res["rec_polys"], res["rec_texts"], res["rec_scores"]
                    ):
                        lines.append({"text": text, "score": float(score), "box": box.tolist()})
                    
                    vis_path = os.path.join("outputs", f"ocr_vis_{uuid.uuid4().hex[:6]}_p{idx}.jpg")
                    res.save_to_img(vis_path)
                    if os.path.exists(vis_path):
                        vis_paths.append(vis_path)

                return NormalizedResult(
                    task=req.task, mode="sdk",
                    structured={"lines": lines, "warnings": warnings_list},
                    raw={"lines": lines, "warnings": warnings_list},
                    visualization=vis_paths
                )

            # ================= 文档解析任务 =================
            else:
                from paddleocr import PPStructureV3

                kwargs = dict(
                    device=self.device,
                    use_table_recognition=req.params.get("use_table_recognition", True),
                    use_formula_recognition=req.params.get("use_formula_recognition", True),
                )
                if self.engine:
                    kwargs["engine"] = self.engine
                
                if self.engine == "onnxruntime" and kwargs["use_formula_recognition"]:
                    kwargs["use_formula_recognition"] = False
                    warnings_list.append("ONNX 引擎暂不支持公式识别，已自动关闭。")

                kwargs.update(self._build_perf_kwargs(req))
                pipeline = self._create_pipeline_with_fallback(PPStructureV3, kwargs, warnings_list)
                
                output = pipeline.predict(req.file_path)
                results = output if isinstance(output, (list, tuple)) else [output]

                md_text = ""
                vis_paths = []
                md_dir = os.path.join("outputs", f"md_{uuid.uuid4().hex[:6]}")
                os.makedirs(md_dir, exist_ok=True)

                for idx, res in enumerate(results):
                    # 1. Markdown
                    try:
                        res.save_to_markdown(save_path=md_dir)
                    except Exception as e:
                        warnings_list.append(f"第 {idx+1} 页 Markdown 保存失败: {e}")

                    # 2. 多页可视化图收集
                    page_img_path = None
                    try:
                        img_dict = getattr(res, "img", None)
                        if isinstance(img_dict, dict):
                            for key in ["layout_det_res", "layout_order_res", "overall_ocr_res"]:
                                if key in img_dict and img_dict[key] is not None:
                                    page_img_path = os.path.join("outputs", f"doc_layout_{uuid.uuid4().hex[:6]}_p{idx}.jpg")
                                    img_dict[key].save(page_img_path)
                                    break
                        
                        if not page_img_path:
                            page_img_path = os.path.join("outputs", f"doc_layout_{uuid.uuid4().hex[:6]}_p{idx}.jpg")
                            res.save_to_img(save_path=page_img_path)
                            
                        if os.path.exists(page_img_path):
                            vis_paths.append(page_img_path)
                    except Exception:
                        pass

                # 合并 Markdown 文件
                md_files = sorted(glob.glob(os.path.join(md_dir, "*.md")))
                for mf in md_files:
                    with open(mf, "r", encoding="utf-8") as f:
                        md_text += f.read() + "\n\n"

                return NormalizedResult(
                    task=req.task, mode="sdk",
                    structured={"markdown": md_text, "pages": len(results), "warnings": warnings_list},
                    raw={"pages": len(results), "warnings": warnings_list},
                    visualization=vis_paths
                )
        except Exception as e:
            import traceback
            return NormalizedResult(
                task=req.task, mode="sdk",
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )