# backends/api_backend.py
import os
import uuid
import json
import base64
import urllib.request
import shutil
import time

try:
    from PIL import Image, ImageDraw
except ImportError:
    pass

from backends.base import PaddleOCRBackend, InferenceRequest, NormalizedResult

class OfficialAPIBackend(PaddleOCRBackend):
    def __init__(self, token: str | None = None):
        self.token = (token or "").strip() or os.environ.get("PADDLEOCR_ACCESS_TOKEN", "").strip()

    def _force_dict(self, obj):
        """深度递归转换为原生类型"""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple, set)):
            return [self._force_dict(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._force_dict(v) for k, v in obj.items()}
        
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            try: return self._force_dict(obj.model_dump())
            except: pass
        if hasattr(obj, "dict") and callable(obj.dict):
            try: return self._force_dict(obj.dict())
            except: pass
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            try: return self._force_dict(obj.to_dict())
            except: pass
        if hasattr(obj, "__dict__"):
            return self._force_dict(obj.__dict__)
            
        try:
            return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
        except:
            return str(obj)

    def _extract_lines(self, data):
        lines = []
        if isinstance(data, dict):
            if "rec_texts" in data:
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])
                boxes = data.get("rec_polys", data.get("dt_polys", []))
                for t, s, b in zip(texts, scores, boxes):
                    lines.append({"text": t, "score": float(s), "box": b})
            else:
                for v in data.values():
                    lines.extend(self._extract_lines(v))
        elif isinstance(data, list):
            for item in data:
                lines.extend(self._extract_lines(item))
        return lines

    def _download_img(self, url, out_path):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as response, open(out_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            return out_path
        except:
            return None

    def _call_with_retry(self, func, *args, **kwargs):
        """针对云端限流的退避重试"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_msg = str(e)
                if ("队列已满" in err_msg or "rate limit" in err_msg.lower() or "too many requests" in err_msg.lower()) and attempt < max_retries - 1:
                    wait_time = 3 * (attempt + 1)  # 等待 3s, 6s, 9s
                    print(f"[API Backend] 触发限流 ({err_msg})，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                raise e

    def run(self, req: InferenceRequest) -> NormalizedResult:
        if not self.token:
            return NormalizedResult(task=req.task, mode="api", error="API Token 为空。")

        try:
            from paddleocr import PaddleOCRClient
            client = PaddleOCRClient(token=self.token)

            # ================= OCR 任务 =================
            if req.task == "ocr":
                raw = self._call_with_retry(client.ocr, file_path=req.file_path, model=req.model_version)
                res_dict = self._force_dict(raw)
                lines = self._extract_lines(res_dict)
                vis_paths = []

                if isinstance(res_dict, dict):
                    for page in res_dict.get("pages", []):
                        if isinstance(page, dict):
                            url = page.get("ocr_image_url") or page.get("ocrImage")
                            if isinstance(url, str) and url.startswith("http"):
                                os.makedirs("outputs", exist_ok=True)
                                p = os.path.join("outputs", f"api_ocr_{uuid.uuid4().hex[:6]}.jpg")
                                if self._download_img(url, p):
                                    vis_paths.append(p)

                if not vis_paths and lines and hasattr(Image, "open"):
                    try:
                        os.makedirs("outputs", exist_ok=True)
                        p = os.path.join("outputs", f"api_ocr_draw_{uuid.uuid4().hex[:6]}.jpg")
                        img = Image.open(req.file_path).convert("RGB")
                        draw = ImageDraw.Draw(img)
                        for l in lines:
                            b = l.get("box")
                            if b and isinstance(b[0], list):
                                draw.polygon([tuple(pt) for pt in b], outline="red", width=3)
                        img.save(p)
                        vis_paths.append(p)
                    except Exception:
                        pass

                return NormalizedResult(
                    task=req.task, mode="api",
                    structured={"lines": lines}, raw=res_dict, visualization=vis_paths
                )

            # ================= 文档解析任务 =================
            else:
                from paddleocr import PPStructureV3Options, PaddleOCRVLOptions

                if "PP-StructureV3" in req.model_version or "PP_STRUCTURE_V3" in req.model_version:
                    options = PPStructureV3Options(visualize=True)
                else:
                    options = PaddleOCRVLOptions(visualize=True)

                raw = self._call_with_retry(client.parse_document, file_path=req.file_path, model=req.model_version, options=options)
                res_dict = self._force_dict(raw)

                md_text = ""
                vis_paths = []

                def extract_md_and_vis(d):
                    nonlocal md_text
                    if isinstance(d, dict):
                        if "markdown" in d:
                            if isinstance(d["markdown"], dict) and "text" in d["markdown"]:
                                md_text += d["markdown"]["text"] + "\n\n"
                            elif isinstance(d["markdown"], str):
                                md_text += d["markdown"] + "\n\n"
                        
                        imgs = d.get("output_images") or d.get("outputImages")
                        if isinstance(imgs, dict):
                            for k, img_content in imgs.items():
                                if k in ("layout_det_res", "layout_order_res") and isinstance(img_content, str):
                                    if img_content.startswith("http"):
                                        os.makedirs("outputs", exist_ok=True)
                                        p = os.path.join("outputs", f"api_doc_{uuid.uuid4().hex[:6]}.jpg")
                                        if self._download_img(img_content, p):
                                            vis_paths.append(p)
                                    elif len(img_content) > 100:
                                        try:
                                            os.makedirs("outputs", exist_ok=True)
                                            p = os.path.join("outputs", f"api_doc_{uuid.uuid4().hex[:6]}.jpg")
                                            with open(p, "wb") as f:
                                                f.write(base64.b64decode(img_content.split(",")[-1]))
                                            vis_paths.append(p)
                                        except: pass

                        for v in d.values():
                            extract_md_and_vis(v)
                    elif isinstance(d, list):
                        for item in d:
                            extract_md_and_vis(item)

                extract_md_and_vis(res_dict)

                return NormalizedResult(
                    task=req.task, mode="api",
                    structured={"markdown": md_text}, raw=res_dict, visualization=vis_paths
                )
        except Exception as e:
            import traceback
            return NormalizedResult(task=req.task, mode="api", error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")