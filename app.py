# app.py
"""PaddleOCR Studio - Gradio 演示程序入口（全功能版）"""
from __future__ import annotations

import json
import os
import shutil
import time
import datetime
import gradio as gr

from backends.base import InferenceRequest
from services.runner import run_inference

CONFIG_PATH = "paddleocr_studio_config.json"
OUTPUT_DIR = "outputs"
SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".pdf"]

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

OCR_MODELS = ["PP-OCRv6", "PP-OCRv5"]
DOC_MODELS = ["PP-StructureV3", "PaddleOCR-VL", "PaddleOCR-VL-1.5", "PaddleOCR-VL-1.6"]
LANGS = ["ch", "en", "japan", "korean", "chinese_cht", "fr", "german", "ru", "ar"]
ENGINES = ["Paddle原生", "ONNX Runtime", "Transformers"]

DEFAULT_CONFIG = {
    "mode": "SDK", "device": "CPU", "engine": "Paddle原生",
    "token": "", "lang": "ch",
    "ocr_model": "PP-OCRv6", "doc_model": "PP-StructureV3",
    "text_rec_score_thresh": 0.5,
    "use_textline_orientation": True, "use_doc_orientation_classify": False, "use_doc_unwarping": False,
    "use_table_recognition": True, "use_formula_recognition": True, "prettify_markdown": True,
    "enable_hpi": False, "use_tensorrt": False, "precision": "fp32", "cpu_threads": 10,
    "batch_delay": 1.5
}

# ---------------- 全局状态 ---------------- #
HISTORY_RECORDS = []    
HISTORY_STATES = {}     
CURRENT_ID = 0

def load_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as exc:
            print(f"[config] 读取失败: {exc}")
    return cfg

def save_config(updates: dict) -> None:
    cfg = load_config()
    cfg.update(updates)
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[config] 写入失败: {exc}")

def _resolve_path(file) -> str | None:
    if isinstance(file, str):
        return file
    return getattr(file, "name", None)

def _ensure_list(vis):
    if not vis: return []
    if isinstance(vis, str): return [vis]
    if isinstance(vis, list): return vis
    return []

def _get_unique_dir(base_name: str) -> str:
    path = os.path.join(OUTPUT_DIR, base_name)
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        new_path = os.path.join(OUTPUT_DIR, f"{base_name}_{idx}")
        if not os.path.exists(new_path):
            return new_path
        idx += 1

def _save_result_to_disk(state: dict, base_name: str) -> str:
    safe_name = (base_name or "result").strip().replace("/", "_").replace("\\", "_")
    save_dir = _get_unique_dir(safe_name) 
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        
    if state.get("task") == "doc_parsing" and state.get("structured", {}).get("markdown"):
        with open(os.path.join(save_dir, "result.md"), "w", encoding="utf-8") as f:
            f.write(state["structured"]["markdown"])
            
    for idx, vis in enumerate(state.get("visualizations", [])):
        if vis and os.path.exists(vis):
            ext = os.path.splitext(vis)[-1]
            shutil.copy(vis, os.path.join(save_dir, f"visualization_{idx}{ext}"))

    zip_path = shutil.make_archive(save_dir, 'zip', save_dir)
    return zip_path

def _append_history(task_type, file_name, model, mode, device, engine, state):
    global CURRENT_ID
    CURRENT_ID += 1
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    h_id = str(CURRENT_ID)
    HISTORY_STATES[h_id] = state
    
    if mode == "API":
        dev_str, eng_str = "云端", "-"
    else:
        dev_str, eng_str = device, engine
        
    HISTORY_RECORDS.append([h_id, ts, task_type, os.path.basename(file_name), model, mode, dev_str, eng_str])
    return HISTORY_RECORDS

CSS = """
.gradio-container {max-width: 1320px !important; margin: auto;}
#shared-panel {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 16px; background: var(--background-fill-secondary); margin-bottom: 12px;}
#perf-panel {border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px 16px; background: #f9fafb; margin-bottom: 12px;}
#status-bar {font-weight: 600; margin-top: 4px;}
"""

cfg = load_config()

with gr.Blocks(title="PaddleOCR Studio") as demo:
    gr.Markdown("# PaddleOCR Studio\n云端 API 与本地 SDK 全系列功能演示")

    # ---------------- 共享参数区 ----------------
    with gr.Group(elem_id="shared-panel"):
        with gr.Row():
            mode_radio = gr.Radio(["SDK", "API"], value=cfg["mode"], label="执行模式")
            device_radio = gr.Radio(["CPU", "GPU"], value=cfg["device"], label="推理设备（仅 SDK）", interactive=(cfg["mode"] == "SDK"))
            engine_dropdown = gr.Dropdown(ENGINES, value=cfg.get("engine", "Paddle原生"), label="推理引擎（仅 SDK）", interactive=(cfg["mode"] == "SDK"))
            token_box = gr.Textbox(value=cfg["token"], label="API Token", type="password", visible=(cfg["mode"] == "API"), placeholder="留空则读取环境变量")
            lang_dropdown = gr.Dropdown(LANGS, value=cfg["lang"], label="语言（仅 SDK OCR）")
        status_md = gr.Markdown(f"当前：{cfg['mode']} · {cfg['device']} · {cfg.get('engine', 'Paddle原生')}", elem_id="status-bar")

    # ---------------- 性能参数面板 ----------------
    is_sdk = cfg["mode"] == "SDK"
    is_paddle = cfg["engine"] == "Paddle原生"
    is_gpu = cfg["device"] == "GPU"
    is_cpu = cfg["device"] == "CPU"

    with gr.Group(elem_id="perf-panel", visible=is_sdk) as perf_panel:
        gr.Markdown("### ⚙️ 性能与精度参数")
        with gr.Row():
            with gr.Group(visible=(is_sdk and is_paddle)) as hpi_group:
                enable_hpi = gr.Checkbox(value=cfg["enable_hpi"], label="启用高性能推理 (HPI)")
                use_tensorrt = gr.Checkbox(value=cfg["use_tensorrt"], label="启用 TensorRT")
            with gr.Group(visible=(is_sdk and is_paddle and is_gpu)) as prec_group:
                precision = gr.Dropdown(["fp32", "fp16"], value=cfg["precision"], label="计算精度")
            with gr.Group(visible=(is_sdk and is_cpu)) as threads_group:
                cpu_threads = gr.Slider(1, 32, value=cfg["cpu_threads"], step=1, label="CPU 线程数")

    # ---------------- 功能分页 ----------------
    with gr.Tabs():
        # ===== 文字识别 =====
        with gr.Tab("文字识别"):
            with gr.Row():
                with gr.Column(scale=1):
                    ocr_file = gr.File(label="上传图片 / PDF", type="filepath", file_types=SUPPORTED_EXTS)
                    ocr_model = gr.Dropdown(OCR_MODELS, value=cfg["ocr_model"], label="模型版本")
                    with gr.Accordion("OCR 参数", open=False):
                        thresh = gr.Slider(0, 1, value=cfg["text_rec_score_thresh"], step=0.01, label="置信度阈值")
                        orientation = gr.Checkbox(value=cfg["use_textline_orientation"], label="文本行方向分类")
                        doc_orient = gr.Checkbox(value=cfg["use_doc_orientation_classify"], label="文档方向分类")
                        unwarp = gr.Checkbox(value=cfg["use_doc_unwarping"], label="文档矫正")
                    ocr_btn = gr.Button("开始识别", variant="primary")
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("可视化"):
                            ocr_vis = gr.Gallery(label="标注结果", object_fit="contain", height=400)
                        with gr.Tab("行级结果"):
                            ocr_df = gr.Dataframe(headers=["文本", "置信度", "坐标"], type="array", interactive=False)
                        with gr.Tab("原始 JSON"):
                            ocr_json = gr.JSON()

        # ===== 文档解析 =====
        with gr.Tab("文档解析"):
            with gr.Row():
                with gr.Column(scale=1):
                    doc_file = gr.File(label="上传 PDF / 图片", type="filepath", file_types=SUPPORTED_EXTS)
                    doc_model = gr.Dropdown(DOC_MODELS, value=cfg["doc_model"], label="解析模型")
                    with gr.Accordion("解析参数", open=False):
                        table_rec = gr.Checkbox(value=cfg["use_table_recognition"], label="表格识别")
                        formula_rec = gr.Checkbox(value=cfg["use_formula_recognition"], label="公式识别")
                        prettify = gr.Checkbox(value=cfg["prettify_markdown"], label="Markdown 美化")
                    doc_btn = gr.Button("开始解析", variant="primary")
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Markdown 预览"):
                            doc_md = gr.Markdown()
                        with gr.Tab("版面可视化"):
                            layout_vis = gr.Gallery(label="版面分析渲染", object_fit="contain", height=400)
                        with gr.Tab("原始 JSON"):
                            doc_json = gr.JSON()

        # ===== 批量处理 =====
        with gr.Tab("本地批量处理"):
            gr.Markdown("### 📦 批量任务队列\n支持选择多个文件或文件夹，自动保存结果到 `outputs/batch_xxx`。")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.Files(label="选择文件/文件夹", file_count="multiple")
                    batch_task_type = gr.Radio(["OCR", "文档解析"], value="OCR", label="任务类型")
                    batch_delay = gr.Slider(
                        0.0, 5.0, value=cfg.get("batch_delay", 1.5), step=0.5, 
                        label="API 请求间隔 (秒)", 
                        info="调用云端 API 时建议至少 1.5 秒，防止限流"
                    )
                    batch_btn = gr.Button("加入队列并开始", variant="primary")
                with gr.Column(scale=2):
                    batch_status = gr.Markdown("等待任务提交...")
                    batch_df = gr.Dataframe(
                        headers=["文件名", "状态", "耗时(秒)", "错误/结果路径"], 
                        interactive=False, 
                        wrap=True
                    )

        # ===== 历史结果与导出 =====
        with gr.Tab("历史结果与导出"):
            gr.Markdown("### 📜 历史记录\n**点击表格中某一行选中**，然后在下方操作（支持按住 Ctrl/Shift 多选或手动输入 ID）。")
            
            history_df = gr.Dataframe(
                headers=["ID", "时间", "任务", "文件名", "模型", "模式", "设备", "引擎"],
                value=HISTORY_RECORDS,
                interactive=False,
                wrap=True,
                elem_id="history-table"
            )
            selected_hist_id = gr.State(None)
            
            with gr.Row():
                load_hist_btn = gr.Button("加载选中记录", variant="secondary")
                delete_btn = gr.Button("删除选中记录", variant="stop")
                clear_all_btn = gr.Button("清空全部记录", variant="stop")
            
            gr.Markdown("---\n### 📥 导出结果")
            with gr.Row():
                export_single_btn = gr.Button("打包导出当前选中结果", variant="primary")
                export_ids = gr.Textbox(label="批量导出指定 ID (英文逗号分隔，如: 1,3,5)", placeholder="1,2,3")
                export_batch_btn = gr.Button("打包导出指定 ID", variant="primary")
                export_all_btn = gr.Button("打包导出全部记录", variant="primary")
            
            export_file = gr.File(label="下载结果 (多次下载请刷新页面)")
            last_state = gr.State(None)

    # ----------------------------------------------------------------------- #
    # 事件绑定
    # ----------------------------------------------------------------------- #
    def _update_perf_vis(mode, device, engine):
        s = (mode == "SDK")
        p = (engine == "Paddle原生")
        g = (device == "GPU")
        c = (device == "CPU")
        return gr.update(visible=(s and p)), gr.update(visible=(s and p and g)), gr.update(visible=(s and c))

    def on_mode_change(mode, device, engine):
        save_config({"mode": mode})
        is_api = mode == "API"
        h, p, t = _update_perf_vis(mode, device, engine)
        return gr.update(interactive=not is_api), gr.update(interactive=not is_api), gr.update(visible=is_api), gr.update(visible=not is_api), h, p, t, f"当前：{mode} · {device} · {engine}"

    def on_device_engine_change(mode, device, engine):
        save_config({"device": device, "engine": engine})
        h, p, t = _update_perf_vis(mode, device, engine)
        return h, p, t, f"当前：{mode} · {device} · {engine}"

    mode_radio.change(on_mode_change, [mode_radio, device_radio, engine_dropdown], [device_radio, engine_dropdown, token_box, perf_panel, hpi_group, prec_group, threads_group, status_md])
    device_radio.change(on_device_engine_change, [mode_radio, device_radio, engine_dropdown], [hpi_group, prec_group, threads_group, status_md])
    engine_dropdown.change(on_device_engine_change, [mode_radio, device_radio, engine_dropdown], [hpi_group, prec_group, threads_group, status_md])

    for comp, key in [
        (token_box, "token"), (lang_dropdown, "lang"), (engine_dropdown, "engine"),
        (ocr_model, "ocr_model"), (doc_model, "doc_model"), (thresh, "text_rec_score_thresh"),
        (orientation, "use_textline_orientation"), (doc_orient, "use_doc_orientation_classify"),
        (unwarp, "use_doc_unwarping"), (table_rec, "use_table_recognition"),
        (formula_rec, "use_formula_recognition"), (prettify, "prettify_markdown"),
        (enable_hpi, "enable_hpi"), (use_tensorrt, "use_tensorrt"),
        (precision, "precision"), (cpu_threads, "cpu_threads"),
        (batch_delay, "batch_delay"),
    ]:
        comp.change(lambda v, k=key: save_config({k: v}), comp, None)

    # ---------------- 单任务提交逻辑 ----------------
    def on_ocr_submit(file, mode, device, engine, model, token, lang, thresh_v, orient_v, doc_orient_v, unwarp_v, hpi_v, trt_v, prec_v, threads_v):
        path = _resolve_path(file)
        if not path: return [], [], {"error": "未上传文件"}, None, HISTORY_RECORDS
        gr.Info("OCR 任务已提交，正在处理中...")
        
        params = {"token": token, "lang": lang, "engine": engine, "text_rec_score_thresh": thresh_v,
                  "use_textline_orientation": orient_v, "use_doc_orientation_classify": doc_orient_v,
                  "use_doc_unwarping": unwarp_v, "enable_hpi": hpi_v, "use_tensorrt": trt_v,
                  "precision": prec_v, "cpu_threads": threads_v}
        req = InferenceRequest(task="ocr", mode="api" if mode == "API" else "sdk", device="gpu:0" if device == "GPU" else "cpu", model_version=model, params=params, file_path=path)
        
        try:
            res = run_inference(req)
            if res.error: return [], [], {"error": res.error}, None, HISTORY_RECORDS
            
            if isinstance(res.raw, dict):
                for w in res.raw.get("warnings", []): gr.Warning(w)

            lines = res.structured.get("lines", [])
            rows = [[l.get("text"), round(float(l.get("score", 0.0)), 4), str(l.get("box"))] for l in lines]
            vis_list = _ensure_list(res.visualization)
            
            state = {"task": "ocr", "mode": res.mode, "model": model, "structured": res.structured, "raw": res.raw, "visualizations": vis_list}
            h_records = _append_history("OCR", path, model, mode, device, engine, state)
            return vis_list, rows, res.raw, state, h_records
        except Exception as e:
            return [], [], {"error": str(e)}, None, HISTORY_RECORDS

    ocr_btn.click(on_ocr_submit, [ocr_file, mode_radio, device_radio, engine_dropdown, ocr_model, token_box, lang_dropdown, thresh, orientation, doc_orient, unwarp, enable_hpi, use_tensorrt, precision, cpu_threads],
                  [ocr_vis, ocr_df, ocr_json, last_state, history_df])

    def on_doc_submit(file, mode, device, engine, model, token, table_v, formula_v, prettify_v, hpi_v, trt_v, prec_v, threads_v):
        path = _resolve_path(file)
        if not path: return "", [], {"error": "未上传文件"}, None, HISTORY_RECORDS
        gr.Info("文档解析任务已提交，正在处理中...")
        
        params = {"token": token, "engine": engine, "use_table_recognition": table_v,
                  "use_formula_recognition": formula_v, "prettify_markdown": prettify_v,
                  "enable_hpi": hpi_v, "use_tensorrt": trt_v, "precision": prec_v, "cpu_threads": threads_v}
        req = InferenceRequest(task="doc_parsing", mode="api" if mode == "API" else "sdk", device="gpu:0" if device == "GPU" else "cpu", model_version=model, params=params, file_path=path)
        
        try:
            res = run_inference(req)
            if res.error: return "", [], {"error": res.error}, None, HISTORY_RECORDS
            
            if isinstance(res.raw, dict):
                for w in res.raw.get("warnings", []): gr.Warning(w)

            md = res.structured.get("markdown") or ""
            vis_list = _ensure_list(res.visualization)
            
            state = {"task": "doc_parsing", "mode": res.mode, "model": model, "structured": res.structured, "raw": res.raw, "visualizations": vis_list}
            h_records = _append_history("文档解析", path, model, mode, device, engine, state)
            return md, vis_list, {"structured": res.structured, "raw": res.raw}, state, h_records
        except Exception as e:
            return "", [], {"error": str(e)}, None, HISTORY_RECORDS

    doc_btn.click(on_doc_submit, [doc_file, mode_radio, device_radio, engine_dropdown, doc_model, token_box, table_rec, formula_rec, prettify, enable_hpi, use_tensorrt, precision, cpu_threads],
                  [doc_md, layout_vis, doc_json, last_state, history_df])

    # ---------------- 批量处理逻辑 ----------------
    def on_batch_submit(files, task_type, mode, device, engine, ocr_model, doc_model, token, delay_sec):
        if not files: return "请先选择文件或文件夹", None, HISTORY_RECORDS
        gr.Info(f"批量任务已启动，共 {len(files)} 个文件。")
        
        df_data = []
        for f in files:
            path = _resolve_path(f)
            if not path: continue
            df_data.append([os.path.basename(path), "等待中", "-", "-"])
            
        yield "任务执行中...", df_data, HISTORY_RECORDS
        
        for i, f in enumerate(files):
            path = _resolve_path(f)
            if not path: continue
            
            # 需求8：拦截不支持的文件格式
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_EXTS:
                df_data[i][1] = "失败"
                df_data[i][3] = f"不支持的文件格式: {ext}"
                yield "任务执行中...", df_data, HISTORY_RECORDS
                continue
                
            df_data[i][1] = "处理中..."
            yield "任务执行中...", df_data, HISTORY_RECORDS
            
            start_t = time.time()
            try:
                m = ocr_model if task_type == "OCR" else doc_model
                params = {"token": token, "engine": engine}
                req = InferenceRequest(
                    task="ocr" if task_type == "OCR" else "doc_parsing", 
                    mode="api" if mode == "API" else "sdk", 
                    device="gpu:0" if device == "GPU" else "cpu", 
                    model_version=m, params=params, file_path=path
                )
                res = run_inference(req)
                cost = round(time.time() - start_t, 2)
                
                if res.error:
                    df_data[i][1] = "失败"
                    df_data[i][3] = res.error
                else:
                    df_data[i][1] = "完成"
                    state = {"task": req.task, "model": m, "structured": res.structured, "raw": res.raw, "visualizations": _ensure_list(res.visualization)}
                    
                    # 需求9：智能命名，不覆盖
                    base_name = os.path.splitext(os.path.basename(path))[0]
                    zip_path = _save_result_to_disk(state, f"batch_{base_name}")
                    df_data[i][3] = zip_path
                    
                    # 需求3：将批量任务加入历史记录
                    _append_history(task_type, path, m, mode, device, engine, state)
            except Exception as e:
                df_data[i][1] = "失败"
                df_data[i][3] = str(e)
                
            df_data[i][2] = cost if 'cost' in locals() else "-"
            yield "任务执行中...", df_data, HISTORY_RECORDS
            
            # 如果是 API 模式，强制休眠设定的间隔
            if mode == "API" and i < len(files) - 1:
                time.sleep(delay_sec)
            
        yield "✅ 批量任务全部完成！", df_data, HISTORY_RECORDS

    batch_btn.click(
        on_batch_submit,
        [batch_files, batch_task_type, mode_radio, device_radio, engine_dropdown, ocr_model, doc_model, token_box, batch_delay],
        [batch_status, batch_df, history_df]
    )

    # ---------------- 历史记录操作 ----------------
    def on_row_select(evt: gr.SelectData, df_data):
        if evt.index is None: return None
        row_idx = evt.index[0]
        if hasattr(df_data, "iloc"):
            h_id = df_data.iloc[row_idx]["ID"]
        else:
            h_id = df_data[row_idx][0]
        return str(h_id)

    def on_load_history_btn(hist_id):
        if not hist_id or hist_id not in HISTORY_STATES:
            return None, "", [], {}, "未选中记录或该记录已丢失"
        
        state = HISTORY_STATES[hist_id]
        md = state.get("structured", {}).get("markdown", "")
        vis = state.get("visualizations", [])
        raw_json = {"structured": state.get("structured"), "raw": state.get("raw")}
        return state, md, vis, raw_json, f"已加载历史记录 ID={hist_id}"

    history_df.select(on_row_select, [history_df], [selected_hist_id])
    load_hist_btn.click(on_load_history_btn, [selected_hist_id], [last_state, doc_md, layout_vis, doc_json, batch_status])

    def on_delete_selected(hist_id):
        global HISTORY_RECORDS
        if not hist_id or hist_id not in HISTORY_STATES:
            return HISTORY_RECORDS, "未选中有效记录"
        
        HISTORY_RECORDS = [row for row in HISTORY_RECORDS if str(row[0]) != str(hist_id)]
        if hist_id in HISTORY_STATES:
            del HISTORY_STATES[hist_id]
            
        gr.Info(f"已删除记录 ID={hist_id}")
        return HISTORY_RECORDS, "记录已删除"

    def on_clear_all():
        global HISTORY_RECORDS
        HISTORY_RECORDS.clear()
        HISTORY_STATES.clear()
        gr.Info("已清空全部记录")
        return HISTORY_RECORDS, "全部清空"

    delete_btn.click(on_delete_selected, [selected_hist_id], [history_df, batch_status])
    clear_all_btn.click(on_clear_all, [], [history_df, batch_status])

    # ---------------- 导出逻辑 ----------------
    def _export_and_zip(states_dict: dict, zip_name: str) -> str:
        main_dir = _get_unique_dir(zip_name)
        os.makedirs(main_dir, exist_ok=True)
        
        for hid, state in states_dict.items():
            sub_dir = os.path.join(main_dir, f"task_{hid}")
            os.makedirs(sub_dir, exist_ok=True)
            with open(os.path.join(sub_dir, "result.json"), "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)
            if state.get("task") == "doc_parsing" and state.get("structured", {}).get("markdown"):
                with open(os.path.join(sub_dir, "result.md"), "w", encoding="utf-8") as f:
                    f.write(state["structured"]["markdown"])
            for idx, vis in enumerate(state.get("visualizations", [])):
                if vis and os.path.exists(vis):
                    ext = os.path.splitext(vis)[-1]
                    shutil.copy(vis, os.path.join(sub_dir, f"vis_{idx}{ext}"))
                    
        return shutil.make_archive(main_dir, 'zip', main_dir)

    def on_export_single(state):
        if not state: return None
        return _save_result_to_disk(state, "single_export")

    def on_export_ids(ids_str):
        if not ids_str: return None
        id_list = [i.strip() for i in ids_str.split(",") if i.strip()]
        target_states = {hid: HISTORY_STATES[hid] for hid in id_list if hid in HISTORY_STATES}
        if not target_states: return None
        return _export_and_zip(target_states, "batch_export_ids")

    def on_export_all():
        if not HISTORY_STATES: return None
        return _export_and_zip(HISTORY_STATES, "batch_export_all")

    export_single_btn.click(on_export_single, [last_state], export_file)
    export_batch_btn.click(on_export_ids, [export_ids], export_file)
    export_all_btn.click(on_export_all, [], export_file)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=gr.themes.Soft(), css=CSS)