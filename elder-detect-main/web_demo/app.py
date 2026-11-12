"""老年人纵向步态画像与未来跌倒预警 Flask 演示。"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
from typing import Any

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from .demo_data import build_sample_profile, profile_with_real_observation
from .video_analysis import MediaPipeVideoAnalyzer, ModelUnavailableError


ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
MISSING_MODEL_WARNING = "当前未安装姿态模型，页面展示示例画像，不代表上传视频分析结果"


def _person_from_form() -> dict[str, Any]:
    person: dict[str, Any] = {
        "name": request.form.get("person_name", "").strip(),
        "id": request.form.get("person_id", "").strip(),
    }
    age = request.form.get("age", "").strip()
    if age:
        try:
            numeric_age = int(age)
        except ValueError as exc:
            raise ValueError("年龄必须是整数") from exc
        if not 1 <= numeric_age <= 120:
            raise ValueError("年龄必须在1至120岁之间")
        person["age"] = numeric_age
    return person


def _observed_at_from_form() -> str:
    """校验并规范观测自然日；页面空值使用服务器本地日期。"""
    raw_value = request.form.get("observed_at", "").strip()
    if not raw_value:
        return date.today().isoformat()
    try:
        return date.fromisoformat(raw_value).isoformat()
    except ValueError as exc:
        raise ValueError("观测日期必须是有效的 YYYY-MM-DD 日期") from exc


def create_app(
    config: dict[str, Any] | None = None,
    *,
    analyzer: Any | None = None,
) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        MAX_CONTENT_LENGTH=200 * 1024 * 1024,
        UPLOAD_ROOT=str(Path(tempfile.gettempdir()) / "elder-fall-demo"),
        JSON_AS_ASCII=False,
    )
    if config:
        app.config.update(config)
    video_analyzer = analyzer or MediaPipeVideoAnalyzer()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "纵向步态画像演示"})

    @app.get("/api/profile")
    def profile():
        return jsonify(build_sample_profile())

    @app.post("/api/analyze")
    def analyze():
        if "video" not in request.files:
            return jsonify({"error": "请选择需要分析的视频文件"}), 400
        upload = request.files["video"]
        if not upload.filename:
            return jsonify({"error": "视频文件名不能为空"}), 400
        raw_name = upload.filename.replace("\\", "/").split("/")[-1]
        suffix = Path(raw_name).suffix.lower().lstrip(".")
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            formats = "、".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
            return jsonify({"error": f"不支持的视频格式，仅接受：{formats}"}), 400
        try:
            person = _person_from_form()
            observed_at = _observed_at_from_form()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        upload_root = Path(app.config["UPLOAD_ROOT"])
        upload_root.mkdir(parents=True, exist_ok=True)
        safe_stem = Path(secure_filename(raw_name) or f"upload.{suffix}").stem
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f"{safe_stem[:40]}-", suffix=f".{suffix}", dir=upload_root
        )
        try:
            import os
            os.close(descriptor)
            temp_path = Path(temp_name)
            upload.save(temp_path)
            file_size = temp_path.stat().st_size
            if file_size == 0:
                return jsonify({"error": "上传的视频是空文件，请重新选择"}), 400
            try:
                result = video_analyzer.analyze(temp_path)
                mode = result.get("mode", "real_observation")
                profile_payload = profile_with_real_observation(
                    result, person, observed_at
                )
                warning = None
                message = "视频姿态摘要已完成；长期基线仍需持续采集"
            except ModelUnavailableError:
                mode = "sample_fallback"
                profile_payload = build_sample_profile(person)
                warning = MISSING_MODEL_WARNING
                message = "视频已接收，当前以示例画像演示页面流程"
            except Exception:
                mode = "sample_fallback"
                profile_payload = build_sample_profile(person)
                warning = "姿态分析未完成，页面展示示例画像，不代表上传视频分析结果"
                message = "视频已接收，但本次姿态摘要未生成"
            return jsonify({
                "upload": {"filename": raw_name, "size_bytes": file_size},
                "observation": {"observed_at": observed_at},
                "processing": {
                    "status": "completed",
                    "mode": mode,
                    "warning": warning,
                    "message": message,
                },
                "profile": profile_payload,
            })
        finally:
            Path(temp_name).unlink(missing_ok=True)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "视频超过200MB上限，请压缩后重试"}), 413

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
