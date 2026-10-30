#!/usr/bin/env python3
"""
app.py
Modular Python backend server connecting `models/trainer.py` and `data/generator.py`
with `index.html` and `static/js/`.
"""
import json
import mimetypes
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from data.generator import generate_synthetic_data, load_and_preprocess_csv
from models.trainer import MLModelPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATE = {
    "df": None,
    "feature_names": [],
    "target_name": "Target",
    "class_names": ["Class 0", "Class 1"],
    "pipeline": MLModelPipeline(save_dir=os.path.join(BASE_DIR, "models", "saved")),
    "last_results": None
}


def init_default_dataset():
    df, feat_names, target_name, class_names = generate_synthetic_data("moons", 240, 0.18)
    STATE["df"] = df
    STATE["feature_names"] = feat_names
    STATE["target_name"] = target_name
    STATE["class_names"] = class_names
    STATE["last_results"] = STATE["pipeline"].train_and_evaluate(
        df, feat_names, target_name, class_names, model_type="poly_logistic"
    )


class MLDashboardRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filepath: str, content_type: str = None):
        if not os.path.exists(filepath):
            self.send_error(404, f"File not found: {filepath}")
            return
        if not content_type:
            content_type, _ = mimetypes.guess_type(filepath)
            content_type = content_type or "application/octet-stream"
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html", "/templates/index.html"):
            if STATE["df"] is None:
                init_default_dataset()
            self._serve_file(os.path.join(BASE_DIR, "index.html"), "text/html; charset=utf-8")

        elif path.startswith("/static/"):
            rel_path = path.lstrip("/")
            self._serve_file(os.path.join(BASE_DIR, rel_path))

        elif path == "/api/export_csv":
            if STATE["df"] is None:
                init_default_dataset()
            csv_bytes = STATE["df"].to_csv(index=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", "attachment; filename=ml_dataset.csv")
            self.send_header("Content-Length", str(len(csv_bytes)))
            self.end_headers()
            self.wfile.write(csv_bytes)

        elif path.startswith("/api/sample_csv/"):
            name = path.split("/api/sample_csv/")[-1]
            mapping = {
                "churn": "customer_churn.csv",
                "iris": "iris_binary.csv",
                "exam": "exam_pass_study.csv"
            }
            filename = mapping.get(name)
            if not filename:
                self._send_json({"status": "error", "message": "Unknown sample dataset"}, 404)
                return

            csv_path = os.path.join(BASE_DIR, "data", "sample_datasets", filename)
            with open(csv_path, "r", encoding="utf-8") as f:
                csv_text = f.read()

            df, feat_names, target_name, class_names = load_and_preprocess_csv(csv_text)
            STATE["df"] = df
            STATE["feature_names"] = feat_names
            STATE["target_name"] = target_name
            STATE["class_names"] = class_names

            results = STATE["pipeline"].train_and_evaluate(
                df=df,
                feature_names=feat_names,
                target_name=target_name,
                class_names=class_names,
                model_type="poly_logistic"
            )
            STATE["last_results"] = results
            self._send_json({"status": "ok", "data": results})

        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        payload = json.loads(raw_body or "{}")

        try:
            if path == "/api/generate":
                preset = payload.get("preset", "moons")
                n_samples = int(payload.get("n_samples", 240))
                noise = float(payload.get("noise", 0.18))
                seed = int(payload.get("seed", 42))

                df, feat_names, target_name, class_names = generate_synthetic_data(
                    preset=preset, n_samples=n_samples, noise=noise, random_state=seed
                )
                STATE["df"] = df
                STATE["feature_names"] = feat_names
                STATE["target_name"] = target_name
                STATE["class_names"] = class_names

                results = STATE["pipeline"].train_and_evaluate(
                    df=df,
                    feature_names=feat_names,
                    target_name=target_name,
                    class_names=class_names,
                    model_type=payload.get("model_type", "poly_logistic"),
                    hyperparams=payload.get("hyperparams", {}),
                    standardize=payload.get("standardize", True)
                )
                STATE["last_results"] = results
                self._send_json({"status": "ok", "data": results})

            elif path == "/api/upload_csv":
                csv_text = payload.get("csv_text", "")
                target_col = payload.get("target_col")
                feature_cols = payload.get("feature_cols")
                df, feat_names, target_name, class_names = load_and_preprocess_csv(
                    csv_text, target_col=target_col, feature_cols=feature_cols
                )
                STATE["df"] = df
                STATE["feature_names"] = feat_names
                STATE["target_name"] = target_name
                STATE["class_names"] = class_names

                results = STATE["pipeline"].train_and_evaluate(
                    df=df,
                    feature_names=feat_names,
                    target_name=target_name,
                    class_names=class_names,
                    model_type=payload.get("model_type", "poly_logistic"),
                    hyperparams=payload.get("hyperparams", {}),
                    standardize=payload.get("standardize", True)
                )
                STATE["last_results"] = results
                self._send_json({"status": "ok", "data": results})

            elif path == "/api/train":
                if STATE["df"] is None:
                    init_default_dataset()

                model_type = payload.get("model_type", "poly_logistic")
                hyperparams = payload.get("hyperparams", {})
                standardize = bool(payload.get("standardize", True))
                axis_x = int(payload.get("axis_x", 0))
                axis_y = int(payload.get("axis_y", 1))

                results = STATE["pipeline"].train_and_evaluate(
                    df=STATE["df"],
                    feature_names=STATE["feature_names"],
                    target_name=STATE["target_name"],
                    class_names=STATE["class_names"],
                    model_type=model_type,
                    hyperparams=hyperparams,
                    standardize=standardize,
                    axis_x=axis_x,
                    axis_y=axis_y
                )
                STATE["last_results"] = results
                self._send_json({"status": "ok", "data": results})

            elif path == "/api/predict":
                features = payload.get("features", [])
                prediction = STATE["pipeline"].predict_single(features)
                self._send_json({"status": "ok", "prediction": prediction})

            else:
                self.send_error(404, "API endpoint not found")
        except Exception as exc:
            self._send_json({"status": "error", "message": str(exc)}, 400)


def run_server(host: str = "127.0.0.1", port: int = 5000):
    init_default_dataset()
    server = HTTPServer((host, port), MLDashboardRequestHandler)
    print(f"ML Classification Dashboard running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
