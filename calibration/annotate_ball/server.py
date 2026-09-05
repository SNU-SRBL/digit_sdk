"""Local HTTP server for staged ball-circle annotation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Mapping
from urllib.parse import quote, unquote, urlparse
import webbrowser

import cv2
import numpy as np

from calibration.ball_geometry import indentation_depth_mm


MAX_JSON_BYTES = 64 * 1024
ASSET_VERSION = "replacement-v1"


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


class BallAnnotationDataset:
    """Staging-manifest ball queue with validated, atomic NPZ saves."""

    def __init__(self, root: Path, background_path: Path = None):
        self.root = Path(root).resolve()
        self.metadata_path = self.root / "metadata.jsonl"
        if not self.metadata_path.is_file():
            raise FileNotFoundError(self.metadata_path)
        self._lock = threading.Lock()
        self._records = [
            json.loads(line)
            for line in self.metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._ball_records = [
            record for record in self._records if record.get("kind") == "ball"
        ]
        if not self._ball_records:
            raise ValueError("metadata has no ball records")
        self._by_id = {record["sample_id"]: record for record in self._ball_records}
        if len(self._by_id) != len(self._ball_records):
            raise ValueError("ball sample_id values must be unique")
        for record in self._ball_records:
            if "ball_diameter_mm" not in record or "ppmm" not in record:
                raise ValueError(
                    f"ball geometry metadata is missing for {record['sample_id']}"
                )
        default_background = self.root.parent / "background" / "reference.png"
        self.background_path = Path(
            background_path or default_background
        ).resolve()
        background = cv2.imread(str(self.background_path), cv2.IMREAD_COLOR)
        if background is None:
            raise FileNotFoundError(
                f"shared background is missing: {self.background_path}"
            )
        expected_shapes = {tuple(record["shape"]) for record in self._ball_records}
        if expected_shapes != {tuple(background.shape)}:
            raise ValueError(
                "shared background shape does not match staged ball images"
            )

    def _record(self, sample_id: str) -> Dict[str, Any]:
        try:
            return self._by_id[sample_id]
        except KeyError as error:
            raise KeyError(f"unknown sample_id: {sample_id}") from error

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("metadata path leaves annotation root") from error
        return path

    @staticmethod
    def _url(sample_id: str, kind: str) -> str:
        return f"/api/samples/{quote(sample_id, safe='')}/{kind}"

    def samples(self):
        output = []
        for index, record in enumerate(self._ball_records):
            label_exists = bool(record.get("label_path")) and self._path(
                record["label_path"]
            ).is_file()
            item = {
                "index": index,
                "sample_id": record["sample_id"],
                "session_id": record["session_id"],
                "saved": label_exists,
                "image_url": self._url(record["sample_id"], "image"),
                "ball_diameter_mm": record["ball_diameter_mm"],
                "ppmm": record["ppmm"],
                "shape": record["shape"],
                "center_px": record.get("center_px"),
                "radius_px": record.get("radius_px"),
                "indentation_depth_mm": record.get("indentation_depth_mm"),
                "annotator": record.get("annotator", ""),
                "annotation_revision": record.get("annotation_revision", 0),
                "rejected": bool(record.get("rejected", False)),
                "rejected_by": record.get("rejected_by", ""),
                "rejection_reason": record.get("rejection_reason", ""),
                "replaces_sample_id": record.get("replaces_sample_id"),
                "background_url": self._url(record["sample_id"], "background"),
            }
            output.append(item)
        return output

    def file_path(self, sample_id: str, kind: str) -> Path:
        record = self._record(sample_id)
        if kind == "image":
            relative = record["path"]
        elif kind == "background":
            return self.background_path
        else:
            raise KeyError(f"unknown resource: {kind}")
        path = self._path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def save_annotation(
        self,
        sample_id: str,
        center_px,
        radius_px,
        annotator: str,
    ) -> Dict[str, Any]:
        annotator = str(annotator).strip()
        if not annotator:
            raise ValueError("annotator is required")
        record = self._record(sample_id)
        if record.get("rejected"):
            raise ValueError("cannot annotate a rejected sample")
        if not isinstance(center_px, list) or len(center_px) != 2:
            raise ValueError("center_px must contain x and y")
        center = [float(center_px[0]), float(center_px[1])]
        height, width, _ = record["shape"]
        if not np.isfinite(center).all() or not (
            0 <= center[0] < width and 0 <= center[1] < height
        ):
            raise ValueError("center_px must be inside the image")
        radius = float(radius_px)
        depth = indentation_depth_mm(
            radius, record["ball_diameter_mm"], record["ppmm"]
        )
        relative = Path("labels") / record["session_id"] / f"{sample_id}.npz"
        destination = self._path(relative.as_posix())
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.tmp")

        with self._lock:
            with temporary.open("wb") as file:
                np.savez(
                    file,
                    ball_diameter_mm=float(record["ball_diameter_mm"]),
                    center_px=np.asarray(center, dtype=np.float64),
                    radius_px=radius,
                    ppmm=float(record["ppmm"]),
                    indentation_depth_mm=depth,
                )
            temporary.replace(destination)
            record.update({
                "label_path": relative.as_posix(),
                "center_px": center,
                "radius_px": radius,
                "indentation_depth_mm": depth,
                "annotator": annotator,
                "annotation_revision": int(record.get("annotation_revision", 0)) + 1,
            })
            _write_jsonl(self.metadata_path, self._records)
        return {
            "sample_id": sample_id,
            "saved": True,
            "center_px": center,
            "radius_px": radius,
            "indentation_depth_mm": depth,
            "annotator": annotator,
            "annotation_revision": record["annotation_revision"],
        }

    def set_rejection(
        self, sample_id: str, rejected: bool, annotator: str
    ) -> Dict[str, Any]:
        if not isinstance(rejected, bool):
            raise ValueError("rejected must be true or false")
        annotator = str(annotator).strip()
        if not annotator:
            raise ValueError("annotator is required")
        record = self._record(sample_id)
        with self._lock:
            if rejected:
                record.update({
                    "rejected": True,
                    "rejected_by": annotator,
                    "rejected_at_utc": datetime.now(timezone.utc).isoformat(),
                    "rejection_reason": "visual quality control",
                })
                result = {
                    "sample_id": sample_id,
                    "rejected": True,
                    "rejected_by": annotator,
                    "rejection_reason": record["rejection_reason"],
                }
            else:
                for field in (
                    "rejected", "rejected_by", "rejected_at_utc",
                    "rejection_reason",
                ):
                    record.pop(field, None)
                result = {"sample_id": sample_id, "rejected": False}
            _write_jsonl(self.metadata_path, self._records)
        return result

    def status(self):
        total = len(self._ball_records)
        completed = sum(
            not record.get("rejected")
            and bool(record.get("label_path"))
            and self._path(record["label_path"]).is_file()
            and int(record.get("annotation_revision", 0)) >= 1
            for record in self._ball_records
        )
        rejected = sum(bool(record.get("rejected")) for record in self._ball_records)
        return {
            "total": total,
            "completed": completed,
            "rejected": rejected,
            "remaining": total - completed - rejected,
        }


class BallAnnotationHandler(SimpleHTTPRequestHandler):
    dataset: BallAnnotationDataset
    static_root: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.static_root), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, value: Mapping[str, Any], status=HTTPStatus.OK):
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    @staticmethod
    def _route(path: str):
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "samples"]:
            return unquote(parts[2]), parts[3]
        return None

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/samples":
                self._json({"samples": self.dataset.samples(),
                            "status": self.dataset.status()})
                return
            if parsed.path == "/api/status":
                self._json(self.dataset.status())
                return
            route = self._route(parsed.path)
            if route and route[1] in {"image", "background"}:
                payload = self.dataset.file_path(*route).read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
        except (KeyError, FileNotFoundError, ValueError) as error:
            self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
            return
        super().do_GET()

    def do_PUT(self):
        route = self._route(urlparse(self.path).path)
        if route is None or route[1] not in {"annotation", "rejection"}:
            self._json({"error": "unknown resource"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_JSON_BYTES:
                raise ValueError("invalid annotation payload size")
            payload = json.loads(self.rfile.read(length))
            if route[1] == "annotation":
                result = self.dataset.save_annotation(
                    route[0], payload.get("center_px"), payload.get("radius_px"),
                    payload.get("annotator", ""),
                )
            else:
                result = self.dataset.set_rejection(
                    route[0], payload.get("rejected"),
                    payload.get("annotator", ""),
                )
        except KeyError as error:
            self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        else:
            self._json(result)


def make_handler(dataset: BallAnnotationDataset):
    return type(
        "BoundBallAnnotationHandler",
        (BallAnnotationHandler,),
        {"dataset": dataset, "static_root": Path(__file__).with_name("static")},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--sensors-root", type=Path, default=Path("sensors"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    staging_root = (
        args.sensors_root / args.serial / "calibration/inbox/ball"
    )
    dataset = BallAnnotationDataset(staging_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(dataset))
    url = f"http://{args.host}:{server.server_port}/?v={ASSET_VERSION}"
    status = dataset.status()
    print(f"ball annotation server: {url}")
    print(f"queue: {status['completed']}/{status['total']} complete")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
