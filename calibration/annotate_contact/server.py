"""Local HTTP server for binary contact-mask annotation."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Mapping
from urllib.parse import parse_qs, quote, unquote, urlparse
import webbrowser

import numpy as np
from PIL import Image


MAX_MASK_BYTES = 4 * 1024 * 1024


def _write_manifest(path: Path, records: List[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


class AnnotationDataset:
    """Canonical or staged manual-mask queue with atomic saves."""

    def __init__(self, root: Path, background_reference: Path | None = None):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.jsonl"
        if not self.manifest_path.is_file():
            self.manifest_path = self.root / "metadata.jsonl"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self._lock = threading.Lock()
        self._records = [
            json.loads(line)
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._manual_records = [
            record for record in self._records
            if record.get("modality") == "manual_mask"
            or record.get("kind") == "touch"
        ]
        if not self._manual_records:
            raise ValueError("annotation queue has no manual contacts")
        self._by_id = {record["sample_id"]: record
                       for record in self._manual_records}
        if len(self._by_id) != len(self._manual_records):
            raise ValueError("manual_mask sample_id values must be unique")
        if background_reference is None:
            candidates = (
                self.root.parent / "background/reference.png",
                self.root / "background/reference.png",
            )
            background_reference = next(
                (path for path in candidates if path.is_file()), None
            )
        self.background_reference = (
            Path(background_reference).resolve()
            if background_reference is not None else None
        )

    def _image_relative(self, record: Mapping[str, Any]) -> str:
        return record.get("image_path") or record["path"]

    def _mask_relative(self, record: Mapping[str, Any]) -> str:
        return record.get("label_path") or (
            Path("masks") / record["session_id"]
            / f"{record['sample_id']}.png"
        ).as_posix()

    def _record(self, sample_id: str) -> Dict[str, Any]:
        try:
            return self._by_id[sample_id]
        except KeyError as exc:
            raise KeyError(f"unknown sample_id: {sample_id}") from exc

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("manifest path leaves dataset root") from exc
        return path

    def samples(self) -> List[Dict[str, Any]]:
        output = []
        for index, record in enumerate(self._manual_records):
            sample_id = record["sample_id"]
            mask_exists = self._path(self._mask_relative(record)).is_file()
            item = {
                "index": index,
                "sample_id": sample_id,
                "session_id": record["session_id"],
                "saved": mask_exists,
                "annotator": record.get("annotator", ""),
                "annotation_revision": record.get("annotation_revision", 0),
                "image_url": self._url(sample_id, "image"),
                "mask_url": self._url(sample_id, "mask") if mask_exists else None,
                "background_url": None,
            }
            if self.background_reference is not None:
                item["background_url"] = self._url(sample_id, "background")
            output.append(item)
        return output

    @staticmethod
    def _url(sample_id: str, kind: str) -> str:
        return f"/api/samples/{quote(sample_id, safe='')}/{kind}"

    def file_path(self, sample_id: str, kind: str) -> Path:
        record = self._record(sample_id)
        if kind == "background":
            if self.background_reference is None:
                raise KeyError(f"background is not available for {sample_id}")
            if not self.background_reference.is_file():
                raise FileNotFoundError(self.background_reference)
            return self.background_reference
        relative = {
            "image": self._image_relative(record),
            "mask": self._mask_relative(record),
        }.get(kind)
        if not relative:
            raise KeyError(f"{kind} is not available for {sample_id}")
        path = self._path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def save_mask(self, sample_id: str, png_bytes: bytes, annotator: str) -> Dict[str, Any]:
        annotator = annotator.strip()
        if not annotator:
            raise ValueError("annotator is required")
        record = self._record(sample_id)
        expected_height, expected_width, _ = record["shape"]
        try:
            with Image.open(BytesIO(png_bytes)) as incoming:
                if incoming.size != (expected_width, expected_height):
                    raise ValueError(
                        f"mask size {incoming.size} does not match "
                        f"{(expected_width, expected_height)}"
                    )
                grayscale = np.asarray(incoming.convert("L"), dtype=np.uint8)
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("request body is not a valid PNG") from exc
        binary = np.where(grayscale >= 128, 255, 0).astype(np.uint8)
        label_relative = self._mask_relative(record)
        destination = self._path(label_relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")

        with self._lock:
            Image.fromarray(binary, mode="L").save(temporary, format="PNG")
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            temporary.replace(destination)
            record["annotator"] = annotator
            record["annotation_revision"] = int(
                record.get("annotation_revision", 0)
            ) + 1
            record["label_sha256"] = digest
            record["label_path"] = label_relative
            _write_manifest(self.manifest_path, self._records)
        return {
            "sample_id": sample_id,
            "saved": True,
            "annotator": annotator,
            "annotation_revision": record["annotation_revision"],
            "label_sha256": digest,
        }

    def status(self) -> Dict[str, int]:
        total = len(self._manual_records)
        completed = sum(
            self._path(self._mask_relative(record)).is_file()
            and bool(record.get("annotator"))
            and int(record.get("annotation_revision", 0)) >= 1
            for record in self._manual_records
        )
        return {"total": total, "completed": completed, "remaining": total - completed}


class AnnotationHandler(SimpleHTTPRequestHandler):
    dataset: AnnotationDataset
    static_root: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.static_root), **kwargs)

    def _json(self, value: Mapping[str, Any], status=HTTPStatus.OK) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status=status)

    @staticmethod
    def _sample_route(path: str):
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "samples"]:
            return unquote(parts[2]), parts[3]
        return None

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/samples":
                self._json({
                    "samples": self.dataset.samples(),
                    "status": self.dataset.status(),
                })
                return
            if parsed.path == "/api/status":
                self._json(self.dataset.status())
                return
            route = self._sample_route(parsed.path)
            if route is not None:
                sample_id, kind = route
                if kind not in {"image", "mask", "background"}:
                    self._error(HTTPStatus.NOT_FOUND, "unknown resource")
                    return
                path = self.dataset.file_path(sample_id, kind)
                payload = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
                return
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
            return
        except FileNotFoundError:
            self._error(HTTPStatus.NOT_FOUND, "file not found")
            return
        super().do_GET()

    def do_PUT(self):
        parsed = urlparse(self.path)
        route = self._sample_route(parsed.path)
        if route is None or route[1] != "mask":
            self._error(HTTPStatus.NOT_FOUND, "unknown resource")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_MASK_BYTES:
            self._error(HTTPStatus.BAD_REQUEST, "invalid mask payload size")
            return
        annotator = parse_qs(parsed.query).get("annotator", [""])[0]
        try:
            result = self.dataset.save_mask(
                route[0], self.rfile.read(length), annotator
            )
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        else:
            self._json(result)


def make_handler(dataset: AnnotationDataset):
    static_root = Path(__file__).with_name("static")
    return type(
        "BoundAnnotationHandler",
        (AnnotationHandler,),
        {"dataset": dataset, "static_root": static_root},
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Annotate binary contact masks")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--sensors-root", type=Path, default=Path("sensors"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    dataset_root = (
        args.sensors_root / args.serial / "calibration/inbox/manual_mask"
    )
    background_reference = (
        args.sensors_root / args.serial
        / "calibration/inbox/background/reference.png"
    )
    dataset = AnnotationDataset(dataset_root, background_reference)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(dataset))
    url = f"http://{args.host}:{server.server_port}/"
    print(f"annotation server: {url}")
    status = dataset.status()
    print(
        f"queue: {status['completed']}/{status['total']} complete, "
        f"{status['remaining']} remaining"
    )
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
