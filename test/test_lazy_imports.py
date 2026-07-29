import subprocess
import sys


def test_shm_reader_import_does_not_load_model_stack():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import digit_sdk.shm_protocol; "
            "assert 'torch' not in sys.modules; assert 'timm' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_public_camera_export_remains_available():
    from digit_sdk import Camera

    assert Camera.__name__ == "Camera"
