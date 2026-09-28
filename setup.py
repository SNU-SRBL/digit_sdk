from setuptools import setup, find_packages

setup(
    name="digit_sdk",
    version="0.1.0",
    description="SDK for DIGIT tactile depth, point-cloud, and force estimation",
    author="Byung-Hyun Song",
    author_email="bh.song@snu.ac.kr",
    packages=find_packages(),
    package_data={
        "calibration.annotate_contact": ["static/*"],
        "calibration.annotate_ball": ["static/*"],
    },
    install_requires=[
        "pillow==10.0.0",
        "numpy==1.26.4",
        "opencv-contrib-python==4.14.0.94",
        "scipy>=1.13.1",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "PyYaml>=6.0.1",
        "matplotlib>=3.9.0",
        "ffmpeg-python",
        "nanogui",
        "open3d>=0.17.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "pyudev>=0.24.0",
        # Force estimation dependencies (Sparsh)
        "einops>=0.6",
        "timm>=0.9",
        "huggingface_hub>=0.19",
        "omegaconf>=2.3",
        "lightning>=2.0",
        "rich>=13.0",
    ],
    extras_require={
        # Optional CUDA acceleration for Sparsh attention kernels
        "gpu": [
            "xformers>=0.0.22",
        ],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "digit-annotate-contact=calibration.annotate_contact.server:main",
            "digit-annotate-ball=calibration.annotate_ball.server:main",
            "digit-collect-background=calibration.collect_background:main",
            "digit-collect-ball=calibration.collect_ball:main",
            "digit-collect-manual=calibration.collect_manual_contacts:main",
            "digit-finalize-calibration=calibration.finalize_dataset:main",
            "digit-evaluate-decoder=calibration.evaluate_decoder:main",
            "digit-promote-decoder=calibration.promote_decoder:main",
            "digit-promote-calibration=calibration.promote_dataset:main",
            "digit-train-decoder=calibration.train_decoder:main",
            "digit-validate-dataset=calibration.validate_dataset:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
