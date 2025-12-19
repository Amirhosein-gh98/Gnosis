# Copyright 2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import os
import re
import shutil
from pathlib import Path
from setuptools import find_packages, setup

# ──────────────────────────────────────────────────────────────────────────────
# Usage with our recommended steps:
#   1) pip install vllm==0.8.5.post1   # brings torch==2.6.0
#   2) pip install flash-attn --no-build-isolation
#   3) pip install -e /home/amirhosein/codes/SelfAwareMachine/transformers
#      pip install -e '/home/amirhosein/codes/SelfAwareMachine/trl[vllm]'
#   4) pip install -e . --no-deps      # from the open-r1 repo root
#
# By default this setup.py does NOT install transformers/trl.
# You can override with:
#   OPENR1_HF_DEPS=pypi  -> install pinned PyPI wheels
#   OPENR1_HF_DEPS=local -> install from local paths (non-editable)
# ──────────────────────────────────────────────────────────────────────────────

# Local paths (used only if OPENR1_HF_DEPS=local)
TRANSFORMERS_LOCAL_PATH = Path("/home/amirhosein/codes/SelfAwareMachine/transformers").resolve()
TRL_LOCAL_PATH = Path("/home/amirhosein/codes/SelfAwareMachine/trl").resolve()

def _uri(p: Path) -> str:
    return p.as_uri()

# Strategy for HF deps: "none" (default), "pypi", or "local"
HF_DEPS_MODE = os.getenv("OPENR1_HF_DEPS", "none").strip().lower()
if HF_DEPS_MODE not in {"none", "pypi", "local"}:
    print(f"[setup.py] Unknown OPENR1_HF_DEPS={HF_DEPS_MODE!r}; defaulting to 'none'")
    HF_DEPS_MODE = "none"

# Remove stale egg-info to avoid pip#5466 issues with editable installs
stale_egg_info = Path(__file__).parent / "open_r1.egg-info"
if stale_egg_info.exists():
    print(f"[setup.py] Removing stale {stale_egg_info} to keep editable installs healthy.")
    shutil.rmtree(stale_egg_info)

# Pins for optional PyPI install
TRANSFORMERS_PIN = "transformers==4.52.3"
TRL_PIN = "trl[vllm]==0.18.0"

# Base deps (NO transformers / trl here — keep core minimal)
_base_deps = [
    "accelerate==1.4.0",
    "bitsandbytes>=0.43.0",
    "datasets>=3.2.0",
    "deepspeed==0.16.8",
    "distilabel[vllm,ray,openai]>=1.5.2",
    "e2b-code-interpreter>=1.0.5",
    "einops>=0.8.0",
    "flake8>=6.0.0",
    "hf_transfer>=0.1.4",
    "huggingface-hub[cli,hf_xet]>=0.30.2,<1.0",
    "isort>=5.12.0",
    "jieba",
    "langdetect",
    "latex2sympy2_extended>=1.0.6",
    "liger-kernel>=0.5.10",
    "lighteval @ git+https://github.com/huggingface/lighteval.git@d3da6b9bbf38104c8b5e1acc86f83541f9a502d1",
    "math-verify==0.5.2",
    "morphcloud==0.1.67",
    "packaging>=23.0",
    "parameterized>=0.9.0",
    "peft>=0.14.0",
    "pytest",
    "python-dotenv",
    "ruff>=0.9.0",
    "safetensors>=0.3.3",
    "sentencepiece>=0.1.99",
    # torch stays pinned to match vLLM wheels; already present if you followed step (2)
    "torch==2.6.0",
    "wandb>=0.19.1",
    "async-lru>=2.0.5",
    "aiofiles>=24.1.0",
    "pandas>=2.2.3",
]

# Optional HF deps depending on mode
_hf_deps = []
if HF_DEPS_MODE == "pypi":
    print("[setup.py] OPENR1_HF_DEPS=pypi → will install pinned PyPI transformers/trl")
    _hf_deps = [TRANSFORMERS_PIN, TRL_PIN]
elif HF_DEPS_MODE == "local":
    # NOTE: This installs NON-editable wheels into site-packages.
    # For true dev, install your clones with `pip install -e` before installing open-r1.
    print("[setup.py] OPENR1_HF_DEPS=local → will install from local paths (non-editable)")
    if not TRANSFORMERS_LOCAL_PATH.exists():
        raise FileNotFoundError(f"Local transformers not found: {TRANSFORMERS_LOCAL_PATH}")
    if not TRL_LOCAL_PATH.exists():
        raise FileNotFoundError(f"Local trl not found: {TRL_LOCAL_PATH}")
    _hf_deps = [
        f"transformers @ {_uri(TRANSFORMERS_LOCAL_PATH)}",
        f"trl[vllm] @ {_uri(TRL_LOCAL_PATH)}",
    ]
else:
    print("[setup.py] OPENR1_HF_DEPS=none (default) → NOT installing transformers/trl")

# Build a lookup (used to compose extras)
_all_deps = _base_deps + _hf_deps

deps_map = {
    b: a
    for a, b in (
        re.findall(r"^(([^!=<>~ \[\]]+)(?:\[[^\]]+\])?(?:[!=<>~ ].*)?$)", x)[0]
        for x in _all_deps
    )
}

def deps_list(*pkgs):
    return [deps_map[p] for p in pkgs if p in deps_map]

extras = {}
extras["tests"]   = deps_list("pytest", "parameterized", "math-verify", "jieba")
extras["torch"]   = deps_list("torch")
extras["quality"] = deps_list("ruff", "isort", "flake8")
extras["code"]    = deps_list("e2b-code-interpreter", "python-dotenv", "morphcloud", "jieba", "pandas", "aiofiles")
extras["eval"]    = deps_list("lighteval", "math-verify")
extras["dev"]     = extras["quality"] + extras["tests"] + extras["eval"] + extras["code"]

# Convenience extras to force-install HF deps if needed:
extras["hf-pypi"]  = [TRANSFORMERS_PIN, TRL_PIN]
extras["hf-local"] = [
    (f"transformers @ {_uri(TRANSFORMERS_LOCAL_PATH)}" if TRANSFORMERS_LOCAL_PATH.exists() else TRANSFORMERS_PIN),
    (f"trl[vllm] @ {_uri(TRL_LOCAL_PATH)}" if TRL_LOCAL_PATH.exists() else TRL_PIN),
]

install_requires = _base_deps + _hf_deps

setup(
    name="open-r1",
    version="0.1.0.dev0",
    author="The Hugging Face team (past and future)",
    author_email="lewis@huggingface.co",
    description="Open R1",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="llm inference-time compute reasoning",
    license="Apache",
    url="https://github.com/huggingface/open-r1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    extras_require=extras,
    python_requires=">=3.10.9",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
