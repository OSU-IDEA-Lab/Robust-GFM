"""Reference check: run ULTRA's own script/run.py (zero-shot, --epochs 0) on ULTRA's own copy of a dataset,
to confirm that ultra_eval.py reproduces upstream on the reference variant. Same two shims as ultra_eval.py
(optional pydeps on the path; PyG version string); the upstream checkout is untouched.

    python ref_check.py -c ref_check_inference.yaml --dataset WN18RR --epochs 0 --bpe null --gpus [0] \
        --ckpt <abs path> --output_root <dir> --dataset_root <dir>
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pydeps"))
os.environ["PATH"] = os.path.join(HERE, "pydeps", "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(HERE, "torch_extensions"))
import torch_geometric  # noqa: E402
torch_geometric.__version__ = ".".join(torch_geometric.__version__.split(".")[:3])
sys.argv = [os.path.join(HERE, "upstream", "script", "run.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
