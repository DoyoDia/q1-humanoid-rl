#!/bin/bash
# Build mjbatch against the MuJoCo already installed in a venv, and install it there.
#   tools/build_mjbatch.sh [VENV_PYTHON] [MJBATCH_VERSION]
#
# Why: every mjbatch wheel on PyPI pins one MuJoCo release and links its libmujoco.so.<ver>
# (0.1.2 -> 3.13.0, 0.1.4 -> 3.14.0). mjlab needs mujoco ~= 3.16, so the PyPI wheel no
# longer loads ("libmujoco.so.3.13.0: cannot open shared object file"). mjbatch's source
# builds as-is against 3.16's headers: this rewrites the pin to the installed version,
# builds a wheel tagged <ver>+mujoco<NNN>, and installs it. Needs cmake and a C++17 compiler.
# Checked 2026-11-05 (mujoco 3.16.0): tools/check_backend_writes.py passes, compare_backends
# agrees with warp to 0.01%, and physics throughput equals the 3.13 build (1.04 vs 1.06 M
# substeps/s on 32 threads).
set -euo pipefail
PY=${1:-python}
VER=${2:-0.1.4}
MJ=$("$PY" -c "import mujoco; print(mujoco.__version__)")
TAG=${MJ//./}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
curl -sSLo "$WORK/src.tar.gz" "https://files.pythonhosted.org/packages/source/m/mjbatch/mjbatch-$VER.tar.gz"
tar xzf "$WORK/src.tar.gz" -C "$WORK"
SRC=$WORK/mjbatch-$VER
sed -i -E "s/mujoco==[0-9.]+/mujoco==$MJ/g; s/^version = \"$VER\"/version = \"$VER+mujoco$TAG\"/" "$SRC/pyproject.toml"
PYVER=$("$PY" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
env -u PYTHONPATH uv build --wheel --python "$PYVER" -o "$WORK/dist" "$SRC"
env -u PYTHONPATH uv pip install --python "$PY" "$WORK"/dist/mjbatch-*.whl
"$PY" -c "import mjbatch, mujoco; print('mjbatch OK against mujoco', mujoco.__version__)"
