uv sync
source .venv/bin/activate

# mmcv installation
cd 3rdparty
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv

export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=8

uv pip install -v --no-build-isolation -e .

# 
cd ..
uv pip install "mmengine>=0.7.1,<1.0.0"
uv pip install -e ./det --no-build-isolation
uv pip install -e ./seg --no-build-isolation
