# Source before running LoopUno scripts: keeps caches and outputs under the repository root.
P=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd $P
[ -f $P/.venv/bin/activate ] && . $P/.venv/bin/activate
export HF_HOME=${HF_HOME:-$P/.cache/hf} XDG_CACHE_HOME=$P/.cache TRITON_CACHE_DIR=$P/.cache/triton \
       TORCHINDUCTOR_CACHE_DIR=$P/.cache/inductor UNO_EVAL_DATA_DIR=$P/data/eval PYTHONPATH=$P
export UNO_MODEL_PATH=${UNO_MODEL_PATH:-$P/models/nanbeige42_3b_uno}
