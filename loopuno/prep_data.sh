#!/bin/bash
# Tokenize 60k OpenThoughts3 conversations with the Nanbeige4.2 tokenizer.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -m training.prepare_openthoughts --output $P/data/openthoughts-nanbeige-60k --num-proc ${NUM_PROC:-64} --max-samples 60000
