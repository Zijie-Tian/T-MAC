#!/bin/zsh

# python3 compile.py -o out -da -nt 1 -tb -gc -ags 64 -t -m hf-bitnet-3b -md /data/hf_models/bitnet_b1_58-3B

# python compile.py -t -o tuned -da -d jetson -nt 1 -fa -gc -gs 32 -ags 64 -md /data/hf_models/bitnet_b1_58-3B

python compile.py -o tuned -da -nt 1 -tb -gc -ags 64 -t -m hf-bitnet-3b -md /data/hf_models/bitnet_b1_58-3B

