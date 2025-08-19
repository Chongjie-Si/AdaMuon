# The project path
export ROOT="./gpt2"

torchrun --standalone --nproc_per_node=8 \
      $ROOT/models/train.py \
      $ROOT/config/small.py
      