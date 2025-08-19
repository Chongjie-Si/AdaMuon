wandb_log = True
wandb_project = 'GPT2'
wandb_run_name='gpt2-large-adamuon-100k'

batch_size = 15
block_size = 1024
gradient_accumulation_steps = 4

# GPT large config
n_layer = 36
n_head = 20
n_embd = 1280

max_iters = 100000

# optimizer
optimizer_name = 'adamuon'
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

min_lr = 6e-5 

out_dir_name = 'large_adamuon_100k'
