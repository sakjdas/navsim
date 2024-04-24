import os
current_file_path = os.path.abspath(__file__)
endtoenddriving_Path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))))
# The ego state parameters
n_feature_dim = 11
n_input_frames = 4
n_output_frames = 8
n_output_dim = 3
hidden_size = 512
n_layers = 2
device = 'cuda'
batch_size = 512
lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.
log_interval = 320
eval_freq = 320
