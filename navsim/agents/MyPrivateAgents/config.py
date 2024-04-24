"""1. General"""
device = 'cuda'
"""2. Input"""
# General
n_input_frames = 4
# Ego state
n_feature_dim = 11
# Images

# Lidar

"""2. Output"""
n_output_frames = 8
n_output_dim = 3

"""3. Model"""
# LSTM, GRU, RNN
hidden_size = 512
n_layers = 2

"""4. Training & Evaluation"""
batch_size = 512
lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.
log_interval = 320
eval_freq = 320
