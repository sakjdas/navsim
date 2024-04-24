import torch
import navsim.agents.MyPrivateAgents.config as config


class MyModel:

    class MLP(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backbone = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(config.n_input_frames * config.n_feature_dim, config.hidden_dim_fc),
                torch.nn.ReLU(),
                torch.nn.Linear(config.hidden_dim_fc, config.hidden_dim_fc),
                torch.nn.ReLU(),
                torch.nn.Linear(config.hidden_dim_fc, config.hidden_dim_fc),
                torch.nn.ReLU(),
                torch.nn.Linear(config.hidden_dim_fc, config.n_output_dim * config.n_output_frames),
            )

        def forward(self, x):
            return self.backbone(x)

    class RNN(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            """
            input dim = 2 (the input x, y coordinates), hidden size: The number of features in the hidden state h
            num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to
            form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results.
            """

            self.hidden_size = config.hidden_size
            self.n_layers = config.n_layers
            self.rnn = torch.nn.RNN(input_size=config.n_feature_dim, hidden_size=self.hidden_size,
                                    num_layers=self.n_layers, batch_first=True)
            self.fc_rnn = torch.nn.Linear(in_features=config.hidden_size,
                                          out_features=config.n_output_dim * config.n_output_frames)

        def forward(self, x):
            batch_size, _, _ = x.size()
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device) # Initialize hidden state
            # The shape of x is: batch_size * sequence length * input size
            # the out containing the output features (h_t) from the last layer (the last of the 2 layers) of the RNN, for each t.
            out, _ = self.rnn(x, h0)  # out -> [batch_size, sequence length, hidden_size]
            out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
            # we want the final output be [batch_size, sequence_length, 16]
            out = self.fc_rnn(out)
            return out

    class LSTM(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hidden_size = config.hidden_size
            self.n_layers = config.n_layers
            self.lstm = torch.nn.LSTM(input_size=config.n_feature_dim, hidden_size=self.hidden_size,
                                      num_layers=self.n_layers, batch_first=True)
            self.fc_lstm = torch.nn.Linear(in_features=config.hidden_size,
                                           out_features=config.n_output_dim * config.n_output_frames)

        def forward(self, x):
            # input x is: batch_size * sequence length * input dim
            batch_size, _, _ = x.size()
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
            out, _ = self.lstm(x, (h0, c0))  # out -> [batch_size, sequence length, hidden_size]
            out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
            # we want the final output be [batch_size, sequence_length, 16]
            out = self.fc_lstm(out)
            return out

    class GRU(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hidden_size = config.hidden_size
            self.n_layers = config.n_layers
            self.gru = torch.nn.GRU(input_size=config.n_feature_dim, hidden_size=self.hidden_size,
                                    num_layers=self.n_layers, batch_first=True)
            self.fc_gru = torch.nn.Linear(in_features=config.hidden_size,
                                          out_features=config.n_output_dim * config.n_output_frames)

        def forward(self, x):
            batch_size, _, _ = x.size()
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
            out, _ = self.gru(x, h0)  # out -> [batch_size, sequence length, hidden_size]
            out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
            # we want the final output be [batch_size, sequence_length, 16]
            out = self.fc_gru(out)
            return out
