import torch
import navsim.agents.MyPrivateAgents.config as config


class MyModel:

    class MLP(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(44, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, config.n_output_frames * 3),
            )

        def forward(self, x):
            return self.backbone(x)
