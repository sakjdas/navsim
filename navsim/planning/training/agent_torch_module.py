from navsim.agents.abstract_agent import AbstractAgent
import torch


class AgentPytorchModel(torch.nn.Module):
    def __init__(
        self,
        agent: AbstractAgent,
    ):
        super().__init__()
        self.agent = agent

    def forward(self, x):
        return self.agent.forward(x)




