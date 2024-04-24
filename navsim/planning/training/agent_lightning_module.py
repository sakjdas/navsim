import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    def __init__(
        self,
        agent: AbstractAgent,
    ):
        super().__init__()
        self.agent = agent

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        logging_prefix: str,
    ):
        # 解析features和targets,然后使用定义的agent的forward函数进行predict
        features, targets = batch
        prediction = self.agent.forward(features)
        # 使用features, targets计算loss
        loss = self.agent.compute_loss(features, targets, prediction)
        # 打印loss并返回
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # 使用我们定义的optimizer
        return self.agent.get_optimizers()
