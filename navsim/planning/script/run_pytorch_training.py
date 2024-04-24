from typing import Tuple
import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_torch_module import AgentPytorchModel
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.agents.abstract_agent import AbstractAgent
import navsim.agents.MyPrivateAgents.config as config
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def monitor_cuda_memory():
    print("CUDA memory allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("CUDA memory cached:", torch.cuda.memory_reserved() / 1024**2, "MB")


def train_val(model, train_dataloader, val_dataloader, agent, cfg, device, early_stopping):
    """
    Train and validate the model.

    Args:
        model (torch.nn.Module): The neural network model.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        early_stopping : Early stopping criterion
        agent: the agent for training
        cfg: config file
        device: cuda or cpu

    """
    print(f"Batch size is {cfg.dataloader.params.batch_size}, one epoch has {len(train_dataloader)} iterations")

    # Declare the loss function and optimizer, using the value from CLI
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps,
                                 weight_decay=config.weight_decay)
    # TensorBoard writer
    writer = SummaryWriter(f"runs/{datetime.now().strftime('%m-%d %H:%M')}")

    # Set model to train, important if the network has dropout or batch norm layers
    model.train()
    best_loss = float('inf')
    for epoch in range(cfg.trainer.params.max_epochs):

        train_loss_running = 0.
        for i, batch in enumerate(train_dataloader):

            # Move input_data and target_labels to device
            features, targets = batch[0].to(device), batch[1].to(device)
            # 1 Zero out gradients from last iteration
            optimizer.zero_grad()
            # 2 Perform forward pass
            prediction = model(features)
            # 3 Calculate loss
            loss = agent.compute_loss(features, targets, prediction)
            # 4 Compute gradients
            loss.backward()
            # 5 Adjust weights using the optimizer
            optimizer.step()
            # del input_data, target_labels
            # torch.cuda.empty_cache()
            # gc.collect()

            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + i
            if iteration % config.log_interval == (config.log_interval - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config.log_interval:.3f}')
                # Write train loss to TensorBoard
                writer.add_scalars('Loss', {'train': train_loss_running / config.log_interval}, iteration)
                train_loss_running = 0.
            # Validation evaluation and logging
            if iteration % config.eval_freq == (config.eval_freq - 1):
                model.eval()
                loss_val = 0.
                for batch_val in val_dataloader:
                    features, targets = batch_val[0].to(device), batch_val[1].to(device)
                    with torch.no_grad():
                        prediction = model(features)

                    loss_val += agent.compute_loss(features, targets, prediction).item()

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_val / len(val_dataloader):.3f}')
                writer.add_scalars('Loss', {'validation': loss_val / len(val_dataloader)}, iteration)
                # early stopping:
                if early_stopping.early_stop(loss_val / len(val_dataloader)):
                    # stop training phase
                    return
                # Saving the best checkpoints
                if loss_val / len(val_dataloader) < best_loss:
                    best_model_path = os.path.join(cfg.output_dir, 'pt_models/')
                    # create best_model folder if not exists
                    if not os.path.exists(best_model_path):
                        os.makedirs(best_model_path)
                    torch.save(model.state_dict(), os.path.join(best_model_path, f'model_best_{datetime.now()}.ckpt'))
                    best_loss = loss_val / len(val_dataloader)
                # del input_data, target_labels
                # Set model back to train
                model.train()
    writer.close()


class EarlyStopper:
    def __init__(self, patience=30, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    train_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [l for l in train_scene_filter.log_names if l in cfg.train_logs]
    else:
        train_scene_filter.log_names = cfg.train_logs  # 使用 config.train_logs里面的场景

    val_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [l for l in val_scene_filter.log_names if l in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs  # 使用 config.val_logs里面的场景

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    # 是一个长度为2482的dict,每一个dict长度为 n_history_frames + n_future_frames这里是14（没太懂14是什么）
    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    logger.info("Global Seed set to 0")
    logger.info(f"Path where all results are stored: {cfg.output_dir}")
    logger.info("Building Agent")
    # 实例化创建的Agent
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building pytorch Module")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: ", device)
    model = AgentPytorchModel(agent=agent).to(device)
    print(model)

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert cfg.force_cache_computation == False, "force_cache_computation must be False when using cached data without building SceneLoader"
        assert cfg.cache_path is not None, "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    # 这个dataloader就相当于我们前面指定了只需要速度，加速度，driving command作为输入，xy——theta作为输出，但是是一个batch
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    early_stopper = EarlyStopper(patience=10, min_delta=0.0)

    logger.info("Starting Trainer")
    train_val(model, train_dataloader, val_dataloader, agent, cfg, device, early_stopper)


if __name__ == "__main__":
    main()
