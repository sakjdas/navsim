from typing import Tuple
import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


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
    pl.seed_everything(0, workers=True)

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    # 实例化创建的Agent
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    # 创建一个lightning model方便我们之后去训练
    lightning_module = AgentLightningModule(
        agent=agent,
    )

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

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
