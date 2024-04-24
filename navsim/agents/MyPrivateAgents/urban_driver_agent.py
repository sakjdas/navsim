from typing import Any, List, Dict, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.common.dataclasses import Scene
import navsim.agents.MyPrivateAgents.config as config

import torch


class EgoStatusFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self):
        super().__init__()

    def get_unique_name(self) -> str:
        return "ego_status_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        ego_statuses = agent_input.ego_statuses  # the ego state in the last 2 seconds with 2HZ
        ego_pose_tensors = [torch.tensor(ego_status.ego_pose, dtype=torch.float32) for ego_status in ego_statuses]
        ego_pose_tensor = torch.stack(ego_pose_tensors).view(-1)
        return {"ego_status": ego_pose_tensor}


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        )
        return {"trajectory": torch.tensor(future_trajectory.poses)}


class UrbanDriverAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        lr: float,
        checkpoint_path: str = None,
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path

        self._lr = lr

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(config.n_input_frames * config.n_feature_dim,
                            2 * config.n_feature_dim * self._trajectory_sampling.num_poses),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * config.n_feature_dim * self._trajectory_sampling.num_poses,
                            self._trajectory_sampling.num_poses * config.n_output_dim),
        )

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(
                self._checkpoint_path, map_location=torch.device("cpu")
            )["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [
            TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling),
        ]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [EgoStatusFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        poses: torch.Tensor = self._mlp(features["ego_status"])
        return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)}

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self._mlp.parameters(), lr=self._lr)
