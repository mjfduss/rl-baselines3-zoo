from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn

from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy

# Custom LSTM Cell
from rl_zoo3.custom_cfc import Cfc


class MlpCfcPolicy(MlpLstmPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            use_sde: bool = False,
            lstm_hidden_size: int = 256,
            lstm_kwargs: Dict[str, Any] | None = None):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            use_sde=use_sde,
            lstm_hidden_size=lstm_hidden_size
        )

        self.lstm_actor = Cfc(
            in_features=observation_space.shape[0],
            hidden_size=lstm_hidden_size,
            out_feature=lstm_kwargs["out_feature"],
            hparams=lstm_kwargs["hparams"],
        )

        self.lstm_critic = Cfc(
            in_features=observation_space.shape[0],
            hidden_size=lstm_hidden_size,
            out_feature=lstm_kwargs["out_feature"],
            hparams=lstm_kwargs["hparams"],
        )
