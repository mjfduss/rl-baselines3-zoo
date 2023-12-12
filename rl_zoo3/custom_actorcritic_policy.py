from typing import Any, Dict, List, Type
from gymnasium.spaces import Space
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)

from rl_zoo3.custom_cfc import Cfc

class CfCActorCriticPolicy(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms with CfC (has both policy and value prediction).

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """
    def __init__(self, 
                 observation_space: Space, 
                 action_space: Space, 
                 lr_schedule: Schedule, 
                 net_arch: List[int] | Dict[str, List[int]] | None = None, 
                 activation_fn: type[nn.Module] = nn.Tanh, 
                 ortho_init: bool = True, 
                 use_sde: bool = False, 
                 log_std_init: float = 0, 
                 full_std: bool = True, 
                 use_expln: bool = False, 
                 squash_output: bool = False, 
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor, 
                 features_extractor_kwargs: Dict[str, Any] | None = None, 
                 share_features_extractor: bool = True, 
                 normalize_images: bool = True, 
                 optimizer_class: type[th.optim.Optimizer] = th.optim.Adam, 
                 optimizer_kwargs: Dict[str, Any] | None = None,):
        
        super().__init__(observation_space, 
                         action_space, 
                         lr_schedule, 
                         net_arch, 
                         activation_fn, 
                         ortho_init, 
                         use_sde, 
                         log_std_init, 
                         full_std, 
                         use_expln, 
                         squash_output, 
                         features_extractor_class, 
                         features_extractor_kwargs, 
                         share_features_extractor, 
                         normalize_images, 
                         optimizer_class, 
                         optimizer_kwargs)
        
        
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        
        cfc_hparams: Dict[str, Any] = dict(
                      hidden_size=64,
                      backbone_activation="lecun",
                      backbone_units=64,
                      backbone_dr=0.3,
                      backbone_layers=2,
                      weight_decay=4e-06,
                      optim="adamw",
                      init=0.6,
                      batch_size=128,
                      use_mixed=True,
                      no_gate=False,
                      minimal=False,
                      use_ltc=False,
                    )
        
        print("in_features=", self.mlp_extractor.latent_dim_vf)
        self.value_net =  Cfc(in_features=self.mlp_extractor.latent_dim_vf,
                              hidden_size=cfc_hparams["hidden_size"],
                              out_feature=1,
                              hparams=cfc_hparams,
                              use_mixed=cfc_hparams["use_mixed"],
                              use_ltc=cfc_hparams["use_ltc"]
                              )
        
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]    
        