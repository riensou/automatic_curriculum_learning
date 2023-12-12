from typing import Optional

from torch import nn

import torch
from torch import distributions
from torch import optim

import numpy as np

from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.distributions import make_tanh_transformed, make_multi_normal

class MLPPolicy(nn.Module):
    """
    Base MLP policy, which can take an observation and output a distribution over actions.

    This class implements `forward()` which takes a (batched) observation and returns a distribution over actions.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        use_tanh: bool = False,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
    ):
        super().__init__()

        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
        else:
            if self.state_dependent_std:
                assert fixed_std is None
                self.net = ptu.build_mlp(
                    input_size=ob_dim,
                    output_size=2*ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)
            else:
                self.net = ptu.build_mlp(
                    input_size=ob_dim,
                    output_size=ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                ).to(ptu.device)

                if self.fixed_std:
                    self.std = 0.1
                else:
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
                    )


    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            logits = self.logits_net(obs)
            action_distribution = distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                mean, std = torch.chunk(self.net(obs), 2, dim=-1)
                std = torch.nn.functional.softplus(std) + 1e-2
            else:
                mean = self.net(obs)
                if self.fixed_std:
                    std = self.std
                else:
                    std = torch.nn.functional.softplus(self.std) + 1e-2

            if self.use_tanh:
                action_distribution = make_tanh_transformed(mean, std)
            else:
                return make_multi_normal(mean, std)

        return action_distribution
 
class MLPPolicyPG(nn.Module):
    """Policy subclass for the policy gradient algorithm."""

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        use_tanh: bool = False,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
    ):
        super().__init__()

        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
                output_activation="sigmoid"
            ).to(ptu.device)
        else:
            if self.state_dependent_std:
                assert fixed_std is None
                self.net = ptu.build_mlp(
                    input_size=ob_dim,
                    output_size=2*ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                    output_activation="sigmoid"
                ).to(ptu.device)
            else:
                self.net = ptu.build_mlp(
                    input_size=ob_dim,
                    output_size=ac_dim,
                    n_layers=n_layers,
                    size=layer_size,
                    output_activation="sigmoid"
                ).to(ptu.device)

                if self.fixed_std:
                    self.std = 0.1
                else:
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
                    )
    
        self.optimizer = optim.Adam(
                self.net.parameters(),
                5e-3,
        )

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if not torch.is_tensor(obs):
            obs = ptu.from_numpy(obs)
        if self.discrete:
            logits = self.logits_net(obs)
            action_distribution = distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                mean, std = torch.chunk(self.net(obs), 2, dim=-1)
                std = torch.nn.functional.softplus(std) + 1e-2
            else:
                mean = self.net(obs)
                if self.fixed_std:
                    std = self.std
                else:
                    std = torch.nn.functional.softplus(self.std) + 1e-2

            if self.use_tanh:
                action_distribution = make_tanh_transformed(mean, std)
            else:
                return make_multi_normal(mean, std)

        return action_distribution
    
    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        action = ptu.to_numpy(self(obs[None, :]).sample())[0]

        return action

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        loss = -1 * torch.mean(self(obs).log_prob(actions).T * advantages) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "Actor Loss": ptu.to_numpy(loss),
        }