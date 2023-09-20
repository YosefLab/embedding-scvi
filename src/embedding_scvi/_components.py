from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from scvi.utils._exceptions import InvalidParameterError
from torch import nn


class MLPBlock(nn.Module):
    """Multi-layer perceptron block.

    Parameters
    ----------
    n_in
        Number of input features.
    n_out
        Number of output features.
    bias
        Whether to include a bias term in the linear layer.
    norm
        Type of normalization to use. One of the following:

        * ``"batch"``: :class:`~torch.nn.BatchNorm1d`
        * ``"layer"``: :class:`~torch.nn.LayerNorm`
        * ``None``: No normalization
    activation
        Type of activation to use. One of the following:

        * ``"relu"``: :class:`~torch.nn.ReLU`
        * ``"leaky_relu"``: :class:`~torch.nn.LeakyReLU`
        * ``"softmax"``: :class:`~torch.nn.Softmax`
        * ``"softplus"``: :class:`~torch.nn.Softplus`
    dropout_rate
        Dropout rate. If ``None``, no dropout is used.
    residual
        Whether to use residual connections. If ``True`` and ``n_in != n_out``,
        then a linear layer is used to project the input to the correct
        dimensionality.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        bias: bool = True,
        norm: Literal["batch", "layer"] | None = None,
        norm_kwargs: dict | None = None,
        activation: Literal["relu", "leaky_relu", "softmax", "softplus"] | None = None,
        activation_kwargs: dict | None = None,
        dropout_rate: float | None = None,
        residual: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        self.norm = nn.Identity()
        self.norm_kwargs = norm_kwargs or {}
        self.activation = nn.Identity()
        self.activation_kwargs = activation_kwargs or {}
        self.dropout = nn.Identity()
        self.residual = residual

        if norm == "batch":
            self.norm = nn.BatchNorm1d(n_out, **self.norm_kwargs)
        elif norm == "layer":
            self.norm = nn.LayerNorm(n_out, **self.norm_kwargs)
        elif norm is not None:
            raise InvalidParameterError(param="norm", value=norm, valid=["batch", "layer", None])

        if activation == "relu":
            self.activation = nn.ReLU(**self.activation_kwargs)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(**self.activation_kwargs)
        elif activation == "softmax":
            self.activation = nn.Softmax(**self.activation_kwargs)
        elif activation == "softplus":
            self.activation = nn.Softplus(**self.activation_kwargs)
        elif activation is not None:
            raise InvalidParameterError(
                param="norm", value=norm, valid=["relu", "leaky_relu", "softmax", "softplus", None]
            )

        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)

        if residual and n_in != n_out:
            self.residual_transform = nn.Linear(n_in, n_out, bias=False)
        elif residual and n_in == n_out:
            self.residual_transform = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = h + self.residual_transform(x) if self.residual else h
        return h


class MultiOutputLinear(nn.Module):
    """Multi-output linear layer.

    Parameters
    ----------
    n_in
        Number of input features.
    n_out
        Number of output features.
    n_out_params
        Number of output parameters.
    activations
        List containing the type of activation to use for each output parameter.
        One of the following:

        * ``"relu"``: :class:`~torch.nn.ReLU`
        * ``"leaky_relu"``: :class:`~torch.nn.LeakyReLU`
        * ``"softmax"``: :class:`~torch.nn.Softmax`
        * ``"softplus"``: :class:`~torch.nn.Softplus`
        * ``None``: No activation
    activation_kwargs
        List containing the keyword arguments to pass to the activation function
        for each output parameter.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_out_params: int,
        activations: list[int] | None,
        activation_kwargs: list[dict] | None = None,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_out_params = n_out_params
        self.activations = activations or [None] * n_out_params
        self.activation_kwargs = activation_kwargs or [{}] * n_out_params

        blocks = []
        for i in range(self.n_out_params):
            blocks.append(
                MLPBlock(
                    n_in=n_in,
                    n_out=n_out,
                    bias=False,
                    activation=self.activations[i],
                    activation_kwargs=self.activation_kwargs[i],
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(block(x) for block in self.blocks)


class MLP(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int,
        n_layers: int,
        bias: bool = True,
        norm: str | None = None,
        norm_kwargs: dict | None = None,
        activation: str | None = None,
        activation_kwargs: dict | None = None,
        dropout_rate: float | None = None,
        residual: bool = False,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        n_ins = [n_in] + [n_hidden for _ in range(n_layers - 1)]
        n_outs = [n_hidden for _ in range(n_layers - 1)] + [n_out]
        blocks = []
        for n_in, n_out in zip(n_ins, n_outs):
            blocks.append(
                MLPBlock(
                    n_in=n_in,
                    n_out=n_out,
                    bias=bias,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    activation=activation,
                    activation_kwargs=activation_kwargs,
                    dropout_rate=dropout_rate,
                    residual=residual,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ExtendableEmbedding(nn.Embedding):
    """Extendable embedding layer."""

    @classmethod
    def extend_embedding(
        cls,
        embedding: nn.Embedding,
        init: int | list[int],
        freeze_prev: bool = True,
    ):
        # (num_embeddings, embedding_dim)
        old_weight = embedding.weight.clone()

        if isinstance(init, int):
            if init <= 0:
                raise ValueError(f"`init` must be > 0, got {init}")
            n_init = init
            # (n_init, embedding_dim)
            new_weight = torch.empty(
                (init, old_weight.shape[1]),
                device=old_weight.device,
            )
            nn.init.normal_(new_weight)
        elif isinstance(init, list):
            n_init = len(init)
            # (n_init, embedding_dim)
            new_weight = old_weight[init]

        # (num_embeddings + n_init, embedding_dim)
        weight = torch.cat([old_weight, new_weight], dim=0)

        new_embedding = cls(
            num_embeddings=embedding.num_embeddings + n_init,
            embedding_dim=embedding.embedding_dim,
            _weight=weight,
            padding_idx=embedding.padding_idx,
            max_norm=embedding.max_norm,
            norm_type=embedding.norm_type,
            scale_grad_by_freq=embedding.scale_grad_by_freq,
            sparse=embedding.sparse,
        )

        # freeze previous embeddings
        def _partial_freeze_hook(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[: embedding.num_embeddings] = 0.0
            return grad

        if freeze_prev:
            new_embedding.weight.register_hook(_partial_freeze_hook)

        return new_embedding

    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        for key, val in state_dict.items():
            if "weight" not in key:
                continue
            self.weight = nn.Parameter(val)
            self.num_embeddings = val.shape[0]
            self.embedding_dim = val.shape[1]
            break

        return super()._load_from_state_dict(state_dict, *args, **kwargs)


class ExtendableEmbeddingList(nn.Module):
    """List of extendable embedding layers."""

    def __init__(
        self,
        num_embeddings: list[int],
        **kwargs,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings

        self._embeddings = nn.ModuleList(
            [
                ExtendableEmbedding(
                    num_embeddings=c,
                    **kwargs,
                )
                for c in num_embeddings
            ]
        )

    def forward(self, x: torch.Tensor, subset: int | list[int] | None = None) -> torch.Tensor:
        if isinstance(subset, int):
            subset = [subset]
        elif subset is None:
            subset = list(range(len(self._embeddings)))

        embeddings_subset = [self._embeddings[i] for i in subset]

        return torch.cat(
            [embedding(x[:, i]).unsqueeze(0) for i, embedding in enumerate(embeddings_subset)],
        )

    def get_embedding_layer(self, index: int) -> nn.Embedding:
        return self._embeddings[index]

    def set_embedding_layer(self, index: int, embedding: nn.Embedding):
        self._embeddings[index] = embedding

    def extend_embedding_layer(self, index: int, init: int | list[int], freeze_prev: bool = True) -> None:
        self.set_embedding_layer(
            index,
            ExtendableEmbedding.extend_embedding(
                self.get_embedding_layer(index),
                init=init,
                freeze_prev=freeze_prev,
            ),
        )

    def get_embedding_weight(self, index: int, as_tensor: bool = False) -> np.ndarray | torch.Tensor:
        weight = self.get_embedding_layer(index).weight.detach().cpu()
        if as_tensor:
            return weight
        return weight.numpy()
