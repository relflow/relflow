"""Vocabulary row storage whose learned shape survives rebuilds and checkpoints."""

from typing import Any

import torch

from relflow.helpers.state import compatible


@torch.no_grad()
def allocate(parameter: torch.Tensor, shape: torch.Size) -> None:
    """Prepare checkpoint storage without invalidating optimizer references.

    Checkpoint restoration is outside forward/backward and restores optimizer
    state separately. Gradients for a different shape cannot be retained.
    """
    storage = parameter.new_empty(shape)
    parameter.set_(storage.untyped_storage(), 0, shape, storage.stride())
    parameter.grad = None


class Embedding(torch.nn.Embedding):
    def rebuild_state(self, previous: torch.nn.Module) -> dict[str, Any]:
        if not isinstance(previous, Embedding):
            return {}
        if self.num_embeddings != previous.num_embeddings:
            self.num_embeddings = previous.num_embeddings
            self.padding_idx = previous.padding_idx
            self.weight = torch.nn.Parameter(
                self.weight.new_empty((self.num_embeddings, self.embedding_dim)),
                requires_grad=self.weight.requires_grad,
            )
            self.reset_parameters()
        return compatible(self.state_dict(), previous.state_dict())

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        weight = state_dict.get(prefix + "weight")
        if isinstance(weight, torch.Tensor) and weight.ndim == 2 and weight.shape[1] == self.embedding_dim:
            if weight.shape[0] != self.num_embeddings:
                self.num_embeddings = weight.shape[0]
                allocate(self.weight, weight.shape)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class Linear(torch.nn.Linear):
    def rebuild_state(self, previous: torch.nn.Module) -> dict[str, Any]:
        if not isinstance(previous, Linear):
            return {}
        if self.out_features != previous.out_features:
            self.out_features = previous.out_features
            self.weight = torch.nn.Parameter(
                self.weight.new_empty((self.out_features, self.in_features)), requires_grad=self.weight.requires_grad
            )
            if self.bias is not None:
                self.bias = torch.nn.Parameter(
                    self.bias.new_empty(self.out_features), requires_grad=self.bias.requires_grad
                )
            self.reset_parameters()
        return compatible(self.state_dict(), previous.state_dict())

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        weight = state_dict.get(prefix + "weight")
        if isinstance(weight, torch.Tensor) and weight.ndim == 2 and weight.shape[1] == self.in_features:
            if weight.shape[0] != self.out_features:
                self.out_features = weight.shape[0]
                allocate(self.weight, weight.shape)
                if self.bias is not None:
                    allocate(self.bias, torch.Size((self.out_features,)))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


__all__ = ["Embedding", "Linear"]
