# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

import torch
from torch.nn import Identity, Linear, Module, ModuleList, Parameter

from heretic.model import Model


class _Mamba2Mixer(Module):
    def __init__(self) -> None:
        super().__init__()
        self.out_proj = Linear(4, 4)


class _AttentionMixer(Module):
    def __init__(self) -> None:
        super().__init__()
        self.o_proj = Linear(4, 4)


class _DenseMixer(Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = Linear(4, 4)


class _Expert(Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = Linear(4, 4)


class _SharedExperts(Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = Linear(4, 4)


class _MoEMixer(Module):
    """One module per routed expert, as the model's own remote code builds it."""

    def __init__(self, num_experts: int = 3, latent: bool = False) -> None:
        super().__init__()
        self.experts = ModuleList([_Expert() for _ in range(num_experts)])
        self.shared_experts = _SharedExperts()
        self.fc2_latent_proj = Linear(4, 4) if latent else Identity()


class _FusedMoEMixer(Module):
    """Every routed expert in one stacked Parameter, as the in-tree implementation builds it."""

    def __init__(self, num_experts: int = 3) -> None:
        super().__init__()
        self.experts = Module()
        self.experts.down_proj = Parameter(torch.empty(num_experts, 4, 4))


class _Layer(Module):
    def __init__(self, mixer: Module) -> None:
        super().__init__()
        self.mixer = mixer


class _Backbone(Module):
    def __init__(self, layers: ModuleList) -> None:
        super().__init__()
        self.layers = layers


class _BackboneModel(Module):
    """A NemotronH-shaped model: the layer stack hangs off `backbone`, not `model`."""

    def __init__(self, mixers: list[Module]) -> None:
        super().__init__()
        self.backbone = _Backbone(ModuleList([_Layer(m) for m in mixers]))


def _model_with(mixers: list[Module]) -> Model:
    # Model.__init__ loads weights; these tests only exercise layer discovery.
    model = Model.__new__(Model)
    model.model = _BackboneModel(mixers)
    return model


class NemotronHLayerDiscoveryTests(unittest.TestCase):
    def test_finds_backbone_rooted_layer_stack(self) -> None:
        model = _model_with([_Mamba2Mixer(), _AttentionMixer()])

        self.assertEqual(len(model.get_layers()), 2)

    def test_maps_mamba2_out_proj(self) -> None:
        model = _model_with([_Mamba2Mixer()])

        self.assertEqual(list(model.get_layer_modules(0)), ["mamba.out_proj"])

    def test_maps_attention_o_proj(self) -> None:
        model = _model_with([_AttentionMixer()])

        self.assertEqual(list(model.get_layer_modules(0)), ["attn.o_proj"])

    def test_maps_dense_mlp_down_proj(self) -> None:
        model = _model_with([_DenseMixer()])

        self.assertEqual(len(model.get_layer_modules(0)["mlp.down_proj"]), 1)

    def test_maps_every_routed_expert_and_the_shared_expert(self) -> None:
        model = _model_with([_MoEMixer(num_experts=3)])

        # Three routed experts plus one shared expert.
        self.assertEqual(len(model.get_layer_modules(0)["mlp.down_proj"]), 4)

    def test_maps_latent_projection_when_present(self) -> None:
        model = _model_with([_MoEMixer(num_experts=3, latent=True)])

        self.assertEqual(len(model.get_layer_modules(0)["mlp.down_proj"]), 5)

    def test_ignores_identity_latent_projection(self) -> None:
        model = _model_with([_MoEMixer(num_experts=3, latent=False)])

        self.assertEqual(len(model.get_layer_modules(0)["mlp.down_proj"]), 4)

    def test_discovers_components_across_mixed_layers(self) -> None:
        model = _model_with([_Mamba2Mixer(), _AttentionMixer(), _MoEMixer()])

        self.assertEqual(
            model.get_abliterable_components(),
            ["attn.o_proj", "mamba.out_proj", "mlp.down_proj"],
        )


class FusedExpertDetectionTests(unittest.TestCase):
    def test_reports_fused_experts(self) -> None:
        model = _model_with([_FusedMoEMixer()])

        self.assertTrue(model.has_fused_experts())

    def test_does_not_report_experts_held_as_modules(self) -> None:
        model = _model_with([_MoEMixer()])

        self.assertFalse(model.has_fused_experts())

    def test_fused_experts_carry_no_abliterable_module(self) -> None:
        # This is the failure the warning exists for: a stacked Parameter cannot take
        # a LoRA adapter, so a model whose only MoE weights are fused yields nothing.
        model = _model_with([_FusedMoEMixer()])

        with self.assertRaises(AssertionError):
            model.get_layer_modules(0)


if __name__ == "__main__":
    unittest.main()
