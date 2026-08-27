import unittest

import torch

from transreid_jpm import (
    TRANSREID_FEATURE_DIM,
    checkpoint_model_state,
    checkpoint_spec,
    shuffle_unit,
)


class _ShapeOnlyTensor:
    def __init__(self, *shape):
        self.shape = torch.Size(shape)


class TransReIDJPMTests(unittest.TestCase):
    def test_checkpoint_spec_recognizes_msmt17_jpm_sie_dimensions(self):
        state = {
            "base.pos_embed": _ShapeOnlyTensor(1, 211, 768),
            "base.sie_embed": _ShapeOnlyTensor(15, 1, 768),
            "base.patch_embed.proj.weight": _ShapeOnlyTensor(768, 3, 16, 16),
            "base.fc.weight": _ShapeOnlyTensor(1000, 768),
            "classifier.weight": _ShapeOnlyTensor(1041, 768),
            "b1.0.attn.qkv.weight": _ShapeOnlyTensor(2304, 768),
            "b2.0.attn.qkv.weight": _ShapeOnlyTensor(2304, 768),
        }

        self.assertEqual(
            checkpoint_spec(state),
            {
                "num_classes": 1041,
                "backbone_classes": 1000,
                "camera_count": 15,
                "embed_dim": 768,
            },
        )
        self.assertEqual(TRANSREID_FEATURE_DIM, 3840)

    def test_checkpoint_wrapper_and_ddp_prefix_are_removed(self):
        value = object()
        state = checkpoint_model_state({"model": {"module.base.pos_embed": value}})
        self.assertEqual(state, {"base.pos_embed": value})

    def test_shuffle_unit_matches_official_patch_order(self):
        features = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
        shuffled = shuffle_unit(features, shift=2, group=2)
        self.assertEqual(shuffled.flatten().tolist(), [2.0, 4.0, 3.0, 1.0])


if __name__ == "__main__":
    unittest.main()
