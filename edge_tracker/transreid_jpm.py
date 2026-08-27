"""Inference-only TransReID JPM/SIE model for the MSMT17 checkpoint.

The module follows the official damo-cv/TransReID ``build_transformer_local``
evaluation path.  It reuses the compatible ViT blocks already vendored in the
project's FastReID folder, while preserving the original checkpoint key names
(``base``, ``b1``, ``b2``, bottlenecks, and classifiers) for strict loading.
"""

import copy

import torch
from torch import nn


TRANSREID_IMAGE_SIZE = (256, 128)
TRANSREID_STRIDE_SIZE = (12, 12)
TRANSREID_EMBED_DIM = 768
TRANSREID_LOCAL_PARTS = 4
TRANSREID_FEATURE_DIM = TRANSREID_EMBED_DIM * (1 + TRANSREID_LOCAL_PARTS)


def shuffle_unit(features, shift=5, group=2, begin=1):
    """Official TransReID JPM shift-and-patch-shuffle operation."""

    batch_size = features.size(0)
    dimension = features.size(-1)
    shifted = torch.cat(
        [features[:, begin - 1 + shift :], features[:, begin : begin - 1 + shift]],
        dim=1,
    )
    try:
        shuffled = shifted.view(batch_size, group, -1, dimension)
    except RuntimeError:
        shifted = torch.cat([shifted, shifted[:, -2:-1, :]], dim=1)
        shuffled = shifted.view(batch_size, group, -1, dimension)
    shuffled = torch.transpose(shuffled, 1, 2).contiguous()
    return shuffled.view(batch_size, -1, dimension)


def checkpoint_model_state(checkpoint):
    """Return a checkpoint state dict with a possible DDP prefix removed."""

    state = checkpoint
    if isinstance(state, dict):
        for wrapper_key in ("model", "state_dict"):
            wrapped = state.get(wrapper_key)
            if isinstance(wrapped, dict):
                state = wrapped
                break
    if not isinstance(state, dict):
        raise TypeError("TransReID checkpoint does not contain a state dictionary.")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def checkpoint_spec(state):
    """Infer and validate the architecture-defining checkpoint dimensions."""

    required = (
        "base.pos_embed",
        "base.sie_embed",
        "base.patch_embed.proj.weight",
        "base.fc.weight",
        "classifier.weight",
        "b1.0.attn.qkv.weight",
        "b2.0.attn.qkv.weight",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Checkpoint is not a JPM/SIE TransReID model; missing: {missing}")

    num_classes, embed_dim = map(int, state["classifier.weight"].shape)
    backbone_classes = int(state["base.fc.weight"].shape[0])
    camera_count = int(state["base.sie_embed"].shape[0])
    position_tokens = int(state["base.pos_embed"].shape[1])
    if embed_dim != TRANSREID_EMBED_DIM:
        raise ValueError(f"Expected a 768-dimensional ViT-Base checkpoint, got {embed_dim}.")
    if position_tokens != 211:
        raise ValueError(
            "Expected 211 position tokens for 256x128 input with stride 12, "
            f"got {position_tokens}."
        )
    return {
        "num_classes": num_classes,
        "backbone_classes": backbone_classes,
        "camera_count": camera_count,
        "embed_dim": embed_dim,
    }


class TransReIDJPM(nn.Module):
    """Official TransReID global-plus-four-local inference architecture."""

    def __init__(
        self,
        vision_transformer_class,
        num_classes,
        camera_count,
        backbone_classes=1000,
    ):
        super().__init__()
        self.base = vision_transformer_class(
            img_size=TRANSREID_IMAGE_SIZE,
            patch_size=16,
            stride_size=TRANSREID_STRIDE_SIZE,
            embed_dim=TRANSREID_EMBED_DIM,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            camera=int(camera_count),
            drop_path_rate=0.1,
            sie_xishu=3.0,
        )

        final_block = self.base.blocks[-1]
        final_norm = self.base.norm
        self.b1 = nn.Sequential(copy.deepcopy(final_block), copy.deepcopy(final_norm))
        self.b2 = nn.Sequential(copy.deepcopy(final_block), copy.deepcopy(final_norm))

        # The official backbone keeps this unused training classifier in its
        # state dict.  Retaining it lets us verify every checkpoint tensor.
        self.base.fc = nn.Linear(TRANSREID_EMBED_DIM, int(backbone_classes))

        self.classifier = nn.Linear(TRANSREID_EMBED_DIM, int(num_classes), bias=False)
        self.classifier_1 = nn.Linear(TRANSREID_EMBED_DIM, int(num_classes), bias=False)
        self.classifier_2 = nn.Linear(TRANSREID_EMBED_DIM, int(num_classes), bias=False)
        self.classifier_3 = nn.Linear(TRANSREID_EMBED_DIM, int(num_classes), bias=False)
        self.classifier_4 = nn.Linear(TRANSREID_EMBED_DIM, int(num_classes), bias=False)

        self.bottleneck = self._make_bottleneck()
        self.bottleneck_1 = self._make_bottleneck()
        self.bottleneck_2 = self._make_bottleneck()
        self.bottleneck_3 = self._make_bottleneck()
        self.bottleneck_4 = self._make_bottleneck()

        self.shift_num = 5
        self.shuffle_groups = 2
        self.divide_length = TRANSREID_LOCAL_PARTS

    @staticmethod
    def _make_bottleneck():
        layer = nn.BatchNorm1d(TRANSREID_EMBED_DIM)
        layer.bias.requires_grad_(False)
        return layer

    def _base_tokens(self, images, camera_labels=None):
        batch_size = images.shape[0]
        tokens = self.base.patch_embed(images)
        class_tokens = self.base.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1)

        if self.base.cam_num > 0:
            if camera_labels is None:
                camera_labels = torch.zeros(
                    batch_size,
                    dtype=torch.long,
                    device=images.device,
                )
            tokens = (
                tokens
                + self.base.pos_embed
                + self.base.sie_xishu * self.base.sie_embed[camera_labels]
            )
        else:
            tokens = tokens + self.base.pos_embed

        tokens = self.base.pos_drop(tokens)
        for block in self.base.blocks[:-1]:
            tokens = block(tokens)
        return tokens

    def forward(self, images, camera_labels=None):
        features = self._base_tokens(images, camera_labels=camera_labels)

        global_feature = self.b1(features)[:, 0]
        patch_length = (features.size(1) - 1) // self.divide_length
        class_token = features[:, 0:1]
        shuffled = shuffle_unit(
            features,
            shift=self.shift_num,
            group=self.shuffle_groups,
        )

        local_features = []
        for part_index in range(self.divide_length):
            start = part_index * patch_length
            end = start + patch_length
            local_tokens = torch.cat((class_token, shuffled[:, start:end]), dim=1)
            local_features.append(self.b2(local_tokens)[:, 0])

        # MSMT17's published config uses TEST.NECK_FEAT='before'.  The local
        # branches are divided by four exactly as in the official evaluator.
        return torch.cat(
            [global_feature, *(feature / 4.0 for feature in local_features)],
            dim=1,
        )


def build_transreid_jpm_from_checkpoint(checkpoint, vision_transformer_class):
    """Build the exact architecture and require every checkpoint key to load."""

    state = checkpoint_model_state(checkpoint)
    spec = checkpoint_spec(state)
    model = TransReIDJPM(
        vision_transformer_class,
        num_classes=spec["num_classes"],
        camera_count=spec["camera_count"],
        backbone_classes=spec["backbone_classes"],
    )
    model.load_state_dict(state, strict=True)
    return model, spec
