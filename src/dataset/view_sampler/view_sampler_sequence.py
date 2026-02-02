from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor
import numpy as np

from .three_view_hack import add_third_context_index
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerSequenceCfg:
    name: Literal["sequence"]
    num_context_views: int
    num_target_views: int
    max_img_per_gpu: int
    min_gap: int
    max_gap: int


class ViewSamplerSequence(ViewSampler[ViewSamplerSequenceCfg]):
    def sample_sequence_with_gaps(
        self,
        num_views: int,
        num_samples: int,
        min_gap: int,
        max_gap: int,
        device: torch.device
    ) -> Int64[Tensor, " num_samples"]:
        """Sample strictly increasing sequence with random gaps within range."""
        assert num_samples >= 1, "At least one sample required."
        assert min_gap <= max_gap, "min_gap must be <= max_gap."
        required_length = (num_samples - 1) * min_gap + 1
        assert num_views >= required_length, (
            f"Not enough views to sample. Need at least {required_length} views."
        )

        max_start_index = num_views - (num_samples - 1) * min_gap
        start_index = torch.randint(0, max_start_index, (1,), device=device).item()

        indices = [start_index]
        current_index = start_index

        for remaining_samples in range(num_samples - 1, 0, -1):
            max_possible_gap = min(
                max_gap,
                num_views - current_index - remaining_samples * min_gap
            )
            gap = torch.randint(min_gap, max_possible_gap + 1, (1,), device=device).item()
            current_index += gap
            indices.append(current_index)

        return indices#torch.tensor(indices, dtype=torch.int64, device=device)

    def sample(
        self,
        scene: str,
        num_context_views: int,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
        Float[Tensor, " overlap"],
    ]:
        """Sample context and target views as ordered sequences with constrained gaps."""
        num_views, _, _ = extrinsics.shape

        index_all = self.sample_sequence_with_gaps(
            num_views=num_views,
            num_samples=num_context_views+self.cfg.num_target_views,
            min_gap=self.cfg.min_gap,
            max_gap=self.cfg.max_gap,
            device=device
        )


        if num_context_views == 3 and len(index_all) == 2:
            index_context = torch.tensor(index_all, dtype=torch.int64, device=device)
            index_context = add_third_context_index(index_context)
            index_target = index_context.clone()
        else:
            # sort index_all
            index_all = sorted(index_all)
            index_context = [index_all[0], index_all[-1]]
            index_all = index_all[1:-1]
            np.random.shuffle(index_all)
            index_context = index_context+ index_all[:num_context_views - 2]
            index_target = index_all[num_context_views-2:]
            index_context = torch.tensor(sorted(index_context), dtype=torch.int64, device=device)
            index_target = torch.tensor(sorted(index_target), dtype=torch.int64, device=device)

        # index_target = self.sample_sequence_with_gaps(
        #     num_views=num_views,
        #     num_samples=self.cfg.num_target_views,
        #     min_gap=self.cfg.min_gap,
        #     max_gap=self.cfg.max_gap,
        #     device=device
        # )

        overlap = torch.tensor([0.5], dtype=torch.float32, device=device)  # Dummy overlap

        return index_context, index_target, overlap

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
