#!/usr/bin/env python3

import unittest

import torch

from diffsynth.models.wan_video_dit import precompute_freqs_cis
from diffsynth.pipelines.wan_video import build_temporal_rope_freqs, sparse_temporal_rope_freqs


class SparseTemporalRoPETest(unittest.TestCase):
    def setUp(self):
        self.freqs = precompute_freqs_cis(8, end=32)

    def test_49_frames_are_spread_over_81_frame_coordinates(self):
        actual = sparse_temporal_rope_freqs(self.freqs, latent_frames=13, target_num_frames=81)
        expected_positions = torch.linspace(0, 20, 13, dtype=torch.float64)
        angular_frequency = torch.angle(self.freqs[1]).to(torch.float64)
        expected = torch.polar(
            torch.ones((13, angular_frequency.numel()), dtype=torch.float64),
            expected_positions[:, None] * angular_frequency[None, :],
        ).to(self.freqs.dtype)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[0], self.freqs[0])
        torch.testing.assert_close(actual[-1], self.freqs[20])
        torch.testing.assert_close(actual.abs(), torch.ones_like(actual.abs()))

    def test_equal_source_and_target_matches_contiguous_rope(self):
        actual = sparse_temporal_rope_freqs(self.freqs, latent_frames=13, target_num_frames=49)
        torch.testing.assert_close(actual, self.freqs[:13])

    def test_reference_uses_storymem_prefix_and_video_starts_at_zero(self):
        actual = build_temporal_rope_freqs(
            self.freqs,
            total_latent_frames=14,
            reference_size=1,
            target_num_frames=81,
        )

        self.assertEqual(actual.shape[0], 14)
        torch.testing.assert_close(actual[0], self.freqs[5].conj())
        torch.testing.assert_close(actual[1], self.freqs[0])
        torch.testing.assert_close(actual[-1], self.freqs[20])

    def test_reference_without_sparse_rope_preserves_legacy_coordinates(self):
        actual = build_temporal_rope_freqs(
            self.freqs,
            total_latent_frames=14,
            reference_size=1,
        )
        torch.testing.assert_close(actual, self.freqs[:14])

    def test_invalid_target_frame_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "4n\\+1"):
            sparse_temporal_rope_freqs(self.freqs, latent_frames=13, target_num_frames=80)

    def test_shorter_target_timeline_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shorter timeline"):
            sparse_temporal_rope_freqs(self.freqs, latent_frames=13, target_num_frames=33)


if __name__ == "__main__":
    unittest.main()
