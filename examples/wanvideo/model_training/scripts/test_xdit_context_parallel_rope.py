import ast
import unittest
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "diffsynth"
    / "utils"
    / "xfuser"
    / "xdit_context_parallel.py"
)


def _load_rope_functions(sp_size=1, sp_rank=0):
    """Load the real function bodies without requiring optional xFuser deps."""
    tree = ast.parse(MODULE_PATH.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"pad_freqs", "rope_apply"}
    ]
    module = ast.Module(body=selected, type_ignores=[])

    def rearrange_heads(tensor, pattern, n):
        assert pattern == "b s (n d) -> b s n d"
        return tensor.reshape(tensor.shape[0], tensor.shape[1], n, -1)

    namespace = {
        "torch": torch,
        "rearrange": rearrange_heads,
        "get_sequence_parallel_world_size": lambda: sp_size,
        "get_sequence_parallel_rank": lambda: sp_rank,
    }
    exec(compile(module, str(MODULE_PATH), "exec"), namespace)
    return namespace["pad_freqs"], namespace["rope_apply"]


class XditContextParallelRopeTest(unittest.TestCase):
    def test_pad_freqs_returns_unchanged_tensor_when_long_enough(self):
        pad_freqs, _ = _load_rope_functions()
        freqs = torch.randn(4, 1, 3, dtype=torch.complex64)

        result = pad_freqs(freqs, 4)

        self.assertEqual(result.data_ptr(), freqs.data_ptr())
        self.assertTrue(torch.equal(result, freqs))

    def test_pad_freqs_uses_complex_rotary_identity(self):
        pad_freqs, _ = _load_rope_functions()
        freqs = torch.randn(3, 1, 2, dtype=torch.complex64)

        result = pad_freqs(freqs, 5)

        self.assertEqual(result.shape, (5, 1, 2))
        self.assertEqual(result.dtype, torch.complex64)
        self.assertTrue(torch.equal(result[:3], freqs))
        self.assertTrue(
            torch.equal(result[3:], torch.ones(2, 1, 2, dtype=torch.complex64))
        )

    def test_rope_apply_matches_complex64_reference_and_restores_dtype(self):
        pad_freqs, rope_apply = _load_rope_functions(sp_size=2, sp_rank=1)

        for input_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(input_dtype=input_dtype):
                x = torch.arange(1, 1 + 1 * 3 * 8, dtype=torch.float32)
                x = x.reshape(1, 3, 8).to(input_dtype).requires_grad_(True)
                freqs = torch.polar(
                    torch.ones(5, 1, 2, dtype=torch.float64),
                    torch.linspace(0, 1, 10, dtype=torch.float64).reshape(5, 1, 2),
                )

                result = rope_apply(x, freqs, num_heads=2)

                x_heads = x.reshape(1, 3, 2, 4)
                x_complex = torch.view_as_complex(
                    x_heads.float().reshape(1, 3, 2, 2, 2).contiguous()
                )
                padded_freqs = pad_freqs(freqs.to(torch.complex64), 6)
                expected_freqs = padded_freqs[3:6]
                expected = torch.view_as_real(
                    x_complex * expected_freqs
                ).flatten(2).to(input_dtype)

                self.assertEqual(result.dtype, input_dtype)
                self.assertTrue(torch.equal(result, expected))

                result.float().sum().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
