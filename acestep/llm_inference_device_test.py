"""Unit tests for device-string handling in ``LLMHandler`` initialization."""

import unittest
from unittest.mock import MagicMock, patch

try:
    import torch

    from acestep.llm_inference import LLMHandler

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - dependency guard
    torch = None
    LLMHandler = None
    _IMPORT_ERROR = exc


@unittest.skipIf(LLMHandler is None, f"llm_inference import unavailable: {_IMPORT_ERROR}")
class LlmDeviceHandlingTests(unittest.TestCase):
    """Verify cuda:N strings are treated as CUDA devices."""

    def test_initialize_cuda_index_defaults_to_bfloat16(self):
        """Device 'cuda:1' should pick CUDA dtype defaults (bfloat16) when dtype=None."""
        handler = LLMHandler()

        def _load_pytorch_stub(_model_path: str, _device: str):
            handler.llm_initialized = True
            return True, "ok"

        fake_gpu_config = MagicMock(max_duration_with_lm=480, tier="tier5")

        with (
            patch("acestep.llm_inference.os.path.exists", return_value=True),
            patch("acestep.llm_inference.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("acestep.llm_inference.MetadataConstrainedLogitsProcessor", return_value=MagicMock()),
            patch("acestep.llm_inference.get_global_gpu_config", return_value=fake_gpu_config),
            patch("torch.cuda.is_available", return_value=True),
            patch.object(handler, "_load_pytorch_model", side_effect=_load_pytorch_stub) as load_mock,
        ):
            _status, ok = handler.initialize(
                checkpoint_dir="checkpoints",
                lm_model_path="acestep-5Hz-lm-4B",
                backend="vllm",  # falls back to pt because device != "cuda"
                device="cuda:1",
                offload_to_cpu=False,
                dtype=None,
            )

        self.assertTrue(ok)
        self.assertTrue(handler.llm_initialized)
        self.assertEqual(handler.device, "cuda:1")
        self.assertEqual(handler.dtype, torch.bfloat16)
        self.assertEqual(load_mock.call_args[0][1], "cuda:1")

    def test_load_pytorch_model_uses_torch_dtype(self):
        """_load_pytorch_model should request weights in self.dtype to reduce peak memory."""
        handler = LLMHandler()
        handler.dtype = torch.bfloat16
        handler.offload_to_cpu = False

        model_mock = MagicMock()
        model_mock.to.return_value = model_mock

        with patch(
            "acestep.llm_inference.AutoModelForCausalLM.from_pretrained",
            return_value=model_mock,
        ) as from_pretrained_mock:
            ok, _status = handler._load_pytorch_model("dummy-model", "cuda:1")

        self.assertTrue(ok)
        _args, kwargs = from_pretrained_mock.call_args
        self.assertEqual(kwargs.get("torch_dtype"), torch.bfloat16)
        self.assertTrue(kwargs.get("low_cpu_mem_usage"))

        _to_args, to_kwargs = model_mock.to.call_args
        self.assertEqual(to_kwargs.get("device"), "cuda:1")
        self.assertEqual(to_kwargs.get("dtype"), torch.bfloat16)


if __name__ == "__main__":
    unittest.main()

