"""Unit tests for LlamaCppEmbedAdapter and LlamaCppClassifyAdapter."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llama_cpp():
    m = MagicMock()
    m.Llama = MagicMock()
    return m


def _reset_cache():
    import mata.adapters.llamacpp_base as base_mod

    base_mod._llama_cpp = None
    base_mod.LLAMA_CPP_AVAILABLE = None


def _make_embed_adapter(tmp_path, mock_llama_cpp, embed_return=None, **kwargs):
    """Create LlamaCppEmbedAdapter with a mock LLM."""
    gguf = tmp_path / "embed.gguf"
    gguf.write_bytes(b"")

    fake_llm = MagicMock()
    if embed_return is not None:
        fake_llm.embed.return_value = embed_return
    mock_llama_cpp.Llama.return_value = fake_llm

    _reset_cache()
    with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
        from mata.adapters.llamacpp_embed_adapter import LlamaCppEmbedAdapter

        adapter = LlamaCppEmbedAdapter(model_path=str(gguf), **kwargs)
    _reset_cache()
    return adapter, fake_llm


def _make_classify_adapter(tmp_path, mock_llama_cpp, embed_returns=None, **kwargs):
    """Create LlamaCppClassifyAdapter with a mock LLM."""
    gguf = tmp_path / "classify.gguf"
    gguf.write_bytes(b"")

    fake_llm = MagicMock()
    if embed_returns is not None:
        fake_llm.embed.side_effect = embed_returns
    mock_llama_cpp.Llama.return_value = fake_llm

    _reset_cache()
    with patch.dict(sys.modules, {"llama_cpp": mock_llama_cpp}):
        from mata.adapters.llamacpp_classify_adapter import LlamaCppClassifyAdapter

        adapter = LlamaCppClassifyAdapter(model_path=str(gguf), **kwargs)
    _reset_cache()
    return adapter, fake_llm


def _make_pil_image(tmp_path, name="img.png"):
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=(50, 100, 150))
    path = tmp_path / name
    img.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# LlamaCppEmbedAdapter — __init__
# ---------------------------------------------------------------------------


class TestLlamaCppEmbedAdapterInit:
    def test_llama_created_with_embedding_true(self, tmp_path):
        """Constructor must call Llama(embedding=True)."""
        mock = _mock_llama_cpp()
        _make_embed_adapter(tmp_path, mock)
        _, call_kwargs = mock.Llama.call_args
        assert call_kwargs.get("embedding") is True

    def test_raises_unsupported_model_error_when_embedding_fails(self, tmp_path):
        """When Llama(embedding=True) raises, UnsupportedModelError is re-raised."""
        from mata.core.exceptions import UnsupportedModelError

        gguf = tmp_path / "bad.gguf"
        gguf.write_bytes(b"")
        mock = _mock_llama_cpp()
        mock.Llama.side_effect = RuntimeError("embedding not supported")

        _reset_cache()
        with patch.dict(sys.modules, {"llama_cpp": mock}):
            from mata.adapters.llamacpp_embed_adapter import LlamaCppEmbedAdapter

            with pytest.raises(UnsupportedModelError, match="embedding mode"):
                LlamaCppEmbedAdapter(model_path=str(gguf))
        _reset_cache()

    def test_embedding_dim_is_none_before_predict(self, tmp_path):
        mock = _mock_llama_cpp()
        adapter, _ = _make_embed_adapter(tmp_path, mock)
        assert adapter.embedding_dim is None


# ---------------------------------------------------------------------------
# LlamaCppEmbedAdapter — predict()
# ---------------------------------------------------------------------------


class TestLlamaCppEmbedAdapterPredict:
    def _make_crops(self, n=2, d=128):
        return [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(n)]

    def test_predict_returns_correct_shape(self, tmp_path):
        """predict([crop1, crop2]) → (2, D) float32 array."""
        dim = 128
        mock = _mock_llama_cpp()
        adapter, fake_llm = _make_embed_adapter(tmp_path, mock)
        fake_llm.embed.return_value = [float(i) for i in range(dim)]

        crops = self._make_crops(n=2)
        result = adapter.predict(crops)
        assert result.shape == (2, dim)
        assert result.dtype == np.float32

    def test_predict_single_crop_returns_1xd(self, tmp_path):
        dim = 64
        mock = _mock_llama_cpp()
        adapter, fake_llm = _make_embed_adapter(tmp_path, mock)
        fake_llm.embed.return_value = [float(i) for i in range(dim)]

        result = adapter.predict(self._make_crops(n=1))
        assert result.shape == (1, dim)

    def test_predict_l2_norms_are_one(self, tmp_path):
        """All output rows should be L2 unit vectors."""
        dim = 64
        mock = _mock_llama_cpp()
        adapter, fake_llm = _make_embed_adapter(tmp_path, mock)
        fake_llm.embed.return_value = [float(i + 1) for i in range(dim)]

        result = adapter.predict(self._make_crops(n=3))
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_predict_embedding_dim_set_after_call(self, tmp_path):
        dim = 256
        mock = _mock_llama_cpp()
        adapter, fake_llm = _make_embed_adapter(tmp_path, mock)
        fake_llm.embed.return_value = [float(i) for i in range(dim)]

        assert adapter.embedding_dim is None
        adapter.predict(self._make_crops(n=1))
        assert adapter.embedding_dim == dim

    def test_predict_empty_crops_returns_zero_rows(self, tmp_path):
        """Empty crop list → (0, ...) shaped array."""
        mock = _mock_llama_cpp()
        adapter, _ = _make_embed_adapter(tmp_path, mock)
        result = adapter.predict([])
        assert result.shape[0] == 0

    def test_predict_zero_vector_does_not_divide_by_zero(self, tmp_path):
        """Zero embedding should not cause NaN or division-by-zero error."""
        dim = 8
        mock = _mock_llama_cpp()
        adapter, fake_llm = _make_embed_adapter(tmp_path, mock)
        fake_llm.embed.return_value = [0.0] * dim

        result = adapter.predict(self._make_crops(n=1))
        assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# EmbedAdapter wrapping
# ---------------------------------------------------------------------------


class TestEmbedAdapterWrapping:
    def test_embed_adapter_wraps_llamacpp_embed_adapter(self, tmp_path):
        """EmbedAdapter(encoder=LlamaCppEmbedAdapter(...)) should not raise."""
        from mata.adapters.embed_adapter import EmbedAdapter

        mock = _mock_llama_cpp()
        adapter, _ = _make_embed_adapter(tmp_path, mock)
        embed = EmbedAdapter(encoder=adapter)
        assert embed is not None
        assert embed._encoder is adapter


# ---------------------------------------------------------------------------
# LlamaCppClassifyAdapter — __init__
# ---------------------------------------------------------------------------


class TestLlamaCppClassifyAdapterInit:
    def test_stores_text_prompts(self, tmp_path):
        mock = _mock_llama_cpp()
        prompts = ["cat", "dog", "bird"]
        adapter, _ = _make_classify_adapter(tmp_path, mock, text_prompts=prompts)
        assert adapter.text_prompts == prompts

    def test_default_text_prompts_is_empty(self, tmp_path):
        mock = _mock_llama_cpp()
        adapter, _ = _make_classify_adapter(tmp_path, mock)
        assert adapter.text_prompts == []

    def test_llama_created_with_embedding_true(self, tmp_path):
        mock = _mock_llama_cpp()
        _make_classify_adapter(tmp_path, mock)
        _, call_kwargs = mock.Llama.call_args
        assert call_kwargs.get("embedding") is True


# ---------------------------------------------------------------------------
# LlamaCppClassifyAdapter — predict()
# ---------------------------------------------------------------------------


class TestLlamaCppClassifyAdapterPredict:
    def _unit_vec(self, dim=8, idx=0):
        v = np.zeros(dim)
        v[idx] = 1.0
        return v.tolist()

    def test_predict_returns_classify_result(self, tmp_path):
        from mata.core.types import ClassifyResult

        mock = _mock_llama_cpp()
        prompts = ["cat", "dog"]
        img_emb = self._unit_vec(8, 0)
        cat_emb = self._unit_vec(8, 0)  # identical → score ≈ 1.0
        dog_emb = self._unit_vec(8, 1)  # orthogonal → score ≈ 0.0

        adapter, fake_llm = _make_classify_adapter(tmp_path, mock, text_prompts=prompts)
        fake_llm.embed.side_effect = [img_emb, cat_emb, dog_emb]

        img_path = _make_pil_image(tmp_path)
        result = adapter.predict(img_path)
        assert isinstance(result, ClassifyResult)
        assert len(result.predictions) == 2

    def test_predict_top1_is_highest_score(self, tmp_path):
        mock = _mock_llama_cpp()
        prompts = ["cat", "dog"]
        img_emb = self._unit_vec(8, 0)
        cat_emb = self._unit_vec(8, 0)  # match → top1
        dog_emb = self._unit_vec(8, 1)

        adapter, fake_llm = _make_classify_adapter(tmp_path, mock, text_prompts=prompts)
        fake_llm.embed.side_effect = [img_emb, cat_emb, dog_emb]

        img_path = _make_pil_image(tmp_path)
        result = adapter.predict(img_path)
        assert result.top1.label_name == "cat"

    def test_predict_all_scores_in_0_1(self, tmp_path):
        """Scores must be clamped to [0.0, 1.0]."""
        mock = _mock_llama_cpp()
        prompts = ["a", "b", "c"]
        # All embeddings random → cosine could be negative; must be clamped
        rng = np.random.default_rng(42)
        embeds = [rng.standard_normal(16).tolist() for _ in range(4)]

        adapter, fake_llm = _make_classify_adapter(tmp_path, mock, text_prompts=prompts)
        fake_llm.embed.side_effect = embeds

        img_path = _make_pil_image(tmp_path)
        result = adapter.predict(img_path)
        for c in result.predictions:
            assert 0.0 <= c.score <= 1.0

    def test_predict_raises_invalid_input_when_no_prompts_anywhere(self, tmp_path):
        """No text_prompts at constructor or call time → InvalidInputError."""
        from mata.core.exceptions import InvalidInputError

        mock = _mock_llama_cpp()
        adapter, _ = _make_classify_adapter(tmp_path, mock)
        img_path = _make_pil_image(tmp_path)
        with pytest.raises(InvalidInputError, match="text_prompts required"):
            adapter.predict(img_path)

    def test_predict_call_time_prompts_override_constructor(self, tmp_path):
        """text_prompts at call time take precedence over constructor prompts."""
        mock = _mock_llama_cpp()
        adapter, fake_llm = _make_classify_adapter(tmp_path, mock, text_prompts=["old"])
        rng = np.random.default_rng(0)
        fake_llm.embed.side_effect = [rng.standard_normal(8).tolist() for _ in range(4)]

        img_path = _make_pil_image(tmp_path)
        result = adapter.predict(img_path, text_prompts=["new_a", "new_b", "new_c"])
        labels = [c.label_name for c in result.predictions]
        assert "new_a" in labels and "new_b" in labels and "new_c" in labels

    def test_predict_meta_contains_model_path_and_backend(self, tmp_path):
        mock = _mock_llama_cpp()
        prompts = ["x"]
        adapter, fake_llm = _make_classify_adapter(tmp_path, mock, text_prompts=prompts)
        rng = np.random.default_rng(1)
        fake_llm.embed.side_effect = [rng.standard_normal(8).tolist() for _ in range(2)]
        img_path = _make_pil_image(tmp_path)
        result = adapter.predict(img_path)
        assert result.meta["backend"] == "llama-cpp-python"
        assert "model_path" in result.meta

    def test_cosine_identical_embeddings_score_approx_one(self, tmp_path):
        """Identical unit vectors → cosine similarity ≈ 1.0."""
        mock = _mock_llama_cpp()
        adapter, fake_llm = _make_classify_adapter(tmp_path, mock, text_prompts=["label"])
        v = self._unit_vec(8, 3)
        fake_llm.embed.side_effect = [v, v]
        img_path = _make_pil_image(tmp_path)
        result = adapter.predict(img_path)
        assert abs(result.predictions[0].score - 1.0) < 1e-4

    def test_cosine_orthogonal_embeddings_score_zero(self, tmp_path):
        """Orthogonal unit vectors → cosine similarity clamped to 0.0."""
        mock = _mock_llama_cpp()
        adapter, fake_llm = _make_classify_adapter(tmp_path, mock, text_prompts=["label"])
        img_v = self._unit_vec(8, 0)
        txt_v = self._unit_vec(8, 1)
        fake_llm.embed.side_effect = [img_v, txt_v]
        img_path = _make_pil_image(tmp_path)
        result = adapter.predict(img_path)
        assert result.predictions[0].score == 0.0
