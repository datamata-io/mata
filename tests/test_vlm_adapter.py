"""Tests for HuggingFaceVLMAdapter (vision-language model inference)."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from mata.adapters.huggingface_vlm_adapter import HuggingFaceVLMAdapter
from mata.core.exceptions import InvalidInputError, ModelLoadError
from mata.core.types import Entity, Instance, VisionResult


@pytest.fixture
def mock_transformers_vlm():
    """Mock transformers library components for VLM."""
    with patch("mata.adapters.huggingface_vlm_adapter._ensure_transformers") as mock_ensure:
        with patch("mata.adapters.huggingface_vlm_adapter.TRANSFORMERS_AVAILABLE", True):
            # Import torch for real tensors
            import torch

            # Create mock classes
            mock_processor_class = Mock()
            mock_model_class = Mock()

            # Mock processor instance
            mock_processor = Mock()

            def apply_chat_template(*args, **kwargs):
                """Return inputs with real tensors that have .to() method."""
                input_ids = torch.randint(0, 100, (1, 50))
                pixel_values = torch.randn(1, 3, 224, 224)
                return MagicMock(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    to=Mock(return_value=MagicMock(input_ids=input_ids, pixel_values=pixel_values)),
                )

            mock_processor.apply_chat_template = Mock(side_effect=apply_chat_template)

            def batch_decode(token_ids, **kwargs):
                """Mock batch_decode to return generated text."""
                return ["A cat sitting on a windowsill looking outside."]

            mock_processor.batch_decode = batch_decode

            # Mock model instance with generate method
            def model_generate(**inputs):
                """Return mock generated token IDs."""
                # Return longer sequence to simulate generation
                return torch.randint(0, 100, (1, 100))

            mock_model = Mock()
            mock_model.generate = model_generate
            mock_model.to = Mock(return_value=mock_model)
            mock_model.eval = Mock(return_value=mock_model)

            # Set up from_pretrained
            mock_processor_class.from_pretrained = Mock(return_value=mock_processor)
            mock_model_class.from_pretrained = Mock(return_value=mock_model)

            # Return dict with mocked classes
            mock_ensure.return_value = {
                "AutoProcessor": mock_processor_class,
                "AutoModelForImageTextToText": mock_model_class,
            }

            yield {
                "ensure": mock_ensure,
                "processor_class": mock_processor_class,
                "model_class": mock_model_class,
                "processor": mock_processor,
                "model": mock_model,
            }


@pytest.fixture
def mock_pytorch_base():
    """Mock PyTorch base adapter functionality."""
    with patch("mata.adapters.pytorch_base._ensure_torch") as mock_ensure:
        # Import real torch for device compatibility
        import torch

        mock_torch = Mock()
        mock_torch.cuda.is_available = Mock(return_value=False)
        # Use real torch.device for tensor.to() compatibility
        mock_torch.device = torch.device
        mock_torch.no_grad = MagicMock()
        mock_ensure.return_value = mock_torch
        yield mock_torch


@pytest.fixture
def sample_image():
    """Create a sample PIL image for testing."""
    return Image.new("RGB", (224, 224), color="blue")


@pytest.fixture
def mock_vlm_adapter(mock_transformers_vlm, mock_pytorch_base):
    """Create a mock VLM adapter instance for testing."""
    vlm = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")
    return vlm


class TestVLMAdapterInitialization:
    """Test VLM adapter initialization."""

    def test_init_default_params(self, mock_transformers_vlm, mock_pytorch_base):
        """Test basic initialization with default parameters."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

        assert adapter.model_id == "Qwen/Qwen3-VL-2B-Instruct"
        assert adapter.max_new_tokens == 512
        assert adapter.temperature == 0.7
        assert adapter.top_p == 0.8
        assert adapter.top_k == 20
        assert adapter.system_prompt is None
        assert adapter.task == "vlm"

    def test_init_custom_system_prompt(self, mock_transformers_vlm, mock_pytorch_base):
        """Test initialization with custom system prompt."""
        system_prompt = "You are a helpful image analyst."
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct", system_prompt=system_prompt)

        assert adapter.system_prompt == system_prompt

    def test_init_custom_generation_params(self, mock_transformers_vlm, mock_pytorch_base):
        """Test initialization with custom generation parameters."""
        adapter = HuggingFaceVLMAdapter(
            "Qwen/Qwen3-VL-2B-Instruct",
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
        )

        assert adapter.max_new_tokens == 256
        assert adapter.temperature == 0.5
        assert adapter.top_p == 0.9
        assert adapter.top_k == 50

    def test_init_model_load_failure(self, mock_pytorch_base):
        """Test initialization when model loading fails."""
        with patch("mata.adapters.huggingface_vlm_adapter._ensure_transformers") as mock_ensure:
            with patch("mata.adapters.huggingface_vlm_adapter.TRANSFORMERS_AVAILABLE", True):
                mock_ensure.return_value = {
                    "AutoProcessor": Mock(from_pretrained=Mock(side_effect=Exception("Network error"))),
                    "AutoModelForImageTextToText": Mock(),
                }

                with pytest.raises(ModelLoadError, match="Network error"):
                    HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

    def test_init_transformers_not_installed(self, mock_pytorch_base):
        """Test initialization when transformers is not available."""
        with patch("mata.adapters.huggingface_vlm_adapter._ensure_transformers") as mock_ensure:
            with patch("mata.adapters.huggingface_vlm_adapter.TRANSFORMERS_AVAILABLE", False):
                mock_ensure.return_value = None

                with pytest.raises(ModelLoadError, match="transformers library not available"):
                    HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")


class TestVLMAdapterInfo:
    """Test VLM adapter info() method."""

    def test_info_returns_correct_metadata(self, mock_transformers_vlm, mock_pytorch_base):
        """Test info() returns all expected metadata fields."""
        adapter = HuggingFaceVLMAdapter(
            "Qwen/Qwen3-VL-2B-Instruct",
            max_new_tokens=256,
            temperature=0.5,
            system_prompt="Test prompt",
        )

        info = adapter.info()

        assert "name" in info
        assert "task" in info
        assert "model_id" in info
        assert "device" in info
        assert "backend" in info
        assert "max_new_tokens" in info
        assert "system_prompt" in info
        assert "temperature" in info
        assert "top_p" in info
        assert "top_k" in info

        assert info["model_id"] == "Qwen/Qwen3-VL-2B-Instruct"
        assert info["max_new_tokens"] == 256
        assert info["temperature"] == 0.5
        assert info["system_prompt"] == "Test prompt"
        assert info["backend"] == "huggingface"

    def test_info_task_is_vlm(self, mock_transformers_vlm, mock_pytorch_base):
        """Test info() task field is 'vlm'."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

        info = adapter.info()

        assert info["task"] == "vlm"


class TestVLMAdapterPredict:
    """Test VLM adapter predict() method."""

    def test_predict_returns_vision_result(self, mock_transformers_vlm, mock_pytorch_base, sample_image):
        """Test predict() returns VisionResult type."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

        result = adapter.predict(sample_image, prompt="Describe this image.")

        assert isinstance(result, VisionResult)

    def test_predict_text_field_populated(self, mock_transformers_vlm, mock_pytorch_base, sample_image):
        """Test predict() returns result with text field populated."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

        result = adapter.predict(sample_image, prompt="What do you see?")

        assert result.text is not None
        assert len(result.text) > 0
        assert isinstance(result.text, str)

    def test_predict_prompt_stored_in_result(self, mock_transformers_vlm, mock_pytorch_base, sample_image):
        """Test predict() stores the prompt in the result."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")
        prompt = "What animal is this?"

        result = adapter.predict(sample_image, prompt=prompt)

        assert result.prompt == prompt

    def test_predict_instances_empty(self, mock_transformers_vlm, mock_pytorch_base, sample_image):
        """Test predict() returns empty instances list (raw text mode)."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

        result = adapter.predict(sample_image, prompt="Describe this.")

        assert result.instances == []
        assert isinstance(result.instances, list)

    def test_predict_no_prompt_raises_error(self, mock_transformers_vlm, mock_pytorch_base, sample_image):
        """Test predict() raises InvalidInputError when prompt is None."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

        with pytest.raises(InvalidInputError, match="prompt is required"):
            adapter.predict(sample_image, prompt=None)

    def test_predict_empty_prompt_raises_error(self, mock_transformers_vlm, mock_pytorch_base, sample_image):
        """Test predict() raises InvalidInputError when prompt is empty string."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

        with pytest.raises(InvalidInputError, match="prompt is required"):
            adapter.predict(sample_image, prompt="")

    def test_predict_with_system_prompt(self, mock_transformers_vlm, mock_pytorch_base, sample_image):
        """Test predict() with system prompt in constructor."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct", system_prompt="You are a helpful assistant.")

        adapter.predict(sample_image, prompt="What is this?")

        # Verify processor.apply_chat_template was called with messages including system prompt
        mock_apply = adapter.processor.apply_chat_template
        assert mock_apply.called
        call_args = mock_apply.call_args
        messages = call_args[0][0]
        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_predict_system_prompt_override(self, mock_transformers_vlm, mock_pytorch_base, sample_image):
        """Test predict() with system prompt override."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct", system_prompt="Default system prompt")

        adapter.predict(sample_image, prompt="What is this?", system_prompt="Override prompt")

        # Verify the override prompt was used
        mock_apply = adapter.processor.apply_chat_template
        call_args = mock_apply.call_args
        messages = call_args[0][0]
        assert messages[0]["content"][0]["text"] == "Override prompt"

    def test_predict_max_new_tokens_override(self, mock_transformers_vlm, mock_pytorch_base, sample_image):
        """Test predict() max_new_tokens parameter override."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct", max_new_tokens=512)

        result = adapter.predict(sample_image, prompt="Describe this.", max_new_tokens=100)

        # Verify model.generate was called with overridden max_new_tokens
        # We can't directly check the call args easily with our mock, but it should succeed
        assert result.meta["max_new_tokens"] == 100

    def test_predict_with_pil_image(self, mock_transformers_vlm, mock_pytorch_base):
        """Test predict() with PIL Image input."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")
        pil_image = Image.new("RGB", (100, 100), color="red")

        result = adapter.predict(pil_image, prompt="What color is this?")

        assert isinstance(result, VisionResult)
        assert result.text is not None

    def test_predict_with_file_path(self, mock_transformers_vlm, mock_pytorch_base, tmp_path):
        """Test predict() with file path input."""
        # Create a temporary image file
        image_path = tmp_path / "test_image.jpg"
        img = Image.new("RGB", (100, 100), color="green")
        img.save(image_path)

        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

        result = adapter.predict(str(image_path), prompt="What is this?")

        assert isinstance(result, VisionResult)
        assert result.text is not None
        assert result.meta["image_path"] == str(image_path)

    def test_predict_meta_contains_model_info(self, mock_transformers_vlm, mock_pytorch_base, sample_image):
        """Test predict() result.meta contains model information."""
        adapter = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

        result = adapter.predict(sample_image, prompt="Describe this.")

        assert "model_id" in result.meta
        assert "device" in result.meta
        assert "backend" in result.meta
        assert "max_new_tokens" in result.meta
        assert "tokens_generated" in result.meta

        assert result.meta["model_id"] == "Qwen/Qwen3-VL-2B-Instruct"
        assert result.meta["backend"] == "huggingface"


class TestVLMAdapterMultiImage:
    """Test multi-image support in VLM adapter (Task 2.2)."""

    def test_predict_single_image_unchanged(self, mock_vlm_adapter, sample_image):
        """Test single image call (existing API) works unchanged."""
        vlm = mock_vlm_adapter
        result = vlm.predict(sample_image, prompt="What is this?")

        assert isinstance(result, VisionResult)
        assert result.text is not None
        assert result.meta["image_count"] == 1
        # PIL image has no path, so image_paths should be empty
        assert result.meta["image_paths"] == []
        assert result.meta["image_path"] is None  # No path for PIL image

    def test_predict_with_images_kwarg(self, mock_vlm_adapter, sample_image, tmp_path):
        """Test images kwarg adds additional images to message content."""
        vlm = mock_vlm_adapter

        # Create multiple test images
        img1_path = tmp_path / "img1.jpg"
        img2_path = tmp_path / "img2.jpg"
        sample_image.save(img1_path)
        sample_image.save(img2_path)

        result = vlm.predict(str(img1_path), images=[str(img2_path)], prompt="Compare these images.")

        assert isinstance(result, VisionResult)
        assert result.meta["image_count"] == 2
        assert len(result.meta["image_paths"]) == 2
        assert str(img1_path) in result.meta["image_paths"]
        assert str(img2_path) in result.meta["image_paths"]
        assert result.meta["image_path"] == str(img1_path)  # First path for backward compat

    def test_predict_image_and_images_combined(self, mock_vlm_adapter, sample_image, tmp_path):
        """Test primary image + additional images are all merged."""
        vlm = mock_vlm_adapter

        # Create test images
        primary_path = tmp_path / "primary.jpg"
        ref1_path = tmp_path / "ref1.jpg"
        ref2_path = tmp_path / "ref2.jpg"
        sample_image.save(primary_path)
        sample_image.save(ref1_path)
        sample_image.save(ref2_path)

        result = vlm.predict(
            str(primary_path), images=[str(ref1_path), str(ref2_path)], prompt="Compare these three images."
        )

        assert isinstance(result, VisionResult)
        assert result.meta["image_count"] == 3
        assert len(result.meta["image_paths"]) == 3
        # Primary image should be first
        assert result.meta["image_paths"][0] == str(primary_path)
        assert result.meta["image_path"] == str(primary_path)

    def test_predict_images_only_no_primary(self, mock_vlm_adapter, sample_image, tmp_path):
        """Test images-only mode (image=None, images=[...]) works correctly."""
        vlm = mock_vlm_adapter

        # Create test images
        img1_path = tmp_path / "img1.jpg"
        img2_path = tmp_path / "img2.jpg"
        sample_image.save(img1_path)
        sample_image.save(img2_path)

        result = vlm.predict(
            image=None, images=[str(img1_path), str(img2_path)], prompt="What's different between these images?"
        )

        assert isinstance(result, VisionResult)
        assert result.meta["image_count"] == 2
        assert len(result.meta["image_paths"]) == 2
        assert result.meta["image_path"] == str(img1_path)  # First from images list

    def test_predict_no_image_no_images_raises(self, mock_vlm_adapter):
        """Test neither image nor images raises InvalidInputError."""
        vlm = mock_vlm_adapter

        with pytest.raises(InvalidInputError) as exc_info:
            vlm.predict(image=None, images=None, prompt="Describe this")

        assert "At least one image is required" in str(exc_info.value)

    def test_predict_images_empty_list_raises(self, mock_vlm_adapter):
        """Test empty images list without primary image raises error."""
        vlm = mock_vlm_adapter

        with pytest.raises(InvalidInputError) as exc_info:
            vlm.predict(image=None, images=[], prompt="Describe this")

        # Empty list means _load_images is called and raises
        assert "images list cannot be empty" in str(exc_info.value)

    def test_predict_multi_image_mixed_types(self, mock_vlm_adapter, sample_image, tmp_path):
        """Test mixing file paths and PIL images in multi-image."""
        vlm = mock_vlm_adapter

        # Create one file, use one PIL image
        img_path = tmp_path / "file.jpg"
        sample_image.save(img_path)

        result = vlm.predict(str(img_path), images=[sample_image], prompt="Compare these.")  # PIL image

        assert isinstance(result, VisionResult)
        assert result.meta["image_count"] == 2
        # File path + PIL (no path) = 1 path total
        assert len(result.meta["image_paths"]) == 1
        assert str(img_path) in result.meta["image_paths"]

    def test_predict_multi_image_meta_backward_compat(self, mock_vlm_adapter, sample_image, tmp_path):
        """Test result.meta["image_path"] maintains backward compatibility."""
        vlm = mock_vlm_adapter

        img1_path = tmp_path / "img1.jpg"
        img2_path = tmp_path / "img2.jpg"
        sample_image.save(img1_path)
        sample_image.save(img2_path)

        result = vlm.predict(str(img1_path), images=[str(img2_path)], prompt="Compare.")

        # image_path should be first path (backward compat)
        assert result.meta["image_path"] == str(img1_path)
        # image_paths should have all paths
        assert result.meta["image_paths"] == [str(img1_path), str(img2_path)]

    def test_predict_multi_image_no_paths(self, mock_vlm_adapter, sample_image):
        """Test multi-image with PIL images (no paths) sets image_path to None."""
        vlm = mock_vlm_adapter

        result = vlm.predict(sample_image, images=[sample_image], prompt="Compare.")

        assert result.meta["image_count"] == 2
        assert result.meta["image_paths"] == []  # No paths from PIL images
        assert result.meta["image_path"] is None  # Backward compat

    def test_predict_multi_image_with_output_mode(self, mock_vlm_adapter, sample_image, tmp_path):
        """Test multi-image works with output_mode (structured output)."""
        vlm = mock_vlm_adapter

        img1_path = tmp_path / "img1.jpg"
        img2_path = tmp_path / "img2.jpg"
        sample_image.save(img1_path)
        sample_image.save(img2_path)

        result = vlm.predict(
            str(img1_path), images=[str(img2_path)], prompt="List objects in both images.", output_mode="detect"
        )

        assert isinstance(result, VisionResult)
        assert result.meta["image_count"] == 2
        assert result.meta["output_mode"] == "detect"
        # Entities may or may not be populated depending on mock output


class TestEntityDataclass:
    """Test Entity dataclass for structured VLM output."""

    def test_entity_creation_defaults(self):
        """Test Entity(label="cat") has score=1.0, attributes={}."""
        entity = Entity(label="cat")

        assert entity.label == "cat"
        assert entity.score == 1.0
        assert entity.attributes == {}

    def test_entity_creation_with_score(self):
        """Test Entity(label="dog", score=0.85) stores correctly."""
        entity = Entity(label="dog", score=0.85)

        assert entity.label == "dog"
        assert entity.score == 0.85
        assert entity.attributes == {}

    def test_entity_creation_with_attributes(self):
        """Test attributes dict is stored correctly."""
        attrs = {"color": "brown", "size": "large", "count": 2}
        entity = Entity(label="dog", score=0.9, attributes=attrs)

        assert entity.label == "dog"
        assert entity.score == 0.9
        assert entity.attributes == attrs
        assert entity.attributes["color"] == "brown"
        assert entity.attributes["count"] == 2

    def test_entity_to_dict(self):
        """Test serialization produces correct dict."""
        entity = Entity(label="cat", score=0.95, attributes={"color": "orange"})
        entity_dict = entity.to_dict()

        assert entity_dict == {"label": "cat", "score": 0.95, "attributes": {"color": "orange"}}

    def test_entity_from_dict(self):
        """Test deserialization restores Entity."""
        data = {"label": "bird", "score": 0.8, "attributes": {"species": "robin"}}
        entity = Entity.from_dict(data)

        assert entity.label == "bird"
        assert entity.score == 0.8
        assert entity.attributes == {"species": "robin"}

    def test_entity_from_dict_missing_score(self):
        """Test from_dict defaults to score=1.0 when missing."""
        data = {"label": "person"}
        entity = Entity.from_dict(data)

        assert entity.label == "person"
        assert entity.score == 1.0
        assert entity.attributes == {}

    def test_entity_promote_with_bbox(self):
        """Test promote() with bbox returns valid Instance."""
        entity = Entity(label="cat", score=0.9, attributes={"color": "orange"})
        bbox = (10.0, 20.0, 100.0, 150.0)

        instance = entity.promote(bbox=bbox)

        assert isinstance(instance, Instance)
        assert instance.bbox == bbox
        assert instance.label_name == "cat"
        assert instance.score == 0.9
        assert instance.label == 0  # Default label_id

    def test_entity_promote_with_mask(self):
        """Test promote() with mask returns valid Instance."""
        entity = Entity(label="dog", score=0.85)
        mask_rle = {"size": [100, 100], "counts": b"fake_rle_data"}

        instance = entity.promote(mask=mask_rle)

        assert isinstance(instance, Instance)
        assert instance.mask == mask_rle
        assert instance.label_name == "dog"
        assert instance.score == 0.85

    def test_entity_promote_no_spatial_raises(self):
        """Test promote() without bbox or mask raises ValueError (from Instance)."""
        entity = Entity(label="car", score=0.7)

        with pytest.raises(ValueError) as exc_info:
            entity.promote()

        # Error comes from Instance.__post_init__ validation
        assert "must have at least one of: bbox, mask" in str(exc_info.value)


class TestVisionResultWithEntities:
    """Test VisionResult with entities field."""

    def test_vision_result_entities_default_empty(self):
        """Test VisionResult(instances=[]) has entities=[] by default."""
        result = VisionResult(instances=[])

        assert result.entities == []
        assert isinstance(result.entities, list)

    def test_vision_result_with_entities(self):
        """Test entities are stored and accessible."""
        entities = [
            Entity(label="cat", score=0.9),
            Entity(label="dog", score=0.85, attributes={"color": "brown"}),
        ]
        result = VisionResult(instances=[], entities=entities)

        assert len(result.entities) == 2
        assert result.entities[0].label == "cat"
        assert result.entities[1].label == "dog"
        assert result.entities[1].attributes["color"] == "brown"

    def test_vision_result_to_dict_includes_entities(self):
        """Test serialization includes entities field."""
        entities = [Entity(label="person", score=0.95)]
        result = VisionResult(instances=[], entities=entities, text="A person walking")

        result_dict = result.to_dict()

        assert "entities" in result_dict
        assert len(result_dict["entities"]) == 1
        assert result_dict["entities"][0]["label"] == "person"
        assert result_dict["entities"][0]["score"] == 0.95

    def test_vision_result_from_dict_with_entities(self):
        """Test deserialization restores entities."""
        data = {
            "instances": [],
            "entities": [
                {"label": "cat", "score": 0.9, "attributes": {}},
                {"label": "dog", "score": 0.8, "attributes": {"color": "brown"}},
            ],
            "meta": {},
            "text": "A cat and a dog",
            "prompt": "List objects",
        }
        result = VisionResult.from_dict(data)

        assert len(result.entities) == 2
        assert result.entities[0].label == "cat"
        assert result.entities[0].score == 0.9
        assert result.entities[1].label == "dog"
        assert result.entities[1].attributes["color"] == "brown"

    def test_vision_result_filter_by_score_filters_entities(self):
        """Test filter_by_score() filters both instances and entities."""
        # Create some entities with different scores
        entities = [
            Entity(label="high_conf", score=0.9),
            Entity(label="low_conf", score=0.4),
            Entity(label="medium_conf", score=0.6),
        ]
        result = VisionResult(instances=[], entities=entities)

        # Filter with threshold 0.5
        filtered = result.filter_by_score(0.5)

        assert len(filtered.entities) == 2
        assert filtered.entities[0].label == "high_conf"
        assert filtered.entities[1].label == "medium_conf"
        # low_conf (0.4) should be filtered out


class TestVLMPredictWithOutputMode:
    """Test VLM predict() with output_mode for structured output."""

    def test_predict_output_mode_none_default(self, mock_vlm_adapter, sample_image, tmp_path):
        """Test no output_mode returns entities=[] (existing behavior)."""
        vlm = mock_vlm_adapter

        img_path = tmp_path / "test.jpg"
        sample_image.save(img_path)

        result = vlm.predict(str(img_path), prompt="Describe this image")

        assert isinstance(result, VisionResult)
        assert result.entities == []
        assert result.text is not None
        assert "output_mode" not in result.meta or result.meta.get("output_mode") is None

    def test_predict_output_mode_detect(self, mock_transformers_vlm, mock_pytorch_base, sample_image, tmp_path):
        """Test output_mode='detect' with mocked JSON response populates entities."""
        # Mock the model to return JSON output
        mock_transformers_vlm["processor"].batch_decode = Mock(
            return_value=['```json\n[{"label": "cat", "confidence": 0.95}, {"label": "dog", "confidence": 0.8}]\n```']
        )

        vlm = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

        img_path = tmp_path / "test.jpg"
        sample_image.save(img_path)

        result = vlm.predict(str(img_path), prompt="List objects", output_mode="detect")

        assert isinstance(result, VisionResult)
        assert len(result.entities) == 2
        assert result.entities[0].label == "cat"
        assert result.entities[0].score == 0.95
        assert result.entities[1].label == "dog"
        assert result.entities[1].score == 0.8
        assert result.meta["output_mode"] == "detect"

    def test_predict_output_mode_malformed_json(self, mock_transformers_vlm, mock_pytorch_base, sample_image, tmp_path):
        """Test malformed JSON degrades gracefully (warning, raw text, entities=[])."""
        # Mock the model to return malformed JSON
        mock_transformers_vlm["processor"].batch_decode = Mock(return_value=["This is not valid JSON {broken"])

        vlm = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

        img_path = tmp_path / "test.jpg"
        sample_image.save(img_path)

        result = vlm.predict(str(img_path), prompt="List objects", output_mode="detect")

        # Should gracefully degrade
        assert isinstance(result, VisionResult)
        assert result.entities == []  # Empty on parsing failure
        assert result.text == "This is not valid JSON {broken"  # Raw text preserved
        assert result.meta["output_mode"] == "detect"

    def test_predict_output_mode_in_meta(self, mock_vlm_adapter, sample_image, tmp_path):
        """Test result.meta["output_mode"] is set when output_mode is provided."""
        vlm = mock_vlm_adapter

        img_path = tmp_path / "test.jpg"
        sample_image.save(img_path)

        result = vlm.predict(str(img_path), prompt="Describe", output_mode="json")

        assert result.meta["output_mode"] == "json"


class TestVLMDoSampleFix:
    """Test do_sample parameter behavior with temperature."""

    def test_predict_temperature_zero_greedy(self, mock_transformers_vlm, mock_pytorch_base, sample_image, tmp_path):
        """Test temperature=0 results in do_sample=False for greedy decoding."""
        mock_model = mock_transformers_vlm["model"]

        # Create a spy to check generate() call arguments
        original_generate = mock_model.generate
        generate_calls = []

        def generate_spy(**kwargs):
            generate_calls.append(kwargs)
            return original_generate(**kwargs)

        mock_model.generate = generate_spy

        vlm = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")

        img_path = tmp_path / "test.jpg"
        sample_image.save(img_path)

        # Call with temperature=0 (should use greedy decoding)
        result = vlm.predict(str(img_path), prompt="Describe", temperature=0.0)

        assert isinstance(result, VisionResult)
        assert len(generate_calls) == 1

        # Verify do_sample was False
        gen_kwargs = generate_calls[0]
        assert gen_kwargs.get("do_sample", True) is False
        # Temperature, top_p, top_k should NOT be passed when do_sample=False
        assert "temperature" not in gen_kwargs
        assert "top_p" not in gen_kwargs
        assert "top_k" not in gen_kwargs


class TestVLMArchitectureDetection:
    """Test VLM architecture detection."""

    def test_is_vlm_model_qwen(self):
        """Test detection of Qwen-VL models."""
        assert HuggingFaceVLMAdapter._is_vlm_model("Qwen/Qwen3-VL-2B-Instruct") is True
        assert HuggingFaceVLMAdapter._is_vlm_model("Qwen/Qwen2-VL-7B") is True
        assert HuggingFaceVLMAdapter._is_vlm_model("qwen-vl-chat") is True

    def test_is_vlm_model_llava(self):
        """Test detection of LLaVA models."""
        assert HuggingFaceVLMAdapter._is_vlm_model("llava-hf/llava-1.5-7b-hf") is True
        assert HuggingFaceVLMAdapter._is_vlm_model("LLaVA-v1.6-34B") is True

    def test_is_vlm_model_internvl(self):
        """Test detection of InternVL models."""
        assert HuggingFaceVLMAdapter._is_vlm_model("OpenGVLab/InternVL2-1B") is True
        assert HuggingFaceVLMAdapter._is_vlm_model("internvl-chat-2b") is True

    def test_is_vlm_model_non_vlm(self):
        """Test non-VLM models return False."""
        assert HuggingFaceVLMAdapter._is_vlm_model("facebook/detr-resnet-50") is False
        assert HuggingFaceVLMAdapter._is_vlm_model("microsoft/resnet-50") is False

    def test_is_vlm_model_clip_not_vlm(self):
        """Test CLIP models are not detected as VLM."""
        assert HuggingFaceVLMAdapter._is_vlm_model("openai/clip-vit-base-patch32") is False
        assert HuggingFaceVLMAdapter._is_vlm_model("clip-vit-large-patch14") is False


class TestVLMLoaderIntegration:
    """Test VLM integration with UniversalLoader and API."""

    def test_universal_loader_vlm_task(self, mock_transformers_vlm, mock_pytorch_base):
        """Test UniversalLoader routes 'vlm' task to HuggingFaceVLMAdapter (auto-wrapped)."""
        from mata.adapters.wrappers.vlm_wrapper import VLMWrapper
        from mata.core.model_loader import UniversalLoader

        loader = UniversalLoader()
        adapter = loader.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")

        # mata.load("vlm", ...) now auto-wraps with VLMWrapper
        assert isinstance(adapter, VLMWrapper)
        assert isinstance(adapter.adapter, HuggingFaceVLMAdapter)
        assert adapter.task == "vlm"  # Delegated via __getattr__
        assert hasattr(adapter, "query")  # Graph node protocol

    def test_api_run_vlm_dispatches_correctly(self, mock_transformers_vlm, mock_pytorch_base, sample_image, tmp_path):
        """Test mata.run('vlm', ...) dispatches correctly."""
        # Create a temporary image file
        image_path = tmp_path / "test_image.jpg"
        sample_image.save(image_path)

        import mata

        result = mata.run("vlm", str(image_path), model="Qwen/Qwen3-VL-2B-Instruct", prompt="What is this?")

        assert isinstance(result, VisionResult)
        assert result.text is not None
        assert result.prompt == "What is this?"

    def test_api_run_vlm_with_images_parameter(self, mock_transformers_vlm, mock_pytorch_base, sample_image, tmp_path):
        """Test mata.run() with images parameter flows correctly to predict()."""
        # Create multiple image files
        img1_path = tmp_path / "img1.jpg"
        img2_path = tmp_path / "img2.jpg"
        sample_image.save(img1_path)
        sample_image.save(img2_path)

        import mata

        # Test end-to-end: mata.run() → load() → predict() with images parameter
        result = mata.run(
            "vlm",
            str(img1_path),
            model="Qwen/Qwen3-VL-2B-Instruct",
            images=[str(img2_path)],
            prompt="Compare these images.",
        )

        assert isinstance(result, VisionResult)
        assert result.text is not None
        assert result.prompt == "Compare these images."
        # Verify multi-image metadata
        assert result.meta["image_count"] == 2
        assert len(result.meta["image_paths"]) == 2
        assert str(img1_path) in result.meta["image_paths"]
        assert str(img2_path) in result.meta["image_paths"]

    def test_api_run_vlm_images_only_mode(self, mock_transformers_vlm, mock_pytorch_base, sample_image, tmp_path):
        """Test mata.run() works with images-only mode (no primary image via positional arg)."""
        # Create image files
        img1_path = tmp_path / "img1.jpg"
        img2_path = tmp_path / "img2.jpg"
        sample_image.save(img1_path)
        sample_image.save(img2_path)

        import mata

        # Note: mata.run() requires `input` as positional arg, so we pass first image there
        # This test verifies that additional images via images=[] parameter work correctly
        result = mata.run(
            "vlm",
            str(img1_path),
            model="Qwen/Qwen3-VL-2B-Instruct",
            images=[str(img2_path)],
            prompt="What's in these images?",
        )

        assert isinstance(result, VisionResult)
        assert result.meta["image_count"] == 2


@pytest.mark.slow
class TestVLMAdapterIntegration:
    """Integration tests requiring Qwen3-VL-2B model download (~4GB)."""

    def test_load_qwen3_vl_2b(self):
        """Test mata.load() with real Qwen3-VL model."""
        import mata

        adapter = mata.load("vlm", "Qwen/Qwen3-VL-2B-Instruct")
        # mata.load("vlm", ...) returns VLMWrapper (auto-wraps for graph system)
        from mata.adapters.wrappers.vlm_wrapper import VLMWrapper

        assert isinstance(adapter, VLMWrapper)
        assert hasattr(adapter, "query")  # Graph node protocol
        assert hasattr(adapter, "predict")  # Delegated from inner adapter
        assert adapter.info()["task"] == "vlm"

    def test_run_describe_image(self, sample_image, tmp_path):
        """Test mata.run() with real image and model."""
        import mata

        # Create a test image file
        image_path = tmp_path / "test_image.jpg"
        sample_image.save(image_path)

        result = mata.run(
            "vlm",
            str(image_path),
            model="Qwen/Qwen3-VL-2B-Instruct",
            prompt="Describe this image in one sentence.",
            max_new_tokens=100,
        )
        assert isinstance(result, VisionResult)
        assert result.text is not None
        assert len(result.text) > 10  # Non-trivial response

    def test_run_with_system_prompt(self, sample_image, tmp_path):
        """Test system prompt affects output."""
        import mata

        # Create a test image file
        image_path = tmp_path / "test_image.jpg"
        sample_image.save(image_path)

        result = mata.run(
            "vlm",
            str(image_path),
            model="Qwen/Qwen3-VL-2B-Instruct",
            prompt="What do you see?",
            system_prompt="You are a concise image analyst. Reply in exactly 5 words.",
            max_new_tokens=50,
        )
        assert isinstance(result, VisionResult)
        assert result.text is not None

    def test_result_serialization(self, sample_image, tmp_path):
        """Test VisionResult with text can be serialized."""
        import mata

        # Create a test image file
        image_path = tmp_path / "test_image.jpg"
        sample_image.save(image_path)

        result = mata.run(
            "vlm",
            str(image_path),
            model="Qwen/Qwen3-VL-2B-Instruct",
            prompt="What is this?",
            max_new_tokens=50,
        )
        json_str = result.to_json()
        assert "text" in json_str


@pytest.mark.slow
class TestVLMAdapterV154Integration:
    """Integration tests for v1.5.4 features (structured output + multi-image)."""

    def test_structured_output_detect_mode(self, sample_image, tmp_path):
        """Test output_mode='detect' with real Qwen3-VL model."""
        import mata

        # Create a test image file
        image_path = tmp_path / "test_image.jpg"
        sample_image.save(image_path)

        result = mata.run(
            "vlm",
            str(image_path),
            model="Qwen/Qwen3-VL-2B-Instruct",
            prompt="List all objects you see in this image.",
            output_mode="detect",
            max_new_tokens=200,
        )
        assert isinstance(result, VisionResult)
        assert result.text is not None
        # entities may or may not parse depending on model output format
        assert isinstance(result.entities, list)
        assert result.meta.get("output_mode") == "detect"

    def test_multi_image_comparison(self, sample_image, tmp_path):
        """Test multi-image with real Qwen3-VL model."""
        import mata

        # Create test image files
        image_path1 = tmp_path / "test_image1.jpg"
        image_path2 = tmp_path / "test_image2.jpg"
        sample_image.save(image_path1)
        sample_image.save(image_path2)  # same image for testing

        result = mata.run(
            "vlm",
            str(image_path1),
            model="Qwen/Qwen3-VL-2B-Instruct",
            images=[str(image_path2)],
            prompt="Compare these two images.",
            max_new_tokens=100,
        )
        assert isinstance(result, VisionResult)
        assert result.text is not None
        assert result.meta.get("image_count") == 2
