"""HuggingFace Vision-Language Model adapter for MATA framework.

Supports vision-language models (VLMs) that combine image understanding with
natural language processing. Enables tasks like image captioning, visual
question answering (VQA), and image-based conversation using chat-style APIs.

Key Features:
- Chat-based API with system prompt support
- Configurable generation parameters (temperature, top_p, top_k)
- Multi-model support: Qwen3-VL, MedGemma, LFM2-VL, LLaVA, InternVL, and more
- Configurable dtype and trust_remote_code for model-specific requirements
- Multi-turn conversation ready (future enhancement)
- Streaming support ready (future enhancement)

Supported Models:
- Qwen/Qwen3-VL-2B-Instruct (recommended for dev/testing)
- google/gemma-4-E2B-it (multimodal, native aspect ratio; requires transformers>=5.5.0)
- google/medgemma-1.5-4b-it (medical imaging, requires dtype="bfloat16")
- LiquidAI/LFM2.5-VL-1.6B (lightweight, requires dtype="bfloat16")
- HuggingFaceTB/SmolVLM-256M-Instruct (ultra-light edge)
- microsoft/Florence-2-large (grounding/captioning, requires trust_remote_code=True)
- google/paligemma2-3b-pt-224 (document understanding)
- llava-hf/llava-v1.6-mistral-7b-hf (high-quality VQA)
- OpenGVLab/InternVL2-1B (multilingual, requires trust_remote_code=True)
- microsoft/Phi-3.5-vision-instruct (code/diagrams, requires trust_remote_code=True)
- vikhyatk/moondream2 (tiny/fast, requires trust_remote_code=True)

Note: Requires transformers >= 5.0.0 for full multi-model support
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mata.core.exceptions import InvalidInputError, ModelLoadError
from mata.core.logging import get_logger
from mata.core.parsers import extract_json_from_text, get_json_schema, parse_entities
from mata.core.types import BBox, Instance, VisionResult

from .pytorch_base import PyTorchBaseAdapter

logger = get_logger(__name__)

# Type alias for image inputs
ImageInput = str | Path | Image.Image | np.ndarray

# Lazy imports for transformers-specific modules
_transformers = None
TRANSFORMERS_AVAILABLE = None


def _ensure_transformers():
    """Ensure transformers library is imported (lazy loading).

    Attempts to import AutoModelForImageTextToText (transformers >= 4.51.0).
    Falls back to model-specific classes for older versions.

    Returns:
        Dictionary of transformers classes if available

    Raises:
        ImportError: If transformers cannot be imported
    """
    global _transformers, TRANSFORMERS_AVAILABLE
    if _transformers is None:
        try:
            from transformers import AutoProcessor

            # Try the newer AutoModelForImageTextToText class first
            try:
                from transformers import AutoModelForImageTextToText

                _transformers = {
                    "AutoProcessor": AutoProcessor,
                    "AutoModelForImageTextToText": AutoModelForImageTextToText,
                }
                logger.debug("Using AutoModelForImageTextToText for VLM models")
            except ImportError:
                # Fallback for older transformers versions
                logger.warning(
                    "AutoModelForImageTextToText not available. "
                    "Please upgrade transformers: pip install -U transformers"
                )
                # For older versions, we'll use AutoModel and let it auto-detect
                from transformers import AutoModel

                _transformers = {
                    "AutoProcessor": AutoProcessor,
                    "AutoModelForImageTextToText": AutoModel,  # Fallback
                }

            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
    return _transformers


def _load_processor_patched(auto_processor_cls, model_id: str, trust_remote_code: bool):
    """Load a processor after patching TokenizersBackend for legacy compat.

    Some custom processor code (e.g. Florence-2) accesses
    ``tokenizer.additional_special_tokens`` which was removed in
    transformers >= 5.0.  This helper temporarily adds the attribute so the
    processor __init__ can proceed.
    """
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
    except ImportError:
        # transformers < 5.0 — no TokenizersBackend to patch.
        return auto_processor_cls.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

    _orig_getattr = TokenizersBackend.__getattr__

    def _compat_getattr(self, key):
        if key == "additional_special_tokens":
            return list(getattr(self, "all_special_tokens", []))
        return _orig_getattr(self, key)

    TokenizersBackend.__getattr__ = _compat_getattr
    try:
        return auto_processor_cls.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
    finally:
        TokenizersBackend.__getattr__ = _orig_getattr


# Mapping of model IDs whose custom code (trust_remote_code) is incompatible
# with the installed transformers version → community-converted versions that
# use native transformers classes.  Only applied when the original repo fails.
_MODEL_COMPAT_REDIRECTS: dict[str, str] = {
    "microsoft/Florence-2-large": "florence-community/Florence-2-large",
    "microsoft/Florence-2-base": "florence-community/Florence-2-base",
    "microsoft/Florence-2-large-ft": "florence-community/Florence-2-large-ft",
    "microsoft/Florence-2-base-ft": "florence-community/Florence-2-base-ft",
}


class HuggingFaceVLMAdapter(PyTorchBaseAdapter):
    """Vision-Language Model adapter using HuggingFace transformers.

    Wraps HuggingFace VLM models to provide a unified interface for image
    understanding tasks. Uses chat-based API (apply_chat_template) for
    natural language interaction with images.

    Supports both simple one-shot queries and more complex prompting scenarios
    with system prompts and custom generation parameters.

    Attributes:
        task: Always "vlm"
        model_id: HuggingFace model identifier
        model: VLM model instance
        processor: Processor for image and text preprocessing
        max_new_tokens: Maximum number of tokens to generate
        system_prompt: Default system prompt for all predictions
        temperature: Sampling temperature for generation
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter

    Examples:
        >>> # Basic image description
        >>> vlm = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")
        >>> result = vlm.predict("image.jpg", prompt="Describe this image.")
        >>> print(result.text)
        "A cat sitting on a windowsill..."

        >>> # Visual question answering with custom generation params
        >>> result = vlm.predict(
        ...     "image.jpg",
        ...     prompt="How many people are visible?",
        ...     max_new_tokens=50,
        ...     temperature=0.3
        ... )
        >>> print(result.text)
        "There are 3 people visible in the image."

        >>> # With system prompt for domain-specific behavior
        >>> vlm = HuggingFaceVLMAdapter(
        ...     "Qwen/Qwen3-VL-2B-Instruct",
        ...     system_prompt="You are a quality inspector. Identify defects concisely."
        ... )
        >>> result = vlm.predict("product.jpg", prompt="Analyze this product.")
    """

    task = "vlm"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """Initialize VLM adapter with model and generation parameters.

        Args:
            model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-VL-2B-Instruct")
            device: Device for inference ("auto", "cpu", "cuda", or specific device)
            max_new_tokens: Maximum tokens to generate (default: 512)
            system_prompt: Optional default system prompt for all predictions
            temperature: Sampling temperature (default: 0.7, higher = more random)
            top_p: Nucleus sampling probability (default: 0.8)
            top_k: Top-k sampling parameter (default: 20)
            dtype: Torch dtype for model loading ("auto", "bfloat16", "float16",
                "float32", or a torch.dtype object). Default "auto" lets
                Transformers select the optimal dtype.
            trust_remote_code: Allow executing custom model code from the
                HuggingFace Hub. Required by InternVL, CogVLM, Phi-Vision,
                Moondream2, Florence-2. Default False for security — must be
                explicitly opted in.
            **kwargs: Additional arguments passed to parent class

        Raises:
            ModelLoadError: If model loading fails
            ImportError: If transformers library not available
        """
        # Ensure transformers is available
        transformers_lib = _ensure_transformers()
        if not TRANSFORMERS_AVAILABLE:
            raise ModelLoadError(
                model_id,
                "transformers library not available. " "Install with: pip install transformers>=4.51.0",
            )

        # Initialize parent class (sets up device, torch lazy loading)
        super().__init__(device=device, **kwargs)

        # Store adapter configuration
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        # Resolve possible compatibility redirect (e.g. microsoft/Florence-2-large
        # → florence-community/Florence-2-large for transformers >= 5.0).
        # The redirect is stored so predict() forward-calls use the same ID.
        self._effective_model_id = model_id

        # Load model and processor
        try:
            import time

            from mata.core.logging import is_model_cached, suppress_third_party_logs

            t0 = time.perf_counter()
            _cached = is_model_cached(model_id)
            if not _cached:
                logger.info(f"Downloading {model_id} (first run — this may take a minute)...")
            logger.info(f"Loading VLM model: {model_id}")

            with suppress_third_party_logs(allow_progress=not _cached):

                # Load processor — fall back to AutoTokenizer for models
                # (e.g. moondream3) that don't ship standard processor files.
                AutoProcessor = transformers_lib["AutoProcessor"]  # noqa: N806
                try:
                    self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=self.trust_remote_code)
                except AttributeError as ae:
                    if "additional_special_tokens" not in str(ae):
                        raise
                    # transformers >=5.0 compat: TokenizersBackend dropped the
                    # additional_special_tokens property that some legacy custom
                    # processor code (e.g. Florence-2) still accesses.  Patch
                    # the class temporarily and retry.
                    logger.warning(f"Patching tokenizer for {model_id} " f"(missing additional_special_tokens).")
                    self.processor = _load_processor_patched(
                        AutoProcessor,
                        model_id,
                        self.trust_remote_code,
                    )
                except (ValueError, OSError):
                    from transformers import AutoTokenizer

                    logger.warning(f"AutoProcessor not available for {model_id}; " f"falling back to AutoTokenizer.")
                    self.processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.trust_remote_code)
                logger.debug(f"Loaded processor for {model_id}")

                # Load model — fallback chain for trust_remote_code models whose
                # custom config is not registered with AutoModelForImageTextToText.
                #
                # Tried in order:
                #   1. AutoModelForImageTextToText  (standard, preferred)
                #   2. AutoModel                    (generic, handles most custom configs)
                #   3. AutoModelForCausalLM         (for LLM-style VLMs like Moondream2)
                #
                # For each class, if Accelerate's meta-device dispatch causes a
                # "Tensor.item() cannot be called on meta tensors" RuntimeError,
                # the load is retried WITHOUT device_map and the model is moved
                # to the target device manually afterward.
                AutoModelForImageTextToText = transformers_lib["AutoModelForImageTextToText"]  # noqa: N806
                load_kwargs = dict(
                    torch_dtype=self.dtype,
                    device_map=self.device,
                    trust_remote_code=self.trust_remote_code,
                )

                def _try_load(cls, kwargs, src=None):
                    """Try loading with device_map; on meta-tensor error, retry without it."""
                    src = src or model_id
                    try:
                        return cls.from_pretrained(src, **kwargs)
                    except RuntimeError as re:
                        if "meta tensors" not in str(re):
                            raise
                        logger.warning(
                            f"{cls.__name__} hit meta-tensor error for {src} "
                            f"(device_map incompatible); retrying without device_map."
                        )
                        no_dmap = {k: v for k, v in kwargs.items() if k != "device_map"}
                        model = cls.from_pretrained(src, **no_dmap)
                        return model.to(self.device)

                try:
                    self.model = _try_load(AutoModelForImageTextToText, load_kwargs)
                except (AttributeError, RuntimeError):
                    # transformers >=5.0 compat: legacy custom configs (e.g.
                    # microsoft/Florence-2) use APIs removed in v5.  If a
                    # community-converted version is registered in
                    # _MODEL_COMPAT_REDIRECTS, silently redirect to it and
                    # reload without trust_remote_code.
                    redirect = _MODEL_COMPAT_REDIRECTS.get(model_id)
                    if redirect is None:
                        raise
                    logger.warning(
                        f"Model {model_id!r} is incompatible with the installed "
                        f"transformers version; redirecting to {redirect!r}."
                    )
                    self._effective_model_id = redirect
                    compat_kwargs = dict(load_kwargs)
                    compat_kwargs["trust_remote_code"] = False
                    self.model = _try_load(AutoModelForImageTextToText, compat_kwargs, src=redirect)
                    # Reload the processor too from the compatible repo.
                    self.processor = AutoProcessor.from_pretrained(redirect, trust_remote_code=False)
                except ValueError as ve:
                    if "Unrecognized configuration class" not in str(ve):
                        raise
                    logger.warning(
                        f"AutoModelForImageTextToText does not support {model_id}'s "
                        f"custom config; trying fallback loaders."
                    )
                    from transformers import AutoModel, AutoModelForCausalLM

                    self.model = None
                    last_exc: Exception = ve
                    for fallback_cls in (AutoModel, AutoModelForCausalLM):
                        try:
                            self.model = _try_load(fallback_cls, load_kwargs)
                            break
                        except ValueError as inner_ve:
                            if "Unrecognized configuration class" in str(inner_ve):
                                last_exc = inner_ve
                                continue
                            raise
                        except Exception as inner_exc:
                            last_exc = inner_exc
                            continue
                    if self.model is None:
                        raise last_exc

            self.model.eval()

            logger.info(f"VLM model loaded on {self.device} ({time.perf_counter() - t0:.1f}s): {model_id}")

        except Exception as e:
            raise ModelLoadError(model_id, str(e))

    def predict(
        self,
        image: ImageInput | None = None,
        prompt: str | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        output_mode: str | None = None,
        images: list[ImageInput] | None = None,
        auto_promote: bool = False,
        **kwargs,
    ) -> VisionResult:
        """Generate text response for single or multiple images with a prompt.

        Uses chat-based API to process image(s) and text prompt, generating
        a natural language response. Supports system prompts for behavior
        customization and configurable generation parameters.

        Multi-image support: Qwen3-VL and other modern VLMs can process
        multiple images in a single query for comparison, relationship
        analysis, or multi-view understanding.

        Args:
            image: Primary image input (file path, PIL Image, or numpy array).
                Optional if images parameter is provided.
            prompt: Text prompt/question about the image(s) (required)
            system_prompt: Optional system prompt (overrides constructor default)
            max_new_tokens: Max tokens to generate (overrides constructor default)
            temperature: Sampling temperature (overrides constructor default)
            top_p: Nucleus sampling probability (overrides constructor default)
            top_k: Top-k sampling parameter (overrides constructor default)
            output_mode: Structured output mode for JSON parsing. Supported values:
                - None: Raw text mode (default, existing behavior)
                - "json": Generic JSON output
                - "detect": Object detection format (label + confidence)
                - "classify": Classification format (label + confidence)
                - "describe": Description format (description + objects + scene)
            images: Additional images for multi-image queries. When provided,
                all images (primary + additional) are sent to the model.
                Can be used alone (without primary image) or combined.
            auto_promote: If True, automatically promote entities with bbox/mask
                to Instance objects when parsing JSON output. Enables direct
                comparison with spatial detection models when using VLMs that
                output bounding boxes (e.g., Qwen3-VL grounding mode).
                Default: False (entities remain as Entity objects).
            **kwargs: Additional arguments (reserved for future use)

        Returns:
            VisionResult with:
                - instances: Instance objects if auto_promote=True and bbox/mask found,
                  else empty list
                - entities: Parsed entities if output_mode is set (unless promoted),
                  else empty list
                - text: Generated text response (always includes raw output)
                - prompt: Original input prompt
                - meta: Model info, generation metadata, image_paths, image_count

        Raises:
            InvalidInputError: If prompt is None or empty, or if neither
                image nor images parameter is provided

        Examples:
            >>> # Single image (existing API, backward compatible)
            >>> result = vlm.predict("cat.jpg", prompt="What animal is this?")
            >>> print(result.text)
            "This is a cat."

            >>> # Multi-image comparison (primary + additional)
            >>> result = vlm.predict(
            ...     "main.jpg",
            ...     images=["ref1.jpg", "ref2.jpg"],
            ...     prompt="Compare these images."
            ... )

            >>> # Multi-image without primary (images-only mode)
            >>> result = vlm.predict(
            ...     images=["img1.jpg", "img2.jpg"],
            ...     prompt="What's different between these images?"
            ... )

            >>> # With structured output and auto-promotion (Qwen3-VL grounding)
            >>> result = vlm.predict(
            ...     "scene.jpg",
            ...     prompt="Detect objects with bboxes in JSON format.",
            ...     output_mode="detect",
            ...     auto_promote=True  # Bboxes in JSON -> Instance objects
            ... )
            >>> print(len(result.instances))  # Instances with bboxes
            3
        """
        # Validate required prompt parameter
        if not prompt:
            raise InvalidInputError("prompt is required for VLM task", prompt)

        # Collect all images from both image and images parameters
        all_images = []  # List of PIL images
        all_image_paths = []  # List of paths (for meta)

        if image is not None:
            pil_image, image_path = self._load_image(image)
            all_images.append(pil_image)
            if image_path:
                all_image_paths.append(image_path)

        if images is not None:
            loaded = self._load_images(images)
            for pil_img, img_path in loaded:
                all_images.append(pil_img)
                if img_path:
                    all_image_paths.append(img_path)

        if not all_images:
            raise InvalidInputError("At least one image is required. Provide 'image' and/or 'images' parameter.")

        # Store original image dimensions for bbox coordinate scaling
        original_width, original_height = all_images[0].size
        logger.debug(f"Original image dimensions: {original_width}x{original_height}")

        # Build chat messages list
        messages = []

        # Add system prompt if provided (predict-time overrides constructor default)
        effective_system = system_prompt if system_prompt is not None else self.system_prompt

        # Inject JSON formatting instruction when output_mode is set
        if output_mode:
            schema_instruction = get_json_schema(output_mode)
            if schema_instruction:
                if effective_system:
                    effective_system = f"{effective_system}\n\n{schema_instruction}"
                else:
                    effective_system = schema_instruction

        if effective_system:
            messages.append({"role": "system", "content": [{"type": "text", "text": effective_system}]})

        # Build user content with all images + text prompt
        user_content = []
        for pil_img in all_images:
            user_content.append({"type": "image", "image": pil_img})
        user_content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": user_content})

        # Process inputs — chat-based API (standard VLMs) or direct API
        # (encoder-decoder VLMs like Florence-2 that pre-date chat templates).
        has_chat_template = (
            hasattr(self.processor, "apply_chat_template")
            and getattr(self.processor, "chat_template", None) is not None
        )

        if has_chat_template:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(self.device)
        else:
            # Florence-2 and similar encoder-decoder VLMs: pass text + images
            # directly to the processor without chat markup.
            if len(all_images) > 1:
                logger.debug(
                    f"Processor for {self.model_id} does not support chat template; "
                    f"only the first image will be used."
                )
            inputs = self.processor(text=prompt, images=all_images[0], return_tensors="pt").to(self.device)
            # Cast floating-point inputs to the model's dtype so they match
            # model weights (e.g. community Florence-2 loads as float16 which
            # mismatches float32 processor outputs).
            model_dtype = next(self.model.parameters()).dtype
            for k, v in inputs.items():
                if hasattr(v, "is_floating_point") and v.is_floating_point():
                    inputs[k] = v.to(model_dtype)

        # Determine generation parameters (predict-time overrides constructor defaults)
        gen_max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        gen_temperature = temperature if temperature is not None else self.temperature
        gen_top_p = top_p if top_p is not None else self.top_p
        gen_top_k = top_k if top_k is not None else self.top_k

        # Generate response under no_grad context
        # Fix: do_sample=False when temperature=0 for greedy decoding
        do_sample = gen_temperature > 0
        with self.torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_max_tokens,
                do_sample=do_sample,
                **({"temperature": gen_temperature, "top_p": gen_top_p, "top_k": gen_top_k} if do_sample else {}),
            )

        # Trim input tokens to get only the generated portion.
        # Encoder-decoder models (e.g. Florence-2): generate() returns the
        # decoder output only — no prompt tokens to strip.
        # Decoder-only (causal LM) models: generate() echoes the prompt, so
        # we must slice it off.
        if getattr(self.model.config, "is_encoder_decoder", False):
            trimmed = generated_ids
        else:
            trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]

        # Decode generated tokens to text
        output_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Parse structured output if output_mode is set
        instances = []
        entities = []
        if output_mode:
            json_data = extract_json_from_text(output_text)
            if json_data is not None:
                parsed = parse_entities(json_data, auto_promote=auto_promote)
                # Separate instances from entities based on type
                if auto_promote:
                    for item in parsed:
                        if isinstance(item, Instance):
                            # Scale bbox from VLM coordinate space to original image dimensions
                            if item.bbox is not None:
                                scaled_bbox = self._scale_bbox_from_vlm(item.bbox, original_width, original_height)
                                # Create new Instance with scaled bbox (frozen dataclass)
                                item = Instance(
                                    bbox=scaled_bbox,
                                    mask=item.mask,
                                    score=item.score,
                                    label=item.label,
                                    label_name=item.label_name,
                                    area=item.area,
                                    is_stuff=item.is_stuff,
                                    embedding=item.embedding,
                                    track_id=item.track_id,
                                    keypoints=item.keypoints,
                                )
                            instances.append(item)
                        else:
                            entities.append(item)
                    logger.info(
                        f"Parsed {len(instances)} instances and {len(entities)} entities "
                        f"from VLM output (mode={output_mode}, auto_promote={auto_promote})"
                    )
                else:
                    entities = parsed
                    logger.info(f"Parsed {len(entities)} entities from VLM output (mode={output_mode})")
            else:
                logger.warning(
                    f"Failed to parse JSON from VLM output (mode={output_mode}). "
                    f"Returning raw text. Output preview: {output_text[:100]}..."
                )

        # Return VisionResult with text output and parsed entities/instances
        return VisionResult(
            instances=instances,  # Populated when auto_promote=True and spatial data found
            entities=entities,  # Populated when output_mode is set (unless promoted)
            meta={
                "model_id": self.model_id,
                "device": str(self.device),
                "backend": "huggingface",
                "max_new_tokens": gen_max_tokens,
                "tokens_generated": len(trimmed[0]),
                "image_path": all_image_paths[0] if all_image_paths else None,  # Backward compat
                "image_paths": all_image_paths,  # All image paths
                "image_count": len(all_images),  # Total number of images
                "image_width": original_width,  # Original image dimensions
                "image_height": original_height,
                "output_mode": output_mode,
            },
            text=output_text.strip(),
            prompt=prompt,
        )

    def _scale_bbox_from_vlm(self, bbox: BBox, image_width: int, image_height: int) -> BBox:
        """Scale bbox from VLM coordinate space to original image dimensions.

        Different VLMs use different coordinate conventions for bbox output.
        This method applies a heuristic to auto-detect the coordinate system:

        1. If all coordinates are in [0.0, 2.0) → treat as [0, 1] normalized
           (e.g., some grounding models output fractional coords)
        2. If all coordinates are in [0, 1500) → treat as ~1000-unit normalized
           (e.g., Qwen3-VL uses ~1000-unit coordinate space)
        3. Otherwise → treat as raw pixel coordinates (no scaling)

        Args:
            bbox: Bbox in VLM coordinate space (x1, y1, x2, y2)
            image_width: Original image width in pixels
            image_height: Original image height in pixels

        Returns:
            Scaled bbox in original image coordinate space (xyxy format)

        Examples:
            >>> # VLM bbox: [500, 500, 1000, 1000] in 1000x1000 space (Qwen3-VL style)
            >>> # Original image: 640x480
            >>> scaled = adapter._scale_bbox_from_vlm(
            ...     (500, 500, 1000, 1000), 640, 480
            ... )
            >>> # Result: (320, 240, 640, 480)
        """
        x1, y1, x2, y2 = bbox
        max_coord = max(x1, y1, x2, y2)

        if max_coord < 2.0:
            # [0, 1] normalized coordinates → scale to image dimensions
            scale_x = image_width
            scale_y = image_height
        elif max_coord < 1500:
            # ~1000-unit normalized (Qwen3-VL style) → scale proportionally
            vlm_size = 1000
            scale_x = image_width / vlm_size
            scale_y = image_height / vlm_size
        else:
            # Raw pixel coordinates → no scaling needed
            scale_x = 1.0
            scale_y = 1.0

        scaled_x1 = x1 * scale_x
        scaled_y1 = y1 * scale_y
        scaled_x2 = x2 * scale_x
        scaled_y2 = y2 * scale_y

        # Clamp to image boundaries
        scaled_x1 = max(0, min(scaled_x1, image_width))
        scaled_y1 = max(0, min(scaled_y1, image_height))
        scaled_x2 = max(0, min(scaled_x2, image_width))
        scaled_y2 = max(0, min(scaled_y2, image_height))

        logger.debug(
            f"Scaled bbox from VLM space {bbox} to image space "
            f"({scaled_x1:.1f}, {scaled_y1:.1f}, {scaled_x2:.1f}, {scaled_y2:.1f})"
        )

        return (scaled_x1, scaled_y1, scaled_x2, scaled_y2)

    def info(self) -> dict[str, Any]:
        """Get adapter metadata.

        Returns:
            Dictionary with adapter information including:
                - name: Adapter class name
                - task: Task type ("vlm")
                - model_id: HuggingFace model ID
                - device: Device being used
                - backend: Runtime backend ("huggingface")
                - max_new_tokens: Default max tokens
                - system_prompt: Default system prompt (if set)
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter

        Examples:
            >>> vlm = HuggingFaceVLMAdapter("Qwen/Qwen3-VL-2B-Instruct")
            >>> info = vlm.info()
            >>> print(info["task"])
            "vlm"
            >>> print(info["model_id"])
            "Qwen/Qwen3-VL-2B-Instruct"
        """
        return {
            "name": self.__class__.__name__,
            "task": self.task,
            "model_id": self.model_id,
            "device": str(self.device),
            "backend": "huggingface",
            "max_new_tokens": self.max_new_tokens,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }

    @staticmethod
    def _is_vlm_model(model_id: str) -> bool:
        """Detect if a model ID corresponds to a known VLM architecture.

        Pattern-matches model IDs against known VLM architecture names.
        Case-insensitive matching for robustness.

        This method is used by UniversalLoader for future auto-detection
        capabilities (not yet wired in v1.5.3, but ready for v1.6).

        Args:
            model_id: HuggingFace model ID or path

        Returns:
            True if model ID matches known VLM patterns, False otherwise

        Examples:
            >>> HuggingFaceVLMAdapter._is_vlm_model("Qwen/Qwen3-VL-2B-Instruct")
            True
            >>> HuggingFaceVLMAdapter._is_vlm_model("llava-hf/llava-1.5-7b-hf")
            True
            >>> HuggingFaceVLMAdapter._is_vlm_model("facebook/detr-resnet-50")
            False
        """
        model_id_lower = model_id.lower()

        # Known VLM architecture patterns (case-insensitive regex)
        vlm_patterns = [
            r"qwen.*vl",  # Qwen-VL, Qwen2-VL, Qwen3-VL
            r"llava",  # LLaVA variants
            r"internvl",  # InternVL
            r"phi.*vision",  # Phi-Vision
            r"cogvlm",  # CogVLM
            r"minicpm.*v",  # MiniCPM-V
            r"molmo",  # Molmo
            r"idefics",  # Idefics
            r"paligemma",  # PaliGemma, PaliGemma 2
            r"florence",  # Florence-2
            # v1.9.3 additions
            r"medgemma",  # Google MedGemma
            r"lfm.*vl",  # LiquidAI LFM2-VL, LFM2.5-VL
            r"smolvlm",  # HuggingFace SmolVLM
            r"moondream",  # Moondream2
            # v1.9.8 additions
            r"gemma-4",  # Google Gemma 4 multimodal (NOT gemma-2/3 text-only)
        ]

        # Check if any pattern matches
        for pattern in vlm_patterns:
            if re.search(pattern, model_id_lower):
                return True

        return False
