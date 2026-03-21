"""HuggingFace OCR adapter for MATA framework.

Supports three OCR architectures:
- GLM-OCR (zai-org/GLM-OCR): Full-image end-to-end OCR via AutoModelForImageTextToText
  with chat-template inference (apply_chat_template). State-of-the-art document OCR.
- GOT-OCR2 (stepfun-ai/GOT-OCR-2.0-hf): Full-image end-to-end OCR via AutoModelForCausalLM.
  Requires ``trust_remote_code=True``.
- TrOCR (microsoft/trocr-*): Sequence-to-sequence OCR via VisionEncoderDecoderModel.
  Designed for pre-cropped single text-line images; performance degrades on full-page documents.
"""

from __future__ import annotations

from typing import Any

from mata.core.exceptions import ModelLoadError
from mata.core.logging import get_logger
from mata.core.types import OCRResult, TextRegion

from ..pytorch_base import PyTorchBaseAdapter

logger = get_logger(__name__)

_transformers = None
TRANSFORMERS_AVAILABLE: bool | None = None


def _ensure_transformers() -> Any:
    """Ensure transformers library is available (lazy loading).

    Returns:
        transformers module

    Raises:
        ImportError: If transformers is not installed.
    """
    global _transformers, TRANSFORMERS_AVAILABLE
    if _transformers is None:
        try:
            import transformers

            _transformers = transformers
            TRANSFORMERS_AVAILABLE = True
            logger.debug(f"transformers {transformers.__version__} loaded successfully")
        except ImportError as exc:
            TRANSFORMERS_AVAILABLE = False
            raise ImportError(
                "transformers is required for HuggingFaceOCRAdapter. " "Install with: pip install transformers"
            ) from exc
    return _transformers


class HuggingFaceOCRAdapter(PyTorchBaseAdapter):
    """OCR adapter for HuggingFace models (GLM-OCR, GOT-OCR2, TrOCR).

    Architecture is auto-detected from the model ID:
    - IDs containing ``glm-ocr``, ``glm_ocr``, or ``glmocr`` → GLM-OCR
      (AutoModelForImageTextToText + chat-template API)
    - IDs containing ``got-ocr``, ``gotocr``, or ``got_ocr`` → GOT-OCR2 (AutoModelForCausalLM)
    - IDs containing ``trocr`` → TrOCR (VisionEncoderDecoderModel)
    - All others fall back to TrOCR pattern.

    .. note::
        GLM-OCR (``zai-org/GLM-OCR``) uses the chat-template inference pattern
        (``apply_chat_template``) and delivers strong document OCR with a single
        ``"Text Recognition:"`` prompt. Pass ``prompt=`` to override.

    .. warning::
        GOT-OCR2 (``stepfun-ai/GOT-OCR-2.0-hf``) is known to hallucinate with some
        transformers versions. If you see garbled output, try upgrading transformers
        (``pip install -U transformers``) or use EasyOCR/PaddleOCR instead.

    .. warning::
        TrOCR is designed for single text-line crops. It underperforms significantly
        on full-page or multi-line documents. Use GLM-OCR, GOT-OCR2, or an external
        engine (EasyOCR / PaddleOCR / Tesseract) for full-page document OCR.

    .. note::
        GOT-OCR2 requires ``trust_remote_code=True``, which executes model-specific
        code from the HuggingFace Hub. Only use models from trusted sources.

    Example::

        from mata.adapters.ocr import HuggingFaceOCRAdapter

        # GLM-OCR — state-of-the-art document OCR (v1.9.4+)
        adapter = HuggingFaceOCRAdapter("zai-org/GLM-OCR")
        result = adapter.predict("document.png")
        print(result.full_text)

        # TrOCR — single text-line crops
        adapter2 = HuggingFaceOCRAdapter("microsoft/trocr-base-handwritten")
        result2 = adapter2.predict("handwritten_note.jpg")
        print(result2.full_text)

        adapter3 = HuggingFaceOCRAdapter("stepfun-ai/GOT-OCR-2.0-hf")
        result3 = adapter3.predict("document.png")
    """

    name = "huggingface_ocr"
    task = "ocr"

    def __init__(self, model_id: str, device: str = "auto", **kwargs: Any) -> None:
        """Initialize the HuggingFace OCR adapter.

        Args:
            model_id: HuggingFace model ID (e.g. ``"microsoft/trocr-base-handwritten"``).
            device: Device to run inference on. ``"auto"`` selects CUDA if available.
            **kwargs: Additional keyword arguments forwarded to the model/processor
                      ``from_pretrained()`` call.

        Raises:
            ImportError: If ``transformers`` is not installed.
            ModelLoadError: If the model cannot be loaded from HuggingFace Hub.
        """
        super().__init__(device=device, threshold=0.0)
        self.model_id = model_id
        self._extra_kwargs = kwargs
        self._arch = self._detect_architecture(model_id)
        self._load_model()

    # ------------------------------------------------------------------
    # Architecture detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_architecture(model_id: str) -> str:
        """Detect OCR architecture from model ID string.

        Args:
            model_id: HuggingFace model ID.

        Returns:
            ``"glm_ocr"``, ``"trocr"``, or ``"got_ocr"``.
        """
        mid = model_id.lower()
        if "glm-ocr" in mid or "glm_ocr" in mid or "glmocr" in mid:
            return "glm_ocr"
        if "trocr" in mid:
            return "trocr"
        if "got-ocr" in mid or "gotocr" in mid or "got_ocr" in mid:
            return "got_ocr"
        # Fallback: attempt TrOCR pattern (VisionEncoderDecoderModel)
        logger.debug(
            f"Could not detect OCR architecture from '{model_id}'. "
            "Defaulting to TrOCR (VisionEncoderDecoderModel). "
            "Pass model_type explicitly if this is incorrect."
        )
        return "trocr"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load model and processor from HuggingFace Hub."""
        tf = _ensure_transformers()
        try:
            from mata.core.logging import suppress_third_party_logs

            if self._arch == "glm_ocr":
                logger.info(f"Loading GLM-OCR model: {self.model_id}")
                load_kwargs = dict(self._extra_kwargs)
                load_kwargs.setdefault("torch_dtype", "auto")
                use_device_map = str(self.device) != "cpu" and load_kwargs.pop("device_map", None) != "disabled"
                if use_device_map:
                    load_kwargs["device_map"] = "auto"
                with suppress_third_party_logs():
                    self._processor = tf.AutoProcessor.from_pretrained(self.model_id, **self._extra_kwargs)
                    self._model = tf.AutoModelForImageTextToText.from_pretrained(self.model_id, **load_kwargs)
                if not use_device_map:
                    self._model = self._model.to(self.device).eval()
                else:
                    self._model = self._model.eval()
                    # When device_map is used the model spans potentially multiple
                    # devices; record the primary device for input tensor placement.
                    self._infer_device = next(self._model.parameters()).device

            elif self._arch == "trocr":
                logger.info(f"Loading TrOCR model: {self.model_id}")
                with suppress_third_party_logs():
                    self._processor = tf.TrOCRProcessor.from_pretrained(self.model_id, **self._extra_kwargs)
                    self._model = tf.VisionEncoderDecoderModel.from_pretrained(self.model_id, **self._extra_kwargs)
                self._model = self._model.to(self.device).eval()

            elif self._arch == "got_ocr":
                logger.info(f"Loading GOT-OCR2 model: {self.model_id}")
                with suppress_third_party_logs():
                    self._processor = tf.AutoProcessor.from_pretrained(self.model_id, **self._extra_kwargs)
                    self._model = tf.AutoModelForImageTextToText.from_pretrained(self.model_id, **self._extra_kwargs)
                self._model = self._model.to(self.device).eval()

            logger.info(f"OCR model loaded successfully on {self.device} (arch={self._arch})")

        except ImportError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                self.model_id,
                f"Failed to load HuggingFace OCR model: {type(exc).__name__}: {exc}",
            ) from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image: Any, **kwargs: Any) -> OCRResult:
        """Run OCR on an image.

        Args:
            image: Input image (path string, URL, PIL Image, numpy array, or
                   MATA ``Image`` artifact).
            **kwargs: Architecture-specific options forwarded to generation:

                - **GLM-OCR**: ``prompt`` (str, default ``"Text Recognition:"``),
                  ``max_new_tokens`` (int, default ``8192``),
                  ``skip_special_tokens`` (bool, default ``True``).
                - **TrOCR**: ``max_new_tokens`` (int), any ``generate()`` kwarg.
                - **GOT-OCR2**: ``ocr_type`` (``"ocr"`` | ``"format"``, default ``"ocr"``),
                  ``max_new_tokens`` (int), any ``generate()`` kwarg.

        Returns:
            :class:`~mata.core.types.OCRResult` with extracted text regions.
        """
        pil_image, input_path = self._load_image(image)
        meta_base = {"model_id": self.model_id, "arch": self._arch}
        if input_path:
            meta_base["input_path"] = input_path

        if self._arch == "glm_ocr":
            return self._predict_glm_ocr(pil_image, meta=meta_base, **kwargs)
        elif self._arch == "trocr":
            return self._predict_trocr(pil_image, meta=meta_base, **kwargs)
        else:  # got_ocr
            return self._predict_got_ocr(pil_image, meta=meta_base, **kwargs)

    def _predict_glm_ocr(self, pil_image: Any, meta: dict[str, Any], **kwargs: Any) -> OCRResult:
        """Run GLM-OCR inference using the chat-template API.

        GLM-OCR follows the VLM chat pattern: it builds a user message containing
        the image and a text prompt, applies the processor's chat template, then
        decodes only the newly generated tokens.

        Args:
            pil_image: PIL Image in RGB format.
            meta: Base metadata dict to include in result.
            **kwargs: Supported options:

                - ``prompt`` (str): Text instruction (default ``"Text Recognition:"``).
                - ``max_new_tokens`` (int): Max tokens to generate (default ``8192``).
                - ``skip_special_tokens`` (bool): Strip special tokens from output
                  (default ``True``).

        Returns:
            :class:`OCRResult` with a single region and ``bbox=None``.
        """
        prompt = kwargs.pop("prompt", "Text Recognition:")
        max_new_tokens = kwargs.pop("max_new_tokens", 8192)
        skip_special_tokens = kwargs.pop("skip_special_tokens", True)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Determine the device for input tensor placement (device_map vs explicit device)
        target_device = getattr(self, "_infer_device", self.device)

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(target_device)
        # GLM-OCR does not use token_type_ids; drop if present to avoid
        # unexpected keyword argument errors in model.generate().
        inputs.pop("token_type_ids", None)

        input_len = inputs["input_ids"].shape[1]
        with self.torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
        text = self._processor.decode(
            generated_ids[0][input_len:],
            skip_special_tokens=skip_special_tokens,
        )
        region = TextRegion(text=text, score=1.0, bbox=None, label="glm-ocr")
        return OCRResult(regions=[region], meta={**meta})

    def _predict_trocr(self, pil_image: Any, meta: dict[str, Any], **kwargs: Any) -> OCRResult:
        """Run TrOCR inference.

        TrOCR processes the entire image as a single text sequence.
        It is optimised for pre-cropped single-line text images.
        Result contains one :class:`TextRegion` with ``bbox=None``.

        Args:
            pil_image: PIL Image in RGB format.
            meta: Base metadata dict to include in result.
            **kwargs: Forwarded to ``model.generate()``.

        Returns:
            :class:`OCRResult` with a single region.
        """
        inputs = self._processor(images=pil_image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            generated_ids = self._model.generate(inputs.pixel_values, **kwargs)
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        region = TextRegion(text=text, score=1.0, bbox=None, label="printed")
        return OCRResult(regions=[region], meta={**meta})

    def _predict_got_ocr(self, pil_image: Any, meta: dict[str, Any], **kwargs: Any) -> OCRResult:
        """Run GOT-OCR2 inference.

        GOT-OCR2 performs full-image end-to-end OCR in a causal LM fashion.
        Result contains one :class:`TextRegion` with ``bbox=None``.

        Args:
            pil_image: PIL Image in RGB format.
            meta: Base metadata dict to include in result.
            **kwargs: Forwarded to ``model.generate()``.
                      ``ocr_type`` (``"ocr"`` | ``"format"``) is consumed here.

        Returns:
            :class:`OCRResult` with a single region.
        """
        # ocr_type="ocr" → plain text; ocr_type="format" → markdown/LaTeX output
        ocr_type = kwargs.pop("ocr_type", "ocr")
        formatted = ocr_type == "format"
        max_new_tokens = kwargs.pop("max_new_tokens", 4096)
        do_sample = kwargs.pop("do_sample", False)

        # Pass format=True for structured output; plain OCR needs no extra flag
        processor_kwargs = {"return_tensors": "pt"}
        if formatted:
            processor_kwargs["format"] = True
        inputs = self._processor(pil_image, **processor_kwargs).to(self.device)
        input_len = inputs["input_ids"].shape[1]
        with self.torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                tokenizer=self._processor.tokenizer,
                stop_strings="<|im_end|>",
                **kwargs,
            )
        # Decode only the newly generated tokens — skip the input prompt
        text = self._processor.decode(
            output[0][input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        region = TextRegion(text=text, score=1.0, bbox=None)
        return OCRResult(regions=[region], meta={**meta, "ocr_type": ocr_type})

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Return adapter metadata.

        Returns:
            Dict with ``name``, ``task``, ``model_id``, ``arch``, ``device``, ``backend``.
        """
        return {
            "name": self.name,
            "task": self.task,
            "model_id": self.model_id,
            "arch": self._arch,
            "device": str(self.device),
            "backend": "transformers",
        }
