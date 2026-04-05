from __future__ import annotations

import threading
from typing import Any

from mata.annotate.coco_io import xyxy_to_xywh
from mata.core.logging import get_logger

logger = get_logger(__name__)


class AIAssist:
	"""AI-assist bridge for annotation workflows.

	Task E1 implements detection pre-labeling only. VLM and CLIP helpers remain
	as explicit stubs until Tasks E2 and E3 land.
	"""

	def __init__(
		self,
		*,
		detect_model: str | None = None,
		vlm_model: str | None = None,
		embed_model: str | None = None,
		detect_kwargs: dict[str, Any] | None = None,
		vlm_kwargs: dict[str, Any] | None = None,
		embed_kwargs: dict[str, Any] | None = None,
	) -> None:
		self._detect_adapter: Any = None
		self._vlm_adapter: Any = None
		self._embed_adapter: Any = None
		self._zeroshot_adapter: Any = None
		self._zeroshot_model: str | None = None

		self._detect_model = detect_model
		self._vlm_model = vlm_model
		self._embed_model = embed_model
		self._detect_kwargs = dict(detect_kwargs or {})
		self._vlm_kwargs = dict(vlm_kwargs or {})
		self._embed_kwargs = dict(embed_kwargs or {})

		self._detect_lock = threading.RLock()
		self._vlm_lock = threading.RLock()
		self._embed_lock = threading.RLock()

	def load_detect(self, model: str | None = None, **kwargs: Any) -> None:
		"""Lazily load the detection model via ``mata.load('detect', ...)``."""
		resolved_model = model or self._detect_model
		if not resolved_model:
			raise ValueError("No detection model configured for annotation assist.")

		load_kwargs = {**self._detect_kwargs, **kwargs}
		with self._detect_lock:
			if self._detect_adapter is not None and resolved_model == self._detect_model and not kwargs:
				return

			import mata

			self._detect_adapter = mata.load("detect", resolved_model, **load_kwargs)
			self._detect_model = resolved_model
			self._detect_kwargs = load_kwargs
			logger.info("Loaded annotate detection assist model: %s", resolved_model)

	def detect_assist(
		self,
		image_path: str,
		threshold: float = 0.3,
		class_map: dict[int, str] | dict[str, str] | None = None,
	) -> list[dict[str, Any]]:
		"""Run detection inference and return editable annotation candidates."""
		if self._detect_adapter is None:
			self.load_detect()

		assert self._detect_adapter is not None
		threshold_value = float(threshold)

		with self._detect_lock:
			result = self._detect_adapter.predict(image_path, threshold=threshold_value)

		candidates: list[dict[str, Any]] = []
		for inst in getattr(result, "instances", []):
			bbox = getattr(inst, "bbox", None)
			if bbox is None or len(bbox) != 4:
				continue

			score = getattr(inst, "score", None)
			score_value = float(score) if score is not None else 0.0
			if score is not None and score_value < threshold_value:
				continue

			bbox_xyxy = [float(coord) for coord in bbox]
			label_id = int(getattr(inst, "label", -1))
			label = self._resolve_label(inst, label_id, class_map)
			candidates.append(
				{
					"bbox_xywh": xyxy_to_xywh(bbox_xyxy),
					"bbox_xyxy": bbox_xyxy,
					"label": label,
					"label_id": label_id,
					"score": score_value,
					"source": "detect",
				}
			)

		return candidates

	# ------------------------------------------------------------------
	# VLM assist (Task E2)
	# ------------------------------------------------------------------

	def load_vlm(self, model: str | None = None, **kwargs: Any) -> None:
		"""Lazily load the VLM adapter via ``mata.load('vlm', ...)``."""
		resolved_model = model or self._vlm_model
		if not resolved_model:
			raise ValueError("No VLM model configured for annotation assist.")

		load_kwargs = {**self._vlm_kwargs, **kwargs}
		with self._vlm_lock:
			if self._vlm_adapter is not None and resolved_model == self._vlm_model and not kwargs:
				return

			import mata

			self._vlm_adapter = mata.load("vlm", resolved_model, **load_kwargs)
			self._vlm_model = resolved_model
			self._vlm_kwargs = load_kwargs
			logger.info("Loaded annotate VLM assist model: %s", resolved_model)

	def vlm_assist(
		self,
		image_path: str,
		class_names: list[str] | None = None,
		prompt: str | None = None,
		max_new_tokens: int = 2048,
	) -> list[dict[str, Any]]:
		"""Run VLM with structured detection output and return annotation candidates.

		If *class_names* is provided the prompt is constructed automatically.
		If *prompt* is provided it is used directly.
		Results are always suggestions — never auto-committed.
		"""
		if self._vlm_adapter is None:
			self.load_vlm()

		if prompt is None and class_names:
			prompt = (
				f"Detect all objects in the image that belong to these classes: {', '.join(class_names)}. "
				"List EVERY occurrence -- there may be multiple instances of the same class. "
				"For each object output one line with the format: "
				"<class_name> [x1, y1, x2, y2] where (x1,y1) is the top-left corner and "
				"(x2,y2) is the bottom-right corner in pixel coordinates. "
				"Do not skip any visible instance."
			)

		try:
			with self._vlm_lock:
				result = self._vlm_adapter.predict(
					image=image_path,
					prompt=prompt,
					output_mode="detect",
					max_new_tokens=max_new_tokens,
					auto_promote=True,
				)
		except Exception as exc:
			logger.warning("VLM assist failed for %s: %s", image_path, exc)
			return []

		candidates: list[dict[str, Any]] = []

		for inst in getattr(result, "instances", []):
			bbox = getattr(inst, "bbox", None)
			if bbox is None or len(bbox) != 4:
				continue
			bbox_xyxy = [float(coord) for coord in bbox]
			label_id = int(getattr(inst, "label", 0))
			label = getattr(inst, "label_name", None) or "object"
			score = getattr(inst, "score", None)
			candidates.append(
				{
					"bbox_xywh": xyxy_to_xywh(bbox_xyxy),
					"bbox_xyxy": bbox_xyxy,
					"label": label,
					"label_id": label_id,
					"score": float(score) if score is not None else 0.5,
					"source": "vlm",
				}
			)

		for ent in getattr(result, "entities", []):
			label = getattr(ent, "label", "")
			if not label:
				continue
			score = getattr(ent, "score", None)
			candidates.append(
				{
					"label": label,
					"score": float(score) if score is not None else 0.5,
					"source": "vlm_entity",
				}
			)

		return candidates

	def vlm_describe(self, image_path: str, prompt: str | None = None) -> str:
		"""Get free-text description of an image for classification context."""
		if self._vlm_adapter is None:
			self.load_vlm()

		try:
			with self._vlm_lock:
				result = self._vlm_adapter.predict(
					image=image_path,
					prompt=prompt or "Describe this image in detail.",
				)
		except Exception as exc:
			logger.warning("VLM describe failed for %s: %s", image_path, exc)
			return ""

		return getattr(result, "text", None) or result.meta.get("text", "") or ""

	# ------------------------------------------------------------------
	# Task E3: CLIP zero-shot classification suggestions
	# ------------------------------------------------------------------

	def load_embed(self, model: str | None = None, **kwargs: Any) -> None:
		"""Lazily load the CLIP/embed model via ``mata.load('classify', ...)``."""
		resolved_model = model or self._embed_model
		if not resolved_model:
			raise ValueError("No embed model configured for annotation assist.")

		load_kwargs = {**self._embed_kwargs, **kwargs}
		with self._embed_lock:
			if self._embed_adapter is not None and resolved_model == self._embed_model and not kwargs:
				return

			import mata

			self._embed_adapter = mata.load("classify", resolved_model, **load_kwargs)
			self._embed_model = resolved_model
			self._embed_kwargs = load_kwargs
			logger.info("Loaded annotate embed/CLIP assist model: %s", resolved_model)

	def classify_assist(
		self,
		image_path: str,
		class_names: list[str],
	) -> list[dict[str, Any]]:
		"""Run CLIP zero-shot classification and return ranked suggestions.

		Args:
			image_path: Path to the image to classify.
			class_names: Candidate class labels, e.g. ``["cat", "dog", "bird"]``.

		Returns:
			List of suggestion dicts sorted by score descending::

			    [{"label": "cat", "score": 0.85, "source": "clip"}, ...]
		"""
		if self._embed_adapter is None:
			self.load_embed()

		assert self._embed_adapter is not None

		with self._embed_lock:
			result = self._embed_adapter.predict(image_path, text_prompts=class_names)

		suggestions = [
			{"label": cls.label, "score": float(cls.score), "source": "clip"}
			for cls in getattr(result, "predictions", [])
		]
		return sorted(suggestions, key=lambda x: x["score"], reverse=True)

	# ------------------------------------------------------------------
	# Task E4: Zero-shot detection via Grounding DINO
	# ------------------------------------------------------------------

	def load_zeroshot(self, model: str = "IDEA-Research/grounding-dino-tiny", **kwargs: Any) -> None:
		"""Lazily load a zero-shot detection model (Grounding DINO by default)."""
		load_kwargs = {**kwargs}
		with self._detect_lock:
			if self._zeroshot_adapter is not None and self._zeroshot_model == model and not kwargs:
				return
			import mata
			self._zeroshot_adapter = mata.load("detect", model, **load_kwargs)
			self._zeroshot_model = model
			logger.info("Loaded zero-shot detection model: %s", model)

	def zeroshot_detect_assist(
		self,
		image_path: str,
		text_prompts: str | list[str],
		threshold: float = 0.3,
		model: str = "IDEA-Research/grounding-dino-tiny",
	) -> list[dict[str, Any]]:
		"""Run zero-shot detection using Grounding DINO and return annotation candidates.

		Args:
			image_path: Path to the image to annotate.
			text_prompts: Space-delimited prompt string or list of class names.
				Lists are joined with " . " (Grounding DINO convention).
			threshold: Confidence threshold (default 0.3).
			model: HuggingFace model ID (default Grounding DINO tiny).

		Returns:
			List of candidate dicts with bbox_xywh, bbox_xyxy, label, score, source.
		"""
		if isinstance(text_prompts, list):
			prompt_str = " . ".join(text_prompts)
		else:
			prompt_str = str(text_prompts)

		with self._detect_lock:
			if self._zeroshot_adapter is None or self._zeroshot_model != model:
				self.load_zeroshot(model)

			assert self._zeroshot_adapter is not None
			try:
				result = self._zeroshot_adapter.predict(
					image_path,
					text_prompts=prompt_str,
					threshold=float(threshold),
				)
			except Exception as exc:
				logger.warning("Zero-shot detection failed for %s: %s", image_path, exc)
				return []

		candidates: list[dict[str, Any]] = []
		for inst in getattr(result, "instances", []):
			bbox = getattr(inst, "bbox", None)
			if bbox is None or len(bbox) != 4:
				continue
			score = getattr(inst, "score", None)
			score_value = float(score) if score is not None else 0.0
			if score_value < float(threshold):
				continue
			bbox_xyxy = [float(coord) for coord in bbox]
			label = getattr(inst, "label_name", None) or str(getattr(inst, "label", "object"))
			candidates.append(
				{
					"bbox_xywh": xyxy_to_xywh(bbox_xyxy),
					"bbox_xyxy": bbox_xyxy,
					"label": label,
					"label_id": int(getattr(inst, "label", -1)),
					"score": score_value,
					"source": "zeroshot",
				}
			)

		return candidates

	@staticmethod
	def _resolve_label(
		inst: Any,
		label_id: int,
		class_map: dict[int, str] | dict[str, str] | None,
	) -> str:
		if class_map:
			if label_id in class_map:
				return str(class_map[label_id])
			label_key = str(label_id)
			if label_key in class_map:
				return str(class_map[label_key])

		label_name = getattr(inst, "label_name", None)
		if label_name:
			return str(label_name)
		return str(label_id)
