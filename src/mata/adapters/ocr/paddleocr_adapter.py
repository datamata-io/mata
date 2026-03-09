"""PaddleOCR adapter for MATA framework."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from mata.adapters.base import BaseAdapter
from mata.core.logging import get_logger
from mata.core.types import OCRResult, TextRegion

logger = get_logger(__name__)

# On Windows, PaddlePaddle ≥3.x (GPU wheel) has several bugs that must be
# worked around BEFORE paddle's C++ backend initializes.  These stubs must be
# placed at module import time, which is guaranteed to execute before
# _ensure_paddleocr() is ever called.
#
# KNOWN WINDOWS LIMITATION (Bug 5 — torch/paddle cuDNN DLL conflict):
# torch 2.x (+cu12x) bundles cuDNN 9.1.x-era DLLs, while paddlepaddle-gpu
# ≥3.3.0 requires nvidia-cudnn-cu12==9.5.1.17 (cuDNN 9.5).  These two cuDNN
# major-release builds are mutually exclusive inside a single Windows process
# because their `cudnn_graph64_9.dll` export tables are incompatible (each
# contains 5–10 symbols the other doesn't provide).  The second framework to
# import always gets WinError 127 ("procedure not found") from LoadLibraryExW.
#
# There is no in-process workaround.  Options to resolve:
#   (a) Upgrade to torch ≥2.6.0 which bundles cuDNN ≥9.5.0 and therefore
#       matches the nvidia-cudnn-cu12 version that paddle requires.
#   (b) Use CPU-only paddle (paddlepaddle) which does not load CUDA cuDNN DLLs.
#   (c) Use a separate subprocess or distinct Python environment for OCR.
#
# Bug 1 — OneDNN / PIR executor (CPU + GPU):
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#   [pir::ArrayAttribute<...>]
#   Fix: disable MKL-DNN and the new PIR executor via env flags.
#
# Bug 2 — monkey_patch_variable() called twice (GPU wheel 3.0.0):
#   AttributeError: partially initialized module 'paddle' has no attribute
#   'tensor' (most likely due to a circular import)
#   Root cause: paddle/base/__init__.py calls monkey_patch_variable() (sets
#   _already_patch_variable=True) before paddle/__init__.py's own call.  The
#   second call takes the else-branch which does ``import paddle.tensor`` —
#   but paddle.tensor is not yet in sys.modules, causing the AttributeError.
#   Fix: _apply_paddle_base_init_patch() comments out the redundant first call
#   from paddle/base/__init__.py so the call in paddle/__init__.py is always
#   the first one (safe if-branch, no paddle.tensor needed).
#
# Bug 3 — broken distributed subpackages (GPU wheel 3.0.0):
#   Multiple ImportErrors when paddle/distributed/__init__.py loads fleet and
#   pipeline_scheduler_pass (training-only code with missing symbols in the
#   GPU inference wheel).
#   Fix: install a sys.meta_path finder (_PaddleFleetStubFinder) that auto-
#   stubs any import under paddle.distributed.fleet.* or
#   paddle.distributed.passes.pipeline_scheduler_pass.*.  Specific symbols
#   re-exported by paddle.distributed are patched onto the stubs manually
#   (ParallelMode, BoxPSDataset, InMemoryDataset, QueueDataset).
if sys.platform == "win32":
    import importlib.machinery as _imm
    import types as _types

    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")

    # Bug 4 — cuDNN DLL search path (GPU wheel, Windows):
    # paddle loads nvidia/cudnn/bin/cudnn_cnn64_9.dll by full path, but Windows
    # LoadLibrary does NOT automatically search the DLL's own directory for its
    # sub-dependencies (cudnn_ops64_9.dll, cudnn_graph64_9.dll).  Calling
    # os.add_dll_directory() for every nvidia/<pkg>/bin directory ensures all
    # cuDNN component DLLs are findable when the cuDNN router loads them.
    # NOTE: os.add_dll_directory() returns a handle that must be kept alive;
    # storing them in _nvidia_dll_dirs prevents GC from calling RemoveDllDirectory.
    _nvidia_dll_dirs: list = []
    try:
        import importlib.util as _ilu

        _sp = Path(_ilu.find_spec("paddle").origin).parent.parent  # site-packages
        _nvidia_dir = _sp / "nvidia"
        if _nvidia_dir.is_dir():
            for _pkg_dir in _nvidia_dir.iterdir():
                _bin = _pkg_dir / "bin"
                if _bin.is_dir():
                    _nvidia_dll_dirs.append(os.add_dll_directory(str(_bin)))
        del _ilu, _sp, _nvidia_dir
    except Exception:
        pass  # non-fatal — only needed on GPU wheels with nvidia-* packages

    # On paddlepaddle-gpu 3.0.0, paddle.distributed (and all its sub-packages)
    # contains broken circular imports and references to training-only symbols
    # that don't exist in the GPU inference wheel.  NONE of paddle.distributed
    # is needed for single-GPU OCR inference.
    #
    # Strategy: install a meta-path finder (_PaddleDistStubFinder) that
    # intercepts ALL imports under paddle.distributed.* and returns empty stubs.
    # We then pre-register a small set of modules with specific attributes that
    # paddle's own __init__.py and core modules access at import time (not just
    # at runtime):
    #
    #   paddle/__init__.py line 150  → from .distributed import DataParallel
    #   paddle/__init__.py line ~21  → from fleet.base.topology import ParallelMode
    #   paddle/__init__.py line ~22  → from fleet.dataset import InMemoryDataset, QueueDataset
    #   paddle/__init__.py line ~109 → from .fleet import BoxPSDataset
    #   paddle/nn/clip.py line ~29   → from distributed.utils.moe_utils import get_complete_pp_mesh

    class _PaddleDistStubLoader:
        """Loader that creates an empty stub for any paddle.distributed sub-import."""

        _ModuleType = _types.ModuleType  # captured value survives del _types

        def create_module(self, spec: _imm.ModuleSpec) -> _types.ModuleType:
            mod = self._ModuleType(spec.name)
            mod.__path__ = []  # type: ignore[attr-defined]
            mod.__spec__ = spec  # type: ignore[attr-defined]
            return mod

        def exec_module(self, module: _types.ModuleType) -> None:
            pass  # intentionally empty

    class _PaddleDistStubFinder:
        """Meta-path finder: auto-stubs ALL of paddle.distributed.* imports."""

        _PREFIX = "paddle.distributed"
        _loader = _PaddleDistStubLoader()
        _ModuleSpec = _imm.ModuleSpec  # captured value survives del _imm

        def find_spec(
            self,
            fullname: str,
            path: object,
            target: object = None,
        ) -> _imm.ModuleSpec | None:
            if fullname == self._PREFIX or fullname.startswith(self._PREFIX + "."):
                if fullname in sys.modules:
                    return None  # use pre-registered stub from sys.modules
                return self._ModuleSpec(fullname, self._loader, is_package=True)
            return None

    # Install FIRST in meta_path so it wins before the normal filesystem finder.
    if not any(isinstance(f, _PaddleDistStubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PaddleDistStubFinder())

    # Pre-register modules with the specific attributes that paddle's own import
    # time code needs.  These are set on stub modules BEFORE any import of paddle
    # resolves them, so every `from X import Y` finds Y already populated.

    def _make_dist_stub(name: str) -> _types.ModuleType:
        """Create and register an empty paddle.distributed stub."""
        if name not in sys.modules:
            m = _types.ModuleType(name)
            m.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = m
        return sys.modules[name]

    # Ensure parent stubs exist so Python's package hierarchy is consistent.
    _make_dist_stub("paddle.distributed")
    _make_dist_stub("paddle.distributed.fleet")
    _make_dist_stub("paddle.distributed.fleet.base")
    _make_dist_stub("paddle.distributed.fleet.utils")
    _make_dist_stub("paddle.distributed.utils")

    # paddle/__init__.py line 150: from .distributed import DataParallel
    sys.modules["paddle.distributed"].DataParallel = None  # type: ignore[attr-defined]

    # paddle.distributed.__init__.py line ~21: from fleet.base.topology import ParallelMode
    _topo = _make_dist_stub("paddle.distributed.fleet.base.topology")

    class _ParallelMode:
        """Minimal stub matching paddle.distributed.fleet.base.topology.ParallelMode."""

        DATA_PARALLEL = 0
        TENSOR_PARALLEL = 1
        PIPELINE_PARALLEL = 2
        SHARDING_PARALLEL = 3
        SEGMENT_PARALLEL = 4

    _topo.ParallelMode = _ParallelMode  # type: ignore[attr-defined]

    # paddle.distributed.__init__.py line ~22: from fleet.dataset import InMemoryDataset, QueueDataset
    _ds = _make_dist_stub("paddle.distributed.fleet.dataset")
    _ds.InMemoryDataset = None  # type: ignore[attr-defined]
    _ds.QueueDataset = None  # type: ignore[attr-defined]

    # paddle.distributed.__init__.py line ~109: from .fleet import BoxPSDataset
    sys.modules["paddle.distributed.fleet"].BoxPSDataset = None  # type: ignore[attr-defined]

    # paddle/nn/clip.py line ~29: from distributed.utils.moe_utils import get_complete_pp_mesh
    _moe = _make_dist_stub("paddle.distributed.utils.moe_utils")
    _moe.get_complete_pp_mesh = None  # type: ignore[attr-defined]

    del (
        _PaddleDistStubLoader,
        _PaddleDistStubFinder,
        _make_dist_stub,
        _topo,
        _ParallelMode,
        _ds,
        _moe,
        _types,
        _imm,
    )

_paddleocr_module = None


def _ensure_paddleocr() -> Any:
    """Ensure PaddleOCR is imported (lazy loading).

    Returns:
        PaddleOCR class

    Raises:
        ImportError: If paddleocr or paddlepaddle cannot be imported
    """
    global _paddleocr_module
    if _paddleocr_module is None:
        # Fail fast with a clear message if paddleocr/paddlepaddle major
        # versions are mismatched (e.g. paddleocr 3.x + paddlepaddle 2.x).
        # Run OUTSIDE the try/except so the helpful message is not swallowed.
        _check_paddle_version_compat()
        # On Windows, detect torch/paddle cuDNN DLL version conflict early so
        # the user gets a clear message instead of "procedure not found".
        # Must run OUTSIDE the try/except (below) so it isn't swallowed.
        if sys.platform == "win32":
            _check_paddle_windows_cudnn_compat()
        try:
            from paddleocr import PaddleOCR  # noqa: F401 (imported for side effects below)

            # paddleocr defers its paddle import to PaddleOCR() instantiation.
            # paddle/base/__init__.py calls monkey_patch_variable() (call #1,
            # sets _already_patch_variable=True) before paddle/__init__.py's
            # own call at line ~47 (call #2).  Call #2 then takes the else-
            # branch which does ``import paddle.tensor`` — but paddle.tensor
            # is not yet in sys.modules at that point, causing:
            #   AttributeError: partially initialized module 'paddle' has no
            #                   attribute 'tensor'
            # Fix: patch paddle/base/__init__.py to remove the premature call
            # (idempotent, MATA_FIX-guarded).  paddle/__init__.py's call then
            # becomes the first one (flag=False → safe if-branch, no tensor).
            if sys.platform == "win32":
                _apply_paddle_base_init_patch()
                # Eagerly pre-initialize paddle (with our stubs active and the
                # base-init patch in place) so PaddleOCR() sees a fully-loaded
                # paddle in sys.modules instead of re-running __init__.py.
                if "paddle" not in sys.modules:
                    try:
                        import paddle  # noqa: F401

                        logger.debug("paddle pre-initialized successfully")
                    except Exception as _pre_exc:
                        logger.debug("paddle pre-init failed (non-fatal): %s", _pre_exc)

            # Belt-and-suspenders: also disable MKL-DNN via the Python API so
            # the flag is respected even if paddle was initialized elsewhere.
            if sys.platform == "win32":
                try:
                    # Use the already-loaded paddle module from sys.modules to avoid
                    # triggering a re-import of a partially-initialized paddle (GPU
                    # wheels are especially prone to circular-import errors here).
                    _paddle = sys.modules.get("paddle")
                    if _paddle is not None and hasattr(_paddle, "set_flags"):
                        _paddle.set_flags({"FLAGS_use_mkldnn": False})
                except Exception:
                    pass

            # Patch paddlepaddle-gpu AnalysisConfig API mismatch.
            # Some GPU wheel builds renamed set_optimization_level →
            # tensorrt_optimization_level, but paddlex still calls the old name.
            # We add the missing method as an alias so paddlex._create() works.
            try:
                _lib = sys.modules.get("paddle.base.libpaddle")
                if _lib is None:
                    import paddle.base.libpaddle as _lib  # type: ignore[no-redef]
                _cfg_cls = getattr(_lib, "AnalysisConfig", None)
                if _cfg_cls is not None and not hasattr(_cfg_cls, "set_optimization_level"):

                    def _set_opt_level(self: Any, level: int) -> None:  # noqa: ANN001
                        """No-op shim: paddle-gpu exposes tensorrt_optimization_level as
                        a read-only C++ binding; the default level is acceptable for
                        inference so we simply swallow the call."""

                    _cfg_cls.set_optimization_level = _set_opt_level  # type: ignore[attr-defined]
                    logger.debug("Patched AnalysisConfig.set_optimization_level → no-op (read-only attr on GPU wheel)")
            except Exception:
                pass

            _paddleocr_module = PaddleOCR
            logger.debug("PaddleOCR loaded successfully")
        except ImportError as exc:
            raise ImportError(
                "paddleocr is required for PaddleOCRAdapter. "
                "Install with: pip install paddleocr paddlepaddle\n"
                "or: pip install datamata[ocr-paddle]\n"
                "Note: paddlepaddle GPU wheel is ~500 MB. Use paddlepaddle-gpu for GPU support."
            ) from exc
    return _paddleocr_module


def _is_paddleocr_v3() -> bool:
    """Return True if PaddleOCR ≥3.x (new API) is installed."""
    try:
        import importlib.metadata

        version = importlib.metadata.version("paddleocr")
        return int(version.split(".")[0]) >= 3
    except Exception:
        return False


def _check_paddle_version_compat() -> None:
    """Warn (or raise) when paddleocr and paddlepaddle major versions mismatch.

    PaddleOCR ≥3.x requires PaddlePaddle ≥3.x.  Installing paddleocr 3.x with
    paddlepaddle 2.x causes native segfaults and missing-symbol ImportErrors
    deep inside paddle's C++ runtime that are otherwise very hard to diagnose.
    """
    try:
        import importlib.metadata

        ocr_ver = importlib.metadata.version("paddleocr")
        ocr_major = int(ocr_ver.split(".")[0])

        # Try both package names (CPU and GPU wheels)
        paddle_ver: str | None = None
        for pkg in ("paddlepaddle-gpu", "paddlepaddle"):
            try:
                paddle_ver = importlib.metadata.version(pkg)
                break
            except importlib.metadata.PackageNotFoundError:
                continue

        if paddle_ver is None:
            return  # can't determine — let the import fail naturally

        paddle_major = int(paddle_ver.split(".")[0])
        if ocr_major >= 3 and paddle_major < 3:
            raise ImportError(
                f"paddleocr {ocr_ver} requires paddlepaddle ≥3.0, "
                f"but paddlepaddle {paddle_ver} is installed.\n"
                f"This version mismatch causes native crashes and missing-symbol errors.\n\n"
                f"Fix — upgrade paddlepaddle to a 3.x GPU wheel that matches your CUDA:\n"
                f"  pip install paddlepaddle-gpu==3.0.0 "
                f"-i https://www.paddlepaddle.org.cn/packages/stable/cu126/\n"
                f"Available CUDA variants: cu118, cu123, cu126\n"
                f"See https://www.paddlepaddle.org.cn/install/quick for more options."
            )

        # Also check for mismatched CPU + GPU wheel versions (Bug 4, Windows).
        # When both paddlepaddle (CPU base) and paddlepaddle-gpu are installed,
        # they must be the SAME version.  A mismatch causes the GPU's common.dll
        # to be missing exports that the CPU's phi.dll expects, producing:
        #   OSError: [WinError 127] The specified procedure could not be found
        if sys.platform == "win32":
            try:
                cpu_ver = importlib.metadata.version("paddlepaddle")
                gpu_ver = importlib.metadata.version("paddlepaddle-gpu")
                if cpu_ver != gpu_ver:
                    raise ImportError(
                        f"paddlepaddle (CPU base) {cpu_ver} and "
                        f"paddlepaddle-gpu {gpu_ver} are mismatched.\n"
                        f"Their shared DLLs (phi.dll, common.dll) must come from "
                        f"the same version to avoid [WinError 127] DLL load failures.\n\n"
                        f"Fix — upgrade paddlepaddle-gpu to match the installed base:\n"
                        f"  pip install paddlepaddle-gpu=={cpu_ver} "
                        f"-i https://www.paddlepaddle.org.cn/packages/stable/cu126/\n"
                        f"Available CUDA variants: cu118, cu123, cu126\n"
                        f"See https://www.paddlepaddle.org.cn/install/quick for more options."
                    )
            except importlib.metadata.PackageNotFoundError:
                pass  # one or both packages not installed — skip check
    except ImportError:
        raise
    except Exception:
        pass  # non-critical check — do not block loading


def _check_paddle_windows_cudnn_compat() -> None:
    """Detect torch/paddle cuDNN DLL version conflict on Windows (Bug 5).

    torch 2.x (+cu12x) bundles cuDNN 9.1.x DLLs while paddlepaddle-gpu ≥3.3.0
    requires nvidia-cudnn-cu12==9.5.1.17 (cuDNN 9.5).  Their ``cudnn_graph64_9.dll``
    export tables differ by 5–10 symbols in each direction, making them mutually
    exclusive inside a single process.  LoadLibraryExW(0x1100) returns
    WinError 127 "procedure not found" for whichever framework is loaded second.

    This function does a *dry-run* import of paddle to detect the error before
    paddleocr is fully loaded, so the user gets a clear diagnosis.  It is a
    no-op when paddle is already in sys.modules, when torch is absent, or on
    non-Windows platforms.
    """
    if "paddle" in sys.modules:
        return  # paddle already loaded — conflict window has passed
    if "torch" not in sys.modules:
        return  # torch not loaded yet — no conflict possible
    try:
        import paddle  # noqa: F401  # trigger the DLL load now
    except OSError as exc:
        if getattr(exc, "winerror", None) == 127:
            _torch = sys.modules.get("torch")
            _tv = getattr(_torch, "__version__", "?") if _torch else "?"
            raise ImportError(
                "PaddleOCR cannot initialize: paddle failed to load its GPU "
                "CUDA libraries (WinError 127 — 'procedure not found').\n\n"
                "Root cause (Windows only): torch and paddlepaddle-gpu ship\n"
                "INCOMPATIBLE cuDNN versions that cannot share a process:\n"
                f"  torch {_tv} bundles cuDNN ≈9.1.x  (in torch/lib/cudnn_graph64_9.dll)\n"
                "  paddlepaddle-gpu ≥3.3.0 needs cuDNN 9.5  (nvidia-cudnn-cu12==9.5.1.17)\n"
                "  The two cudnn_graph64_9.dll builds each expose 5–10 symbols the\n"
                "  other does not, so only ONE can be active per process.\n\n"
                "Fixes:\n"
                "  1. Upgrade torch to ≥2.6.0 — it bundles cuDNN ≥9.5.0 which\n"
                "     matches what paddle requires:\n"
                "       pip install torch --upgrade\n"
                "  2. Use CPU-only paddle (no CUDA DLLs, no conflict):\n"
                "       pip install paddlepaddle  # CPU-only wheel\n"
                "  3. Run OCR in a separate subprocess or virtual environment.\n"
            ) from exc
        raise  # re-raise other OSErrors (unexpected)
    except Exception:
        pass  # non-OSError failures are handled later in _ensure_paddleocr()


def _apply_paddle_base_init_patch() -> None:
    """Remove the redundant monkey_patch_variable() call from paddle/base/__init__.py.

    paddlepaddle-gpu 3.0.0 calls monkey_patch_variable() in TWO places:
      1. paddle/base/__init__.py  (line ~201) — executed first, as a side-effect
         of ``from .base import core`` at paddle/__init__.py line 40.
      2. paddle/__init__.py        (line ~47)  — the intentional call.

    The first call sets ``_already_patch_variable = True``.  When the second
    call executes, it takes the ``else:`` branch inside monkey_patch_variable(),
    which runs ``import paddle.tensor``.  At that point paddle.tensor is not
    yet in sys.modules, producing:
        AttributeError: partially initialized module 'paddle' has no attribute 'tensor'

    Fix: remove the premature call from paddle/base/__init__.py.  The call in
    paddle/__init__.py line 47 is the authoritative one; it runs after enough
    of paddle's own namespace is set up.  Removing the duplicate ensures the
    flag is still False when line 47 executes, so it safely takes the
    ``if not _already_patch_variable:`` branch.

    This patch is idempotent (guarded by a MATA_FIX marker).
    """
    try:
        import importlib.util

        # Locate paddle's root __init__.py without importing paddle.base
        # (using find_spec("paddle.base") could trigger the very circular import
        # we are trying to fix).
        spec = importlib.util.find_spec("paddle")
        if spec is None or not spec.origin:
            return
        base_init = Path(spec.origin).parent / "base" / "__init__.py"
        if not base_init.exists():
            return

        content = base_init.read_text(encoding="utf-8")
        _marker = "# MATA_FIX: removed dup call — paddle/__init__.py:47 is authoritative"
        if _marker in content:
            return  # already patched

        # Comment out the standalone monkey_patch_variable() call.
        # We match the exact surrounding context to avoid replacing any other
        # occurrence (e.g. an import statement or a different function call).
        _target = "\nmonkey_patch_variable()\n"
        if _target not in content:
            logger.debug(
                "Could not locate monkey_patch_variable() call in %s — skipping patch",
                base_init,
            )
            return

        patched = content.replace(
            _target,
            f"\n{_marker}\n# monkey_patch_variable()  <- called by paddle/__init__.py instead\n",
            1,  # first occurrence only
        )
        base_init.write_text(patched, encoding="utf-8")
        logger.debug(
            "Patched %s: commented out redundant monkey_patch_variable() call",
            base_init,
        )

        # Invalidate .pyc so Python re-reads the patched source on next import.
        cache_path = importlib.util.cache_from_source(str(base_init))
        _cache = Path(cache_path)
        if _cache.exists():
            try:
                _cache.unlink()
                logger.debug("Removed stale .pyc: %s", _cache)
            except OSError:
                pass  # not critical

    except Exception as exc:
        logger.debug("paddle/base __init__ patch failed (non-fatal): %s", exc)


def _paddle_polygon_to_xyxy(polygon: list) -> tuple[float, float, float, float]:
    """Convert PaddleOCR [[x,y],...] polygon (4 points) to xyxy bbox.

    PaddleOCR returns bounding boxes as a list of 4 corner points
    (clockwise from top-left). This converts to axis-aligned xyxy format by
    taking the min/max of all point coordinates.

    Args:
        polygon: List of [x, y] points (4 points, may also be numpy arrays)

    Returns:
        Tuple (x_min, y_min, x_max, y_max) in absolute pixel coordinates

    Example:
        >>> _paddle_polygon_to_xyxy([[10, 20], [100, 20], [100, 50], [10, 50]])
        (10.0, 20.0, 100.0, 50.0)
    """
    xs = [float(pt[0]) for pt in polygon]
    ys = [float(pt[1]) for pt in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


class PaddleOCRAdapter(BaseAdapter):
    """OCR adapter wrapping the PaddleOCR engine.

    PaddleOCR is a practical, ultra-lightweight OCR toolkit supporting 80+
    languages. It returns text regions as polygon bounding boxes (4 corner
    points) which are converted to xyxy format for consistency with MATA's
    ``TextRegion.bbox`` contract.

    **PaddleOCR ≥3.x (new API):** uses ``predict()`` and returns a list of dicts::

        [{'rec_texts': [...], 'rec_scores': [...], 'dt_polys': [...]}]

    **PaddleOCR <3.x (legacy API):** uses ``ocr()`` and returns a nested list::

        [  # page list (1 element per image)
            [  # line list
                [bbox_polygon, (text, confidence)],
                ...
            ]
        ]

    Both formats are handled transparently by this adapter.

    Note:
        PaddleOCR ≥3.x uses ``device="gpu"/"cpu"``; older versions use
        ``use_gpu=True/False``. The adapter detects which API is available and
        passes the appropriate argument automatically.

        ``paddlepaddle`` (CPU) or ``paddlepaddle-gpu`` (GPU) must be installed
        in addition to ``paddleocr``. The GPU wheel is approximately 500 MB.

    Args:
        lang: Language code for recognition (default: ``"en"``).
            See PaddleOCR documentation for supported languages.
        use_gpu: Whether to use GPU acceleration (default: ``False``).
            Mapped to ``device="gpu"/"cpu"`` automatically for PaddleOCR ≥3.x.
            Defaults to ``False`` because requesting GPU when it is unavailable
            causes PaddlePaddle to fall back to CPU with MKL-DNN re-enabled,
            which triggers a runtime error on Windows with PaddlePaddle ≥3.x.
        **kwargs: Additional keyword arguments forwarded to ``PaddleOCR()``.

    Example::

        adapter = PaddleOCRAdapter(lang="en")
        result = adapter.predict(pil_image)
        print(result.full_text)
    """

    name = "paddleocr"
    task = "ocr"

    def __init__(self, lang: str = "en", use_gpu: bool = False, **kwargs: Any) -> None:
        super().__init__()
        self.lang = lang
        self.use_gpu = use_gpu
        self._kwargs = kwargs
        paddleocr_cls = _ensure_paddleocr()
        logger.debug("Initializing PaddleOCR (lang=%s, use_gpu=%s)", lang, use_gpu)
        # PaddleOCR ≥3.x removed ``use_gpu`` and ``show_log`` in favour of
        # ``device``.  Detect the installed version and call the right API.
        if _is_paddleocr_v3():
            # New API: only whitelisted kwargs accepted by parse_common_args.
            # Strip any legacy-only kwargs (show_log, use_gpu) that callers
            # may have forwarded; retain the rest (device, enable_hpi, …).
            _legacy_keys = {"show_log", "use_gpu"}
            new_kwargs = {k: v for k, v in kwargs.items() if k not in _legacy_keys}
            device = "gpu" if use_gpu else "cpu"
            # Disable MKL-DNN by default on Windows to avoid a known OneDNN
            # runtime crash (ConvertPirAttribute2RuntimeAttribute) in
            # PaddlePaddle ≥3.x.  Users can opt back in via enable_mkldnn=True.
            if sys.platform == "win32" and "enable_mkldnn" not in new_kwargs:
                new_kwargs.setdefault("enable_mkldnn", False)
            self._ocr = paddleocr_cls(lang=lang, device=device, **new_kwargs)
        else:
            # Legacy API accepts ``use_gpu`` and ``show_log``.
            _new_keys = {
                "device",
                "enable_hpi",
                "use_tensorrt",
                "precision",
                "enable_mkldnn",
                "mkldnn_cache_capacity",
                "cpu_threads",
                "enable_cinn",
            }
            old_kwargs = {k: v for k, v in kwargs.items() if k not in _new_keys}
            self._ocr = paddleocr_cls(lang=lang, use_gpu=use_gpu, show_log=False, **old_kwargs)

    def predict(self, image: Any, **kwargs: Any) -> OCRResult:
        """Run OCR on an image and return structured text results.

        Args:
            image: Input image. Accepts PIL Image, numpy array, file path, or
                URL. Converted internally to a numpy array for PaddleOCR.
            **kwargs: Additional keyword arguments forwarded to
                ``PaddleOCR.ocr()``, such as ``cls=True`` for text-angle
                classification.

        Returns:
            :class:`~mata.core.types.OCRResult` containing detected
            :class:`~mata.core.types.TextRegion` instances with xyxy bboxes.
            Returns an empty ``OCRResult`` when no text is detected.
        """
        import numpy as np

        pil_img, _ = self._load_image(image)
        img_array = np.array(pil_img)

        logger.debug("Running PaddleOCR prediction on image shape=%s", img_array.shape)

        # Strip constructor-only kwargs that must not be forwarded to the
        # underlying engine's inference call (e.g. lang / use_gpu arriving
        # via mata.run() which forwards all kwargs to both load() and predict()).
        _constructor_keys = {
            "lang",
            "use_gpu",
            "show_log",
            "device",
            "enable_hpi",
            "use_tensorrt",
            "precision",
            "enable_mkldnn",
            "mkldnn_cache_capacity",
            "cpu_threads",
            "enable_cinn",
        }
        predict_kwargs = {k: v for k, v in kwargs.items() if k not in _constructor_keys}

        regions: list[TextRegion] = []

        # PaddleOCR ≥3.x uses predict() with a dict-based output format.
        # Older versions use ocr() with a nested-list output format.
        if _is_paddleocr_v3():
            # New API (PaddleOCR ≥3.x)
            # Output: [{'rec_texts': [...], 'rec_scores': [...], 'dt_polys': [...]}]
            raw = self._ocr.predict(img_array, **predict_kwargs)
            if raw:
                for page in raw:
                    if page is None:
                        continue
                    texts = page.get("rec_texts") or []
                    scores = page.get("rec_scores") or []
                    polys = page.get("dt_polys") or []
                    for text, confidence, polygon in zip(texts, scores, polys):
                        regions.append(
                            TextRegion(
                                text=text,
                                score=float(confidence),
                                bbox=_paddle_polygon_to_xyxy(polygon),
                            )
                        )
        else:
            # Legacy API (PaddleOCR <3.x)
            # Output: [[[bbox_polygon, (text, confidence)], ...]] or None
            raw = self._ocr.ocr(img_array, **predict_kwargs)
            if raw and raw[0]:
                for line in raw[0]:
                    if line is None:
                        continue
                    polygon, (text, confidence) = line
                    regions.append(
                        TextRegion(
                            text=text,
                            score=float(confidence),
                            bbox=_paddle_polygon_to_xyxy(polygon),
                        )
                    )

        logger.debug("PaddleOCR detected %d text regions", len(regions))
        return OCRResult(regions=regions, meta={"engine": "paddleocr", "lang": self.lang})

    def info(self) -> dict[str, Any]:
        """Return adapter metadata.

        Returns:
            Dictionary with ``name``, ``task``, ``lang``, and ``use_gpu`` keys.
        """
        return {
            "name": self.name,
            "task": self.task,
            "lang": self.lang,
            "use_gpu": self.use_gpu,
        }
