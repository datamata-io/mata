(function () {
  "use strict";

  var MIN_DRAW_SIZE = 6;
  var HANDLE_SIZE = 10;
  var VERTEX_HANDLE_SIZE = 8;
  var UNDO_LIMIT = 50;
  var SAVE_DEBOUNCE_MS = 2000;
  var CATEGORY_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
  ];
  var RESIZE_CURSOR = {
    n: "ns-resize",
    s: "ns-resize",
    e: "ew-resize",
    w: "ew-resize",
    ne: "nesw-resize",
    sw: "nesw-resize",
    nw: "nwse-resize",
    se: "nwse-resize",
  };

  // D7: Auto-annotate mode definitions
  var AUTO_ANNOTATE_MODES = [
    {
      id: "detect",
      label: "Detection",
      promptLabel: "Threshold",
      promptDefault: "0.30",
      promptType: "number",
      endpoint: "/api/assist/auto-annotate",
    },
    {
      id: "vlm",
      label: "VLM",
      promptLabel: "VLM Prompt",
      promptDefault: "Detect all objects in this image.",
      promptType: "textarea",
      endpoint: "/api/assist/vlm",
    },
    {
      id: "clip",
      label: "CLIP Classify",
      promptLabel: "Class Names (comma-separated)",
      promptDefault: "",
      promptType: "textarea",
      endpoint: "/api/assist/classify",
    },
    {
      id: "zeroshot",
      label: "Zero-Shot Detection",
      promptLabel: "Objects to detect (comma-separated)",
      promptDefault: "cat, dog, person",
      promptType: "textarea",
      endpoint: "/api/assist/zeroshot-detect",
    },
  ];

  var state = {
    datasets: [],
    rescanStatus: {},
    datasetStats: null,
    images: [],
    coco: null,
    classes: [],
    selectedDataset: null,
    selectedDatasetType: null,
    selectedImageIndex: -1,
    selectedAnnotationId: null,
    selectedAnnotationIds: [], // multi-select list (superset of selectedAnnotationId)
    activeTool: "select",
    polygonVertices: [],
    lastPolygonClickTime: 0,
    interaction: null,
    pendingAnnotation: null,
    undoStack: [],
    lastUsedCategoryId: null,
    editorNavSplit: null, // split context for editor prev/next navigation
    dirty: false,
    isSaving: false,
    saveTimer: null,
    pendingSave: false,
    view: {
      imageWidth: 0,
      imageHeight: 0,
      canvasWidth: 0,
      canvasHeight: 0,
      scaleX: 1,
      scaleY: 1,
    },
  };

  // ── Browser view state ─────────────────────────────────────────────────────
  var browserState = {
    datasets: [],
    datasetStats: null,
    pagination: {
      page: 1,
      perPage: 50,
      sort: "name_asc",
      split: null,
      annotated: null,
      search: "",
      total: 0,
      totalPages: 0,
    },
    thumbnailCache: {},
    scrollTop: 0,
  };

  // ── D4: Tool palette definitions ──────────────────────────────────────────────
  var TOOLS = [
    { id: "select", shortcut: "V", active: true, label: "Select / Move" },
    { id: "bbox", shortcut: "B", active: true, label: "Bounding Box" },
    { id: "polygon", shortcut: "P", active: true, label: "Polygon" },
    { id: "polyline", shortcut: "L", active: false, label: "Polyline" },
    { id: "rotate", shortcut: "R", active: false, label: "Rotate / Transform" },
    { id: "ai", shortcut: "A", active: true, label: "AI Auto-detect" },
    { id: "split", shortcut: "X", active: false, label: "Split Polygon" },
    { id: "merge", shortcut: "M", active: false, label: "Merge Annotations" },
    "separator",
    { id: "undo", shortcut: "Ctrl+Z", active: true, label: "Undo" },
    { id: "redo", shortcut: "Ctrl+Y", active: false, label: "Redo" },
    "separator",
    { id: "delete", shortcut: "Del", active: true, label: "Delete Selected" },
  ];

  // ── Editor view state ────────────────────────────────────────────────────────
  var editorState = {
    coco: null,
    images: [],
    classes: [],
    selectedAnnotationId: null,
    activeTool: "select",
    zoom: 1,
    panX: 0,
    panY: 0,
    brightness: 100, // D8: CSS brightness filter (%)
    contrast: 100, // D8: CSS contrast filter (%)
    panMode: false, // D8: true while Space key held (enables drag-to-pan)
    undoStack: [],
    dirty: false,
    // D3: left panel tab state (persists during image navigation)
    leftPanelTab: "labels", // "labels" | "attributes" | "raw"
    leftPanelSubTab: "layers", // "layers" | "classes"
    // D7: auto-annotate mode state (persists during image navigation)
    autoAnnotateMode: "detect", // "detect" | "vlm" | "clip" | "zeroshot"
    // D9: annotation visibility toggle
    showAnnotations: true,
  };

  // ── Browser View ──────────────────────────────────────────────────────────────
  var BrowserView = {
    /** Called when switching to browse view. Loads datasets if not yet cached. */
    mount: function () {
      var main = document.getElementById("browserMain");
      if (main) {
        main.scrollTop = browserState.scrollTop || 0;
      }
      // Restore search/filter/sort UI to preserved pagination state (C3)
      syncSearchBarUI();
      // Resume training panel polling if a job is active (C4)
      if (window.TrainingPanel) {
        window.TrainingPanel.resume();
      }
      if (state.datasets.length === 0) {
        loadDatasets();
      } else {
        browserState.datasets = state.datasets;
        BrowserView.renderSidebar();
        // Re-fetch the thumbnail grid page when returning from the editor (C3)
        fetchImages();
      }
    },

    /** Called when leaving browse view. Saves scroll position. */
    unmount: function () {
      var main = document.getElementById("browserMain");
      if (main) {
        browserState.scrollTop = main.scrollTop;
      }
      // Pause training panel polling while editor view is active (C4)
      if (window.TrainingPanel) {
        window.TrainingPanel.pause();
      }
    },

    /**
     * Render the browser sidebar: workspace status grid + dataset list with
     * type badges. Called by renderDatasets() and BrowserView.mount().
     */
    renderSidebar: function () {
      // ── Dataset list ───────────────────────────────────────────────────────
      var list = elements.datasetList;
      if (!list) {
        return;
      }
      list.innerHTML = "";
      elements.datasetError.hidden = true;

      if (!state.datasets.length) {
        elements.datasetEmpty.hidden = false;
        elements.datasetEmpty.textContent =
          "No datasets found in the configured data directory.";
        elements.datasetStatus.textContent = "Empty";
        updateCounts();
        return;
      }

      elements.datasetEmpty.hidden = true;
      elements.datasetStatus.textContent = state.datasets.length + " loaded";

      state.datasets.forEach(function (dataset) {
        var button = document.createElement("button");
        var badgeClass = typeBadgeClass(dataset.type);
        var badgeLabel = typeBadgeLabel(dataset.type);
        var isActive = state.selectedDataset === dataset.name;
        var rescanRunning = state.rescanStatus[dataset.name] === "running";
        button.type = "button";
        button.className = "dataset-item" + (isActive ? " active" : "");

        var countHtml;
        if (rescanRunning) {
          countHtml =
            '<span class="dataset-item-count rescan-spinner" title="Scanning…">…</span>';
        } else if (dataset.cache_valid === false) {
          countHtml =
            '<span class="dataset-item-count" title="Count unknown — click Rescan">?</span>' +
            '<button class="rescan-btn" type="button" title="Rescan to count images" data-dataset="' +
            escapeHtml(dataset.name) +
            '">\u21bb</button>';
        } else {
          countHtml =
            '<span class="dataset-item-count">' +
            (dataset.image_count || 0) +
            "</span>";
        }

        button.innerHTML =
          '<span class="dataset-item-name">' +
          escapeHtml(dataset.name) +
          "</span>" +
          '<span class="type-badge ' +
          badgeClass +
          '">' +
          badgeLabel +
          "</span>" +
          countHtml;

        button.addEventListener("click", function (e) {
          // Rescan button inside the dataset row — stop propagation so we
          // don't also trigger selectDataset.
          if (e.target && e.target.classList.contains("rescan-btn")) {
            e.stopPropagation();
            rescanDataset(e.target.getAttribute("data-dataset"));
            return;
          }
          selectDataset(dataset.name);
        });
        list.appendChild(button);
      });

      updateCounts();
    },
  };

  // ── BrowserView: dataset header ──────────────────────────────────────────────
  /**
   * Update the browser view header with dataset name, info line, and the
   * browse-progress bar.  Called after a dataset's stats have been loaded.
   */
  BrowserView.renderHeader = function (stats, datasetName) {
    var nameEl = document.getElementById("browserDatasetName");
    var infoEl = document.getElementById("browserDatasetInfo");
    var fillEl = document.getElementById("browseProgressFill");
    var wrapEl = document.getElementById("browseProgressWrap");

    var name = datasetName || state.selectedDataset || "Select a dataset";
    if (nameEl) {
      nameEl.textContent = name;
    }

    if (stats && infoEl) {
      var parts = [];
      parts.push((stats.image_count || 0).toLocaleString() + " images");
      if (typeof stats.total_annotated === "number") {
        parts.push(stats.total_annotated.toLocaleString() + " annotated");
      }
      if (typeof stats.total_unannotated === "number") {
        parts.push(stats.total_unannotated.toLocaleString() + " unannotated");
      }
      if (stats.folder_path) {
        parts.push(stats.folder_path);
      }
      if (typeof stats.total_size_bytes === "number") {
        parts.push(formatBytes(stats.total_size_bytes));
      }
      infoEl.textContent = parts.join(" \u00b7 ");
    } else if (infoEl) {
      infoEl.textContent =
        "Browse datasets, load existing COCO labels, and annotate images.";
    }

    var progress =
      stats && typeof stats.browse_progress === "number"
        ? stats.browse_progress
        : 0;
    if (fillEl) {
      fillEl.style.width = progress + "%";
    }
    if (wrapEl) {
      wrapEl.hidden = !stats;
    }
  };

  // ── BrowserView: split tabs ──────────────────────────────────────────────────
  /**
   * Update the split-tab count badges and active highlight.  Pins the active
   * tab to browserState.pagination.split (null / "" = "All").
   */
  BrowserView.renderSplitTabs = function (stats) {
    var splits = stats && stats.splits ? stats.splits : {};
    var trainCount = document.getElementById("splitCountTrain");
    var testCount = document.getElementById("splitCountTest");
    var valCount = document.getElementById("splitCountVal");

    if (trainCount) {
      var t = splits.train;
      trainCount.textContent = t ? "(" + (t.total || 0) + ")" : "";
    }
    if (testCount) {
      var ts = splits.test;
      testCount.textContent = ts ? "(" + (ts.total || 0) + ")" : "";
    }
    if (valCount) {
      var v = splits.val;
      valCount.textContent = v ? "(" + (v.total || 0) + ")" : "";
    }

    var activeSplit = browserState.pagination.split || "";
    var tabs = document.querySelectorAll(".split-tab");
    tabs.forEach(function (tab) {
      var tabSplit = tab.dataset.split || "";
      var isActive = tabSplit === activeSplit;
      tab.classList.toggle("active", isActive);
      tab.setAttribute("aria-selected", String(isActive));
    });
  };

  // ── BrowserView: thumbnail grid ──────────────────────────────────────────────
  /**
   * Render paginated image cards into #thumbnailGrid.  Each card shows a
   * lazy-loaded thumbnail, a green ✓ badge when the image has annotations, and
   * the truncated filename.  Clicking a card navigates to the editor at the
   * correct absolute index in state.images.
   */
  BrowserView.renderGrid = function (images) {
    var grid = document.getElementById("thumbnailGrid");
    var placeholder = document.getElementById("browserPlaceholder");
    if (!grid) {
      return;
    }

    grid.innerHTML = "";

    if (!images || !images.length) {
      if (placeholder) {
        placeholder.hidden = false;
      }
      return;
    }

    if (placeholder) {
      placeholder.hidden = true;
    }

    // Track which split the user is currently browsing so the editor
    // prev/next navigation is scoped to this same subset.
    state.editorNavSplit = browserState.pagination.split || null;

    var ds = state.selectedDataset;
    images.forEach(function (img) {
      var imageIndex = findImageIndexByFilename(img.filename);
      if (imageIndex < 0) {
        // Image is not in state.images (e.g. a split sub-directory not covered
        // by the loaded COCO file). Register it on-the-fly so navigation works.
        imageIndex = state.images.length;
        state.images.push({
          filename: img.filename,
          width: img.width || 0,
          height: img.height || 0,
          annotation_count: img.annotation_count || 0,
          split: img.split || null,
        });
      } else {
        // Image already registered (e.g. from COCO). Update the split field
        // so the editor nav filter can correctly scope to the active tab.
        state.images[imageIndex].split = img.split || null;
      }

      var card = document.createElement("div");
      card.className = "thumb-card";
      card.setAttribute("tabindex", "0");
      card.setAttribute("role", "button");
      card.setAttribute("aria-label", img.filename);

      var imgWrap = document.createElement("div");
      imgWrap.className = "thumb-img-wrapper";

      var imgEl = document.createElement("img");
      imgEl.src =
        "/api/datasets/" +
        encodeURIComponent(ds) +
        "/images/" +
        encodeURIComponent(img.filename);
      imgEl.alt = img.filename;
      imgEl.loading = "lazy";
      imgWrap.appendChild(imgEl);

      if (img.annotation_count > 0) {
        var badge = document.createElement("span");
        badge.className = "annotated-badge";
        badge.textContent = "\u2713"; // ✓
        badge.setAttribute("aria-label", "Annotated");
        imgWrap.appendChild(badge);
      }

      var filenameEl = document.createElement("span");
      filenameEl.className = "thumb-filename";
      filenameEl.textContent = img.filename;
      filenameEl.title = img.filename;

      card.appendChild(imgWrap);
      card.appendChild(filenameEl);

      // Capture idx and the current split context in closure so each card
      // navigates to the correct image under the correct split scope.
      (function (idx, splitCtx) {
        card.addEventListener("click", function () {
          state.editorNavSplit = splitCtx;
          Router.navigate("edit", { dataset: ds, imageIndex: idx });
        });
        card.addEventListener("keydown", function (e) {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            state.editorNavSplit = splitCtx;
            Router.navigate("edit", { dataset: ds, imageIndex: idx });
          }
        });
      })(imageIndex, state.editorNavSplit);

      grid.appendChild(card);
    });
  };

  // ── BrowserView: pagination footer ──────────────────────────────────────────
  /**
   * Sync the pagination footer controls (prev / next buttons, "1–50 of N"
   * label) with the current browserState.pagination values.
   */
  BrowserView.renderPagination = function () {
    var infoEl = document.getElementById("paginationInfo");
    var prevBtn = document.getElementById("pagePrevBtn");
    var nextBtn = document.getElementById("pageNextBtn");

    var total = browserState.pagination.total;
    var page = browserState.pagination.page;
    var perPage = browserState.pagination.perPage;
    var totalPages = browserState.pagination.totalPages;

    if (!state.selectedDataset || total === 0) {
      if (infoEl) {
        infoEl.textContent = "\u2014"; // em dash
      }
      if (prevBtn) {
        prevBtn.disabled = true;
      }
      if (nextBtn) {
        nextBtn.disabled = true;
      }
      return;
    }

    var from = (page - 1) * perPage + 1;
    var to = Math.min(page * perPage, total);
    if (infoEl) {
      infoEl.textContent =
        from.toLocaleString() +
        "\u2013" +
        to.toLocaleString() +
        " of " +
        total.toLocaleString();
    }
    if (prevBtn) {
      prevBtn.disabled = page <= 1;
    }
    if (nextBtn) {
      nextBtn.disabled = page >= totalPages;
    }
  };

  // ── Editor View ───────────────────────────────────────────────────────────────
  var EditorView = {
    /** Called when switching to edit view. Loads the dataset and selects image. */
    mount: function (params) {
      params = params || {};
      var dataset = params.dataset;
      var imageIndex =
        typeof params.imageIndex === "number" ? params.imageIndex : 0;
      if (!dataset) {
        return;
      }

      // D2: attach wheel-zoom listener on first mount
      this.initCanvas();

      // Update breadcrumb with the dataset name (D1)
      this.renderTopBar(dataset, imageIndex, state.images.length, "");

      // If the dataset is already loaded jump straight to the image
      if (state.selectedDataset === dataset && state.images.length > 0) {
        selectImage(imageIndex);
        return;
      }

      // Otherwise load the full dataset then jump to the requested image
      selectDataset(dataset, { silentFocus: false }).then(function () {
        if (imageIndex > 0 && imageIndex < state.images.length) {
          selectImage(imageIndex);
        }
      });
    },

    /** Called when leaving edit view. Auto-saves any unsaved work. */
    unmount: function () {
      if (state.dirty) {
        saveAnnotationsNow(true).catch(function () {
          // Auto-save failed; user was warned by Router's confirm dialog.
        });
      }
    },

    /**
     * Render / refresh the editor top bar.
     * Updates: breadcrumb dataset name, filename span, nav counter ("N of M").
     * @param {string} dataset   - Dataset name shown in breadcrumb.
     * @param {number} imageIndex - 0-based index of the current image.
     * @param {number} totalImages - Total number of images in the dataset.
     * @param {string} filename  - Filename of the current image ("" for none).
     */
    renderTopBar: function (dataset, imageIndex, totalImages, filename) {
      var breadcrumb = document.getElementById("editorBreadcrumbDataset");
      if (breadcrumb) {
        breadcrumb.textContent = dataset || "DATASET";
        breadcrumb.title = "Back to " + (dataset || "") + " browser";
      }

      var filenameEl = document.getElementById("editorFilename");
      if (filenameEl) {
        filenameEl.textContent = filename || "No image selected";
        filenameEl.title = filename || "";
      }

      var counter = document.getElementById("editorNavCounter");
      if (counter) {
        var current = totalImages > 0 ? imageIndex + 1 : 0;
        counter.textContent = current + " of " + totalImages;
      }

      // Sync reviewToggleBtn active state
      var reviewBtn = document.getElementById("reviewToggleBtn");
      if (reviewBtn) {
        var imgRecord = getCurrentImageRecord(false);
        var isReviewed = !!(imgRecord && imgRecord.reviewed);
        reviewBtn.classList.toggle("is-active", isReviewed);
        reviewBtn.title = isReviewed ? "Unmark reviewed" : "Mark as reviewed";
      }
    },

    /**
     * Render / refresh the editor bottom bar.
     * Updates the zoom-level percentage label from editorState.zoom.
     */
    renderBottomBar: function () {
      var zl = document.getElementById("zoomLevel");
      if (zl) {
        zl.textContent = Math.round(editorState.zoom * 100) + "%";
      }
    },

    /**
     * D8: Apply brightness + contrast CSS filters to the canvas area wrapper.
     * Reads editorState.brightness and editorState.contrast (0–200%).
     */
    applyBrightness: function () {
      var wrapper = document.getElementById("editorCanvasArea");
      if (wrapper) {
        wrapper.style.filter =
          "brightness(" +
          editorState.brightness +
          "%) contrast(" +
          editorState.contrast +
          "%)";
      }
      // Keep badge in sync
      var badge = document.getElementById("brightnessValueBadge");
      if (badge) {
        badge.textContent = editorState.brightness + "%";
      }
    },

    /**
     * Navigate to the next (+1) or previous (-1) image with wrapping.
     * Auto-saves dirty state before switching.
     * @param {number} offset - +1 for next, -1 for previous.
     */
    navigateImage: function (offset) {
      if (!state.images.length) {
        return;
      }
      // Build a list of image indices scoped to the current split context.
      var navIndices;
      if (state.editorNavSplit) {
        navIndices = [];
        for (var ni = 0; ni < state.images.length; ni++) {
          if ((state.images[ni].split || null) === state.editorNavSplit) {
            navIndices.push(ni);
          }
        }
      }
      if (!navIndices || !navIndices.length) {
        navIndices = state.images.map(function (_, i) {
          return i;
        });
      }
      var pos = navIndices.indexOf(state.selectedImageIndex);
      if (pos < 0) {
        pos = 0;
      }
      var nextPos =
        (((pos + offset) % navIndices.length) + navIndices.length) %
        navIndices.length;
      var next = navIndices[nextPos];

      function doNav() {
        Router.params.imageIndex = next;
        // Update URL hash to keep browser history accurate without re-mounting
        var newHash =
          "#/edit/" +
          encodeURIComponent(state.selectedDataset || "") +
          "/" +
          next;
        window.history.replaceState(null, "", newHash);
        selectImage(next);
      }

      if (state.dirty) {
        saveAnnotationsNow(false).then(doNav).catch(doNav);
      } else {
        doNav();
      }
    },

    // ── D3: Left Panel — Labels Tab with Layers + Classes ────────────────────

    /**
     * D3: Initialise left‑panel interactivity.
     * Wires the main tab buttons (Labels / Attrs / Raw) via event delegation.
     * Call once at app start (before Router.init).
     */
    initLeftPanel: function () {
      var self = this;
      var tabBar = document.querySelector(".left-panel-tabs");
      if (!tabBar) {
        return;
      }
      tabBar.addEventListener("click", function (e) {
        var btn = e.target.closest(".left-panel-tab");
        if (!btn) {
          return;
        }
        var tab = btn.getAttribute("data-tab");
        tabBar.querySelectorAll(".left-panel-tab").forEach(function (t) {
          var isActive = t === btn;
          t.classList.toggle("active", isActive);
          t.setAttribute("aria-selected", isActive ? "true" : "false");
        });
        editorState.leftPanelTab = tab;
        self.renderLeftPanel();
      });
    },

    /**
     * D3: Full re‑render of the left panel content area.
     * Called from renderAll() and on sub‑tab / tab switches.
     */
    renderLeftPanel: function () {
      var container = document.getElementById("leftPanelContent");
      if (!container) {
        return;
      }
      var tab = editorState.leftPanelTab || "labels";
      if (tab === "labels") {
        this.renderLabelsTab(container);
      } else if (tab === "attributes") {
        this.renderAttributesTab(container);
      } else {
        this.renderRawDataTab(container);
      }
    },

    /**
     * D3: Render the Labels tab into the given container element.
     * Contains: annotation count header, Classes/Layers sub-tab toggle,
     * and the active sub‑tab content.
     * @param {HTMLElement} container
     */
    renderLabelsTab: function (container) {
      var annotations = getCurrentAnnotations();
      var categories = state.classes || [];
      var subTab = editorState.leftPanelSubTab || "layers";
      var self = this;

      var html = '<div class="labels-tab-body">';

      // Header: "Annotations N"
      html += '<div class="labels-panel-header">';
      html +=
        'Annotations <span class="labels-ann-count">' +
        annotations.length +
        "</span>";
      html += "</div>";

      // Sub-tab toggle bar
      html += '<div class="labels-subtab-bar" id="labelsSubtabBar">';
      html +=
        '<button class="labels-subtab' +
        (subTab === "layers" ? " active" : "") +
        '" data-subtab="layers">Layers</button>';
      html +=
        '<button class="labels-subtab' +
        (subTab === "classes" ? " active" : "") +
        '" data-subtab="classes">Classes</button>';
      html += "</div>";

      // Sub-tab content (scrollable)
      html += '<div class="labels-tab-scroll" id="labelsTabScroll">';
      if (subTab === "layers") {
        html += this._buildLayersHTML(annotations, categories);
      } else {
        html += this._buildClassesHTML(annotations, categories);
      }
      html += "</div>";
      // D5: annotation properties panel (shown below layers list)
      html += '<div id="annotPropsPanel" class="props-panel-wrap"></div>';

      // D7: Auto-annotate section (always shown at bottom of Labels tab)
      html += this._buildAutoAnnotateHTML(categories);

      html += "</div>";

      container.innerHTML = html;
      // D5: populate annotation properties panel
      this.renderAnnotationProperties();

      // Wire sub-tab button clicks
      var subtabBar = container.querySelector("#labelsSubtabBar");
      if (subtabBar) {
        subtabBar.addEventListener("click", function (e) {
          var btn = e.target.closest(".labels-subtab");
          if (!btn) {
            return;
          }
          editorState.leftPanelSubTab = btn.getAttribute("data-subtab");
          self.renderLabelsTab(container);
        });
      }

      // Wire classes CRUD (classes sub-tab only)
      if (subTab !== "layers") {
        this._initClassesListeners(container);
      }

      // Wire layer row clicks with Ctrl/Shift multi-select (layers sub‑tab only)
      if (subTab === "layers") {
        var scroll = container.querySelector("#labelsTabScroll");
        if (scroll) {
          scroll.addEventListener("click", function (e) {
            var row = e.target.closest(".annotation-row");
            // Batch action bar buttons are wired elsewhere
            if (!row) return;
            var annId = parseInt(row.getAttribute("data-ann-id"), 10);
            if (isNaN(annId)) return;

            if (e.ctrlKey || e.metaKey) {
              // Ctrl+Click: toggle this id in selectedAnnotationIds
              var idx = state.selectedAnnotationIds.indexOf(annId);
              if (idx === -1) {
                state.selectedAnnotationIds = state.selectedAnnotationIds.concat([annId]);
              } else {
                state.selectedAnnotationIds = state.selectedAnnotationIds.filter(function (id) { return id !== annId; });
              }
              state.selectedAnnotationId = annId;
            } else if (e.shiftKey && state.selectedAnnotationId !== null) {
              // Shift+Click: range-select from last primary to this row
              var annotations = getCurrentAnnotations();
              var fromIdx = annotations.findIndex(function (a) { return a.id === state.selectedAnnotationId; });
              var toIdx = annotations.findIndex(function (a) { return a.id === annId; });
              if (fromIdx !== -1 && toIdx !== -1) {
                var lo = Math.min(fromIdx, toIdx);
                var hi = Math.max(fromIdx, toIdx);
                var rangeIds = annotations.slice(lo, hi + 1).map(function (a) { return a.id; });
                // Merge with existing selection
                var merged = state.selectedAnnotationIds.slice();
                rangeIds.forEach(function (id) {
                  if (merged.indexOf(id) === -1) merged.push(id);
                });
                state.selectedAnnotationIds = merged;
              }
              state.selectedAnnotationId = annId;
            } else {
              // Plain click: single select
              state.selectedAnnotationId = annId;
              state.selectedAnnotationIds = [annId];
            }

            EditorView._syncBatchBar(container);
            // If attrs tab is active, refresh it
            if (editorState.leftPanelTab === "attributes") {
              EditorView.renderAttributesTab(container);
            }
            renderClasses();
            renderCanvas();
            updateSelectedLabel();
          });
        }

        // Batch action bar: Delete
        var batchDeleteBtn = container.querySelector("#batchDeleteBtn");
        if (batchDeleteBtn) {
          batchDeleteBtn.addEventListener("click", function () {
            var ids = state.selectedAnnotationIds.slice();
            if (!ids.length) return;
            if (!confirm("Delete " + ids.length + " annotation(s)?")) return;
            pushUndoState();
            ids.forEach(function (id) {
              state.coco.annotations = (state.coco.annotations || []).filter(function (a) { return a.id !== id; });
            });
            state.selectedAnnotationIds = [];
            state.selectedAnnotationId = null;
            scheduleSave();
            EditorView.renderLabelsTab(container);
            renderClasses();
            renderCanvas();
          });
        }

        // Batch action bar: Change Class
        var batchClassBtn = container.querySelector("#batchClassBtn");
        if (batchClassBtn) {
          batchClassBtn.addEventListener("click", function () {
            var ids = state.selectedAnnotationIds.slice();
            if (!ids.length || !state.classes.length) return;
            var options = state.classes.map(function (c, i) { return (i + 1) + ": " + c.name; }).join("\n");
            var input = prompt("Change class for " + ids.length + " annotation(s).\n" + options + "\n\nEnter class number:");
            if (!input) return;
            var choice = parseInt(input, 10) - 1;
            if (isNaN(choice) || choice < 0 || choice >= state.classes.length) return;
            var newCatId = state.classes[choice].id;
            pushUndoState();
            (state.coco.annotations || []).forEach(function (a) {
              if (ids.indexOf(a.id) !== -1) a.category_id = newCatId;
            });
            scheduleSave();
            EditorView.renderLabelsTab(container);
            renderClasses();
            renderCanvas();
          });
        }
      }

      // D7: Wire auto-annotate controls
      this._initAutoAnnotateListeners(container, categories);
    },

    /**
     * D3: Build HTML string for the Layers sub‑tab (annotation list + batch bar).
     * @private
     */
    _buildLayersHTML: function (annotations, categories) {
      if (!annotations.length) {
        return '<p class="hint">No annotations yet. Use the tools to draw boxes.</p>';
      }
      var selectedIds = state.selectedAnnotationIds || [];
      var html = '<div class="annotation-list">';
      annotations.forEach(function (ann, idx) {
        var cat =
          categories.find(function (c) {
            return c.id === ann.category_id;
          }) || null;
        var color = getCategoryColor(ann.category_id);
        var label = cat ? escapeHtml(cat.name) : "Category " + ann.category_id;
        var isSelected = selectedIds.indexOf(ann.id) !== -1 ||
            (selectedIds.length === 0 && ann.id === state.selectedAnnotationId);
        html +=
          '<div class="annotation-row' +
          (isSelected ? " selected" : "") +
          '" data-ann-id="' +
          ann.id +
          '" role="button" tabindex="0">';
        html +=
          '<span class="annotation-color-swatch" style="background:' +
          color +
          '"></span>';
        html += '<span class="annotation-label">' + label + "</span>";
        html += '<span class="annotation-id">#' + (idx + 1) + "</span>";
        html += "</div>";
      });
      html += "</div>";
      // Batch action bar (shown whenever >1 annotation is selected)
      var multiCount = selectedIds.length;
      html += '<div class="batch-action-bar" id="batchActionBar"' +
        (multiCount > 1 ? '' : ' hidden') + '>';
      html += '<span class="batch-action-label">' + multiCount + ' selected</span>';
      html += '<button type="button" class="batch-action-btn" id="batchDeleteBtn" title="Delete selected">Delete</button>';
      html += '<button type="button" class="batch-action-btn" id="batchClassBtn" title="Change class for selected">Class…</button>';
      html += '</div>';
      return html;
    },

    /**
     * D3: Build HTML string for the Classes sub‑tab (category color legend + CRUD).
     * @private
     */
    _buildClassesHTML: function (annotations, categories) {
      // Header row with class count and Add button
      var html = '<div class="class-legend-header">';
      html += '<span class="class-legend-title">Classes <span class="class-legend-total">' + categories.length + '</span></span>';
      html += '<button type="button" class="class-add-btn" id="classAddBtn" title="Add new class">+</button>';
      html += '</div>';

      if (!categories.length) {
        html += '<p class="hint">No classes yet — click + to add one.</p>';
        return html;
      }
      // Count annotations per category for the current image
      var counts = {};
      annotations.forEach(function (ann) {
        counts[ann.category_id] = (counts[ann.category_id] || 0) + 1;
      });
      html += '<div class="class-legend-list">';
      categories.forEach(function (cat) {
        var color = getCategoryColor(cat.id);
        var count = counts[cat.id] || 0;
        html += '<div class="class-legend-row" data-cat-id="' + cat.id + '">';
        html += '<span class="class-color-swatch class-color-btn" data-cat-id="' + cat.id + '" style="background:' + color + '" title="Change color" role="button" tabindex="0"></span>';
        html += '<span class="class-legend-name">' + escapeHtml(cat.name) + '</span>';
        html += '<span class="class-legend-count">' + count + '</span>';
        html += '<button type="button" class="class-crud-btn class-edit-btn" data-cat-id="' + cat.id + '" title="Rename class">&#9998;</button>';
        html += '<button type="button" class="class-crud-btn class-delete-btn" data-cat-id="' + cat.id + '" title="Delete class">&#128465;</button>';
        html += '</div>';
      });
      html += '</div>';
      return html;
    },

    /**
     * D3: Wire Classes sub-tab CRUD controls (add / rename / delete / recolor).
     * Called from renderLabelsTab whenever the classes sub-tab is shown.
     * @private
     */
    _initClassesListeners: function (container) {
      var ds = state.selectedDataset;
      if (!ds) return;

      function jsonFetch(url, method, body) {
        return fetch(url, {
          method: method,
          headers: { Accept: "application/json", "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }).then(function (resp) {
          return resp.json().then(function (data) {
            if (!resp.ok) throw new Error(data && data.error ? data.error : "Request failed");
            return data;
          });
        });
      }

      // Add class
      var addBtn = container.querySelector("#classAddBtn");
      if (addBtn) {
        addBtn.addEventListener("click", function () {
          var name = prompt("New class name:");
          if (!name || !name.trim()) return;
          jsonFetch("/api/datasets/" + encodeURIComponent(ds) + "/categories", "POST", { name: name.trim() })
            .then(function (data) {
              if (data.category) {
                state.coco.categories.push(data.category);
                refreshClasses();
                EditorView.renderLabelsTab(container);
                renderClasses();
              }
            })
            .catch(function (err) { alert("Error adding class: " + err.message); });
        });
      }

      // Rename class (pencil)
      container.querySelectorAll(".class-edit-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
          var catId = parseInt(btn.getAttribute("data-cat-id"), 10);
          var cat = state.classes.find(function (c) { return c.id === catId; });
          if (!cat) return;
          var name = prompt("Rename class:", cat.name);
          if (!name || !name.trim() || name.trim() === cat.name) return;
          jsonFetch("/api/datasets/" + encodeURIComponent(ds) + "/categories/" + catId, "PUT", { name: name.trim() })
            .then(function () {
              cat.name = name.trim();
              refreshClasses();
              EditorView.renderLabelsTab(container);
              renderClasses();
            })
            .catch(function (err) { alert("Error renaming class: " + err.message); });
        });
      });

      // Delete class (trash)
      container.querySelectorAll(".class-delete-btn").forEach(function (btn) {
        btn.addEventListener("click", function () {
          var catId = parseInt(btn.getAttribute("data-cat-id"), 10);
          var cat = state.classes.find(function (c) { return c.id === catId; });
          if (!cat) return;
          var annCount = (state.coco.annotations || []).filter(function (a) { return a.category_id === catId; }).length;
          var msg = "Delete class \"" + cat.name + "\"?";
          if (annCount > 0) msg += "\n" + annCount + " annotation(s) on this image will also be removed.";
          if (!confirm(msg)) return;
          jsonFetch("/api/datasets/" + encodeURIComponent(ds) + "/categories/" + catId, "DELETE", {})
            .then(function () {
              state.coco.categories = state.coco.categories.filter(function (c) { return c.id !== catId; });
              state.coco.annotations = (state.coco.annotations || []).filter(function (a) { return a.category_id !== catId; });
              if (state.selectedAnnotationId && !(state.coco.annotations || []).find(function (a) { return a.id === state.selectedAnnotationId; })) {
                state.selectedAnnotationId = null;
              }
              refreshClasses();
              EditorView.renderLabelsTab(container);
              renderClasses();
              renderCanvas();
            })
            .catch(function (err) { alert("Error deleting class: " + err.message); });
        });
      });

      // Recolor class (color swatch click → native color picker)
      container.querySelectorAll(".class-color-btn").forEach(function (swatch) {
        swatch.addEventListener("click", function () {
          var catId = parseInt(swatch.getAttribute("data-cat-id"), 10);
          var cat = state.classes.find(function (c) { return c.id === catId; });
          if (!cat) return;
          var currentColor = cat.color || getCategoryColor(catId);
          var inp = document.createElement("input");
          inp.type = "color";
          inp.value = currentColor;
          inp.style.cssText = "position:absolute;opacity:0;width:0;height:0;";
          document.body.appendChild(inp);
          inp.addEventListener("change", function () {
            var newColor = inp.value;
            jsonFetch("/api/datasets/" + encodeURIComponent(ds) + "/categories/" + catId, "PUT", { color: newColor })
              .then(function () {
                cat.color = newColor;
                refreshClasses();
                EditorView.renderLabelsTab(container);
                renderClasses();
                renderCanvas();
              })
              .catch(function (err) { alert("Error updating color: " + err.message); });
            if (inp.parentNode) inp.parentNode.removeChild(inp);
          });
          inp.addEventListener("blur", function () {
            setTimeout(function () { if (inp.parentNode) inp.parentNode.removeChild(inp); }, 300);
          });
          inp.click();
        });
      });
    },

    /**
     * D3: Update the batch action bar count label and visibility.
     * Called after selectedAnnotationIds changes without a full tab re-render.
     */
    _syncBatchBar: function (container) {
      var bar = container ? container.querySelector("#batchActionBar") : document.getElementById("batchActionBar");
      if (!bar) return;
      var count = (state.selectedAnnotationIds || []).length;
      bar.hidden = count <= 1;
      var label = bar.querySelector(".batch-action-label");
      if (label) label.textContent = count + " selected";
    },

    /**
     * D3: Lightweight layer‑highlight sync.
     * Toggles the .selected class on existing annotation rows without
     * rebuilding the whole panel. Called from renderCanvas() for
     * canvas→panel bidirectional selection sync.
     */
    syncLayersHighlight: function () {
      if ((editorState.leftPanelTab || "labels") !== "labels") {
        return;
      }
      if ((editorState.leftPanelSubTab || "layers") !== "layers") {
        return;
      }
      var selectedId = state.selectedAnnotationId;
      var scroll = document.getElementById("labelsTabScroll");
      if (!scroll) {
        return;
      }
      scroll.querySelectorAll(".annotation-row").forEach(function (row) {
        var annId = parseInt(row.getAttribute("data-ann-id"), 10);
        var multiSelected = (state.selectedAnnotationIds || []).indexOf(annId) !== -1;
        var shouldSelect = multiSelected || (!multiSelected && !state.selectedAnnotationIds.length && annId === selectedId);
        var isSelected = row.classList.contains("selected");
        if (shouldSelect !== isSelected) {
          row.classList.toggle("selected", shouldSelect);
          if (shouldSelect) {
            row.scrollIntoView({ block: "nearest" });
          }
        }
      });
    },

    /**
     * D9: Render the Attributes tab into *container*.
     * Shows per-annotation key-value metadata with add/edit/delete.
     * Shows a placeholder when no annotation is selected.
     * @param {HTMLElement} container
     */
    renderAttributesTab: function (container) {
      var ann = getSelectedAnnotation();
      var self = this;

      if (!ann) {
        container.innerHTML =
          '<p class="hint">Select an annotation to view its attributes.</p>';
        return;
      }

      var attrs = ann.attributes || {};
      var keys = Object.keys(attrs);
      var html = '<div class="attr-panel">';
      html += '<div class="attr-list" id="attrList">';
      keys.forEach(function (key) {
        var escapedKey = escapeHtml(key);
        var escapedVal = escapeHtml(String(attrs[key]));
        html += '<div class="attr-row" data-key="' + escapedKey + '">';
        html +=
          '<input class="attr-key" value="' +
          escapedKey +
          '" readonly aria-label="Attribute key" />';
        html +=
          '<input class="attr-value" value="' +
          escapedVal +
          '" aria-label="Attribute value for ' +
          escapedKey +
          '" />';
        html +=
          '<button class="icon-btn attr-delete" data-key="' +
          escapedKey +
          '" title="Delete attribute" aria-label="Delete attribute \'' +
          escapedKey +
          '">\u00d7</button>';
        html += "</div>";
      });
      html += "</div>";
      html +=
        '<button class="attr-add" id="attrAddBtn" type="button">+ Add Attribute</button>';
      html += "</div>";

      container.innerHTML = html;

      // Wire value-change handlers (use index to capture each key by position)
      var attrList = container.querySelector("#attrList");
      var valueInputs = attrList.querySelectorAll(".attr-value");
      keys.forEach(function (key, idx) {
        var input = valueInputs[idx];
        if (!input) {
          return;
        }
        input.addEventListener("change", function (e) {
          var rawVal = e.target.value;
          var coerced = rawVal;
          if (rawVal === "true") {
            coerced = true;
          } else if (rawVal === "false") {
            coerced = false;
          } else if (rawVal !== "" && !isNaN(Number(rawVal))) {
            coerced = Number(rawVal);
          }
          if (!ann.attributes) {
            ann.attributes = {};
          }
          ann.attributes[key] = coerced;
          scheduleSave("Attributes updated");
        });
      });

      // Wire delete buttons
      attrList.querySelectorAll(".attr-delete").forEach(function (btn) {
        btn.addEventListener("click", function (e) {
          var key = e.currentTarget.getAttribute("data-key");
          if (!key || !ann.attributes) {
            return;
          }
          delete ann.attributes[key];
          if (Object.keys(ann.attributes).length === 0) {
            delete ann.attributes;
          }
          scheduleSave("Attributes updated");
          self.renderAttributesTab(container);
        });
      });

      // Wire add-attribute button
      var addBtn = container.querySelector("#attrAddBtn");
      if (addBtn) {
        addBtn.addEventListener("click", function () {
          addBtn.disabled = true;
          var list = container.querySelector("#attrList");
          var row = document.createElement("div");
          row.className = "attr-row attr-row-new";
          row.innerHTML =
            '<input class="attr-key attr-key-new" placeholder="key" aria-label="New attribute key" />' +
            '<input class="attr-value attr-value-new" placeholder="value" aria-label="New attribute value" />' +
            '<button class="icon-btn attr-confirm" type="button" title="Save attribute" aria-label="Save attribute">\u2713</button>' +
            '<button class="icon-btn attr-cancel" type="button" title="Cancel" aria-label="Cancel">\u00d7</button>';
          list.appendChild(row);

          var keyInput = row.querySelector(".attr-key-new");
          var valInput = row.querySelector(".attr-value-new");
          var confirmBtn = row.querySelector(".attr-confirm");
          var cancelBtn = row.querySelector(".attr-cancel");

          keyInput.focus();

          function confirmAdd() {
            var newKey = keyInput.value.trim();
            if (!newKey) {
              keyInput.focus();
              return;
            }
            var rawVal = valInput.value;
            var coerced = rawVal;
            if (rawVal === "true") {
              coerced = true;
            } else if (rawVal === "false") {
              coerced = false;
            } else if (rawVal !== "" && !isNaN(Number(rawVal))) {
              coerced = Number(rawVal);
            }
            if (!ann.attributes) {
              ann.attributes = {};
            }
            ann.attributes[newKey] = coerced;
            scheduleSave("Attributes updated");
            self.renderAttributesTab(container);
          }

          function cancelAdd() {
            list.removeChild(row);
            addBtn.disabled = false;
          }

          confirmBtn.addEventListener("click", confirmAdd);
          cancelBtn.addEventListener("click", cancelAdd);
          valInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter") {
              e.preventDefault();
              confirmAdd();
            }
            if (e.key === "Escape") {
              cancelAdd();
            }
          });
          keyInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter") {
              e.preventDefault();
              valInput.focus();
            }
            if (e.key === "Escape") {
              cancelAdd();
            }
          });
        });
      }
    },

    /**
     * D9: Render the Raw Data tab into *container*.
     * Shows the current image's COCO JSON with syntax highlighting and a copy
     * button. Shows a placeholder when no image is loaded.
     * Updates automatically when renderAll() / renderLeftPanel() are called.
     * @param {HTMLElement} container
     */
    renderRawDataTab: function (container) {
      var imageRecord = getCurrentImageRecord(false);
      if (!imageRecord) {
        container.innerHTML =
          '<p class="hint">No image loaded. Open an image to view raw data.</p>';
        return;
      }

      var annotations = getCurrentAnnotations();
      var data = {
        image: imageRecord,
        annotations: annotations,
        categories: state.classes || [],
      };
      var jsonStr = JSON.stringify(data, null, 2);
      var highlighted = this._highlightJson(jsonStr);

      var html = '<div class="raw-data-panel">';
      html += '<pre class="json-viewer">' + highlighted + "</pre>";
      html +=
        '<button class="copy-btn" id="copyRawBtn" type="button">Copy JSON</button>';
      html += "</div>";

      container.innerHTML = html;

      var copyBtn = container.querySelector("#copyRawBtn");
      if (!copyBtn) {
        return;
      }

      if (navigator.clipboard) {
        copyBtn.addEventListener("click", function () {
          navigator.clipboard
            .writeText(jsonStr)
            .then(function () {
              copyBtn.textContent = "Copied!";
              window.setTimeout(function () {
                copyBtn.textContent = "Copy JSON";
              }, 1500);
            })
            .catch(function () {
              copyBtn.textContent = "Copy failed";
              window.setTimeout(function () {
                copyBtn.textContent = "Copy JSON";
              }, 1500);
            });
        });
      } else {
        // Fallback for non-secure contexts
        copyBtn.addEventListener("click", function () {
          try {
            var ta = document.createElement("textarea");
            ta.value = jsonStr;
            ta.style.cssText = "position:fixed;opacity:0;top:0;left:0";
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
            copyBtn.textContent = "Copied!";
          } catch (_) {
            copyBtn.textContent = "Copy failed";
          }
          window.setTimeout(function () {
            copyBtn.textContent = "Copy JSON";
          }, 1500);
        });
      }
    },

    /**
     * D9: Single-pass JSON syntax highlighter for well-formed JSON.stringify
     * output. Returns an HTML string with span elements for color coding:
     *   .json-key  — object keys
     *   .json-str  — string values
     *   .json-num  — numbers
     *   .json-kw   — true / false / null
     * HTML-unsafe chars inside string tokens are escaped.
     * @private
     * @param {string} jsonStr
     * @returns {string}
     */
    _highlightJson: function (jsonStr) {
      var result = "";
      var i = 0;
      var len = jsonStr.length;

      function esc(s) {
        return s
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;");
      }

      while (i < len) {
        var ch = jsonStr[i];

        // JSON string token
        if (ch === '"') {
          var j = i + 1;
          while (j < len) {
            if (jsonStr[j] === "\\") {
              j += 2;
              continue;
            }
            if (jsonStr[j] === '"') {
              j++;
              break;
            }
            j++;
          }
          var token = jsonStr.slice(i, j);
          // Check if this string is a key (followed by optional ws then ':')
          var tail = jsonStr.slice(j).replace(/^[ \t\r\n]*/, "");
          var isKey = tail.length > 0 && tail[0] === ":";
          var escapedToken = esc(token);
          if (isKey) {
            result += '<span class="json-key">' + escapedToken + "</span>";
          } else {
            result += '<span class="json-str">' + escapedToken + "</span>";
          }
          i = j;
          continue;
        }

        // Number token
        if (ch === "-" || (ch >= "0" && ch <= "9")) {
          var numStart = i;
          i++;
          while (
            i < len &&
            ((jsonStr[i] >= "0" && jsonStr[i] <= "9") || jsonStr[i] === ".")
          ) {
            i++;
          }
          if (i < len && (jsonStr[i] === "e" || jsonStr[i] === "E")) {
            i++;
            if (i < len && (jsonStr[i] === "+" || jsonStr[i] === "-")) {
              i++;
            }
            while (i < len && jsonStr[i] >= "0" && jsonStr[i] <= "9") {
              i++;
            }
          }
          result +=
            '<span class="json-num">' + jsonStr.slice(numStart, i) + "</span>";
          continue;
        }

        // Keyword tokens
        if (jsonStr.slice(i, i + 4) === "true") {
          result += '<span class="json-kw">true</span>';
          i += 4;
          continue;
        }
        if (jsonStr.slice(i, i + 5) === "false") {
          result += '<span class="json-kw">false</span>';
          i += 5;
          continue;
        }
        if (jsonStr.slice(i, i + 4) === "null") {
          result += '<span class="json-kw">null</span>';
          i += 4;
          continue;
        }

        // Structural and whitespace characters
        if (ch === "&") {
          result += "&amp;";
        } else if (ch === "<") {
          result += "&lt;";
        } else if (ch === ">") {
          result += "&gt;";
        } else {
          result += ch;
        }
        i++;
      }

      return result;
    },
  };

  // ── D4: EditorView tool palette ───────────────────────────────────────────────

  /**
   * Wire up the tool palette via a single delegated click handler on the
   * #editorToolPalette container.  Safe to call multiple times — guarded by
   * _paletteInitialized flag so only one listener is ever attached.
   */
  EditorView.initToolPalette = function () {
    if (this._paletteInitialized) {
      return;
    }
    this._paletteInitialized = true;

    var palette = document.getElementById("editorToolPalette");
    if (!palette) {
      return;
    }

    palette.addEventListener("click", function (e) {
      if (Router.current !== "edit") {
        return;
      }
      var btn = e.target.closest("[data-tool]");
      if (!btn || btn.disabled) {
        return;
      }
      selectTool(btn.getAttribute("data-tool"));
    });
  };

  // ── D2: EditorView canvas methods ─────────────────────────────────────────────

  /**
   * D8: Initialise zoom and pan event handlers for the editor canvas.
   * Sets up:
   *   - Mouse-wheel zoom toward cursor (10% steps, range 10%–500%)
   *   - Middle-click drag-to-pan
   * Space+drag panning is handled via the global keydown/keyup handlers.
   * Guarded by _zoomPanInitialized — safe to call on every mount.
   */
  EditorView.initZoomPan = function () {
    if (this._zoomPanInitialized) {
      return;
    }
    this._zoomPanInitialized = true;

    var canvas = elements.imageCanvas;
    if (!canvas) {
      return;
    }

    // Wheel zoom anchored to cursor (10% steps, max 500%)
    canvas.addEventListener(
      "wheel",
      function (e) {
        if (!elements.imageLoader.naturalWidth) {
          return; // no image loaded yet
        }
        e.preventDefault();

        var factor = e.deltaY < 0 ? 1.1 : 0.9; // 10% steps per wheel tick
        var oldZoom = editorState.zoom;
        var newZoom = Math.max(0.1, Math.min(5.0, oldZoom * factor));

        // Anchor zoom to the cursor position in the scrollable container.
        var containerEl = elements.canvasStack;
        var rect = containerEl.getBoundingClientRect();
        var vx = e.clientX - rect.left;
        var vy = e.clientY - rect.top;

        var scaleX = state.view.scaleX;
        var scaleY = state.view.scaleY;
        var imgX, imgY;
        if (scaleX > 0 && scaleY > 0 && oldZoom > 0) {
          imgX = (containerEl.scrollLeft + vx) / (scaleX * oldZoom);
          imgY = (containerEl.scrollTop + vy) / (scaleY * oldZoom);
        }
        editorState.zoom = newZoom;
        renderCanvas();
        if (imgX !== undefined) {
          containerEl.scrollLeft = Math.round(imgX * scaleX * newZoom - vx);
          containerEl.scrollTop = Math.round(imgY * scaleY * newZoom - vy);
        }
        EditorView.renderBottomBar();
      },
      { passive: false },
    );
  };

  /**
   * Initialise the canvas interaction layer for the editor.
   * Delegates to initZoomPan() for zoom/pan setup.
   * Should be called once on first mount; subsequent calls are no-ops.
   */
  EditorView.initCanvas = function () {
    if (this._canvasInitialized) {
      return;
    }
    this._canvasInitialized = true;
    this.initZoomPan();
  };

  /**
   * Fit the canvas element to the current image dimensions (zoom=1, pan=0).
   * Delegates to the existing top-level fitCanvasToImage() function so that
   * state.view is always updated in one place.
   */
  EditorView.fitCanvasToImage = function () {
    fitCanvasToImage();
  };

  /**
   * Re-render canvas annotations + image.
   * Delegates to the top-level renderCanvas() function.
   */
  EditorView.renderCanvas = function () {
    renderCanvas();
  };

  /**
   * Load image at the given index into the editor canvas.
   * Delegates to selectImage() and returns the resulting Promise.
   * @param {number} index - 0-based index in state.images.
   * @param {object} [options] - forwarded to selectImage().
   */
  EditorView.loadImage = function (index, options) {
    return selectImage(index, options);
  };

  /**
   * Persist annotations for the current dataset immediately.
   * Delegates to saveAnnotationsNow().
   * @param {boolean} [force] - if true, save even when state is clean.
   */
  EditorView.saveNow = function (force) {
    return saveAnnotationsNow(force);
  };

  /**
   * Schedule an auto-save after SAVE_DEBOUNCE_MS milliseconds.
   * Delegates to scheduleSave().
   * @param {string} [message] - status label shown during dirty state.
   */
  EditorView.scheduleSave = function (message) {
    scheduleSave(message);
  };

  /**
   * Push the current COCO state onto the undo stack.
   * Delegates to pushUndoState().
   */
  EditorView.pushUndo = function () {
    pushUndoState();
  };

  /**
   * D5: Render the Annotation Properties panel into #annotPropsPanel.
   * Shows bbox (xyxy), area, category dropdown, and score badge for the
   * currently selected annotation. Shows a placeholder when nothing is selected.
   * Wires the category <select> to update the annotation and trigger auto-save.
   */
  EditorView.renderAnnotationProperties = function () {
    var panel = document.getElementById("annotPropsPanel");
    if (!panel) {
      return;
    }
    var ann = getSelectedAnnotation();
    var categories = state.coco ? state.coco.categories : [];

    if (!ann) {
      panel.innerHTML =
        '<p class="props-placeholder">Select an annotation to view its properties.</p>';
      return;
    }

    var x = ann.bbox[0];
    var y = ann.bbox[1];
    var w = ann.bbox[2];
    var h = ann.bbox[3];
    var x1 = Math.round(x);
    var y1 = Math.round(y);
    var x2 = Math.round(x + w);
    var y2 = Math.round(y + h);
    var area = Math.round(w * h);
    var annId = ann.id;

    var catOptions = categories
      .map(function (c) {
        return (
          '<option value="' +
          c.id +
          '"' +
          (c.id === ann.category_id ? " selected" : "") +
          ">" +
          escapeHtml(c.name) +
          "</option>"
        );
      })
      .join("");

    var html = '<div class="props-panel">';
    html += '<div class="props-header">Properties</div>';
    // Bbox row
    html += '<div class="prop-row">';
    html += '<span class="prop-label">Bbox</span>';
    html +=
      '<span class="prop-value" id="propsBboxCoords">(' +
      x1 +
      ", " +
      y1 +
      ") \u2192 (" +
      x2 +
      ", " +
      y2 +
      ")</span>";
    html += "</div>";
    // Area row
    html += '<div class="prop-row">';
    html += '<span class="prop-label">Area</span>';
    html +=
      '<span class="prop-value" id="propsArea">' + area + " px\u00B2</span>";
    html += "</div>";
    // Category dropdown row
    html += '<div class="prop-row">';
    html += '<span class="prop-label">Category</span>';
    html +=
      '<select class="prop-cat-select" id="propsCategorySelect" data-ann-id="' +
      annId +
      '">' +
      catOptions +
      "</select>";
    html += "</div>";
    // Score row (only if annotation has a score)
    if (ann.score !== undefined) {
      html += '<div class="prop-row">';
      html += '<span class="prop-label">Score</span>';
      html +=
        '<span class="score-badge">' +
        (ann.score * 100).toFixed(1) +
        "%</span>";
      html += "</div>";
    }
    html += "</div>";
    panel.innerHTML = html;

    // Wire category dropdown
    var select = document.getElementById("propsCategorySelect");
    if (select) {
      select.addEventListener("change", function () {
        var newCatId = parseInt(this.value, 10);
        if (isNaN(newCatId)) {
          return;
        }
        var targetId = parseInt(select.getAttribute("data-ann-id"), 10);
        var annotations = getCurrentAnnotations();
        var annotation = annotations.find(function (a) {
          return a.id === targetId;
        });
        if (!annotation) {
          return;
        }
        pushUndoState();
        annotation.category_id = newCatId;
        state.lastUsedCategoryId = newCatId;
        scheduleSave("Category changed");
        renderAll();
      });
    }
  };

  /**
   * D5: Lightweight sync of the properties panel during annotation drag/resize.
   * Updates only the bbox-coords and area text spans without rebuilding the DOM.
   * Falls back to a full renderAnnotationProperties() when the displayed
   * annotation ID no longer matches the current selection.
   * Called from renderCanvas() on every frame.
   */
  EditorView.syncPropsPanel = function () {
    var panel = document.getElementById("annotPropsPanel");
    if (!panel) {
      return;
    }
    var ann = getSelectedAnnotation();
    var bboxEl = document.getElementById("propsBboxCoords");
    var areaEl = document.getElementById("propsArea");
    var select = document.getElementById("propsCategorySelect");

    if (!ann) {
      // If panels shows a props-panel (stale selection) → show placeholder
      if (panel.querySelector(".props-panel")) {
        this.renderAnnotationProperties();
      }
      return;
    }

    // Full rebuild if annotation ID changed or elements are missing
    if (
      !bboxEl ||
      !areaEl ||
      !select ||
      parseInt(select.getAttribute("data-ann-id"), 10) !== ann.id
    ) {
      this.renderAnnotationProperties();
      return;
    }

    // Fast path: update just the dynamic text values
    var x = ann.bbox[0];
    var y = ann.bbox[1];
    var w = ann.bbox[2];
    var h = ann.bbox[3];
    var x1 = Math.round(x);
    var y1 = Math.round(y);
    var x2 = Math.round(x + w);
    var y2 = Math.round(y + h);
    bboxEl.textContent =
      "(" + x1 + ", " + y1 + ") \u2192 (" + x2 + ", " + y2 + ")";
    areaEl.textContent = Math.round(w * h) + " px\u00B2";
  };

  // ── D7: Auto-Annotate Panel ────────────────────────────────────────────────

  /**
   * D7: Build the HTML string for the Auto Annotate section rendered at the
   * bottom of the Labels tab. The section includes a mode dropdown, an
   * adaptive prompt input, the ANNOTATE button, and a status indicator.
   * @param {Array} categories - current dataset categories (for CLIP pre-fill)
   * @returns {string} HTML string
   */
  EditorView._buildAutoAnnotateHTML = function (categories) {
    // Restore mode from localStorage if available
    var _autoSaved = {};
    try { _autoSaved = JSON.parse(localStorage.getItem("mata-annotate-auto-settings") || "{}"); } catch(e) {}
    if (_autoSaved.mode && !editorState._autoModeSaved) {
      editorState.autoAnnotateMode = _autoSaved.mode;
      editorState._autoModeSaved = true;
    }

    var mode = editorState.autoAnnotateMode || "detect";
    var modeConfig = null;
    for (var i = 0; i < AUTO_ANNOTATE_MODES.length; i++) {
      if (AUTO_ANNOTATE_MODES[i].id === mode) {
        modeConfig = AUTO_ANNOTATE_MODES[i];
        break;
      }
    }
    if (!modeConfig) {
      modeConfig = AUTO_ANNOTATE_MODES[0];
    }

    var html = '<div class="auto-annotate-section" id="autoAnnotateSection">';
    html += '<div class="auto-annotate-header">';
    html += '<span class="auto-annotate-title">AUTO ANNOTATE</span>';
    html += "</div>";

    // Mode dropdown
    html +=
      '<select class="auto-annotate-mode-select" id="autoAnnotateModeSelect">';
    AUTO_ANNOTATE_MODES.forEach(function (m) {
      html +=
        '<option value="' +
        m.id +
        '"' +
        (m.id === mode ? " selected" : "") +
        ">" +
        escapeHtml(m.label) +
        "</option>";
    });
    html += "</select>";

    // Adaptive prompt input
    html +=
      '<label class="auto-annotate-prompt-label" id="autoAnnotatePromptLabel">';
    html += escapeHtml(modeConfig.promptLabel);
    html += "</label>";

    var promptValue = (_autoSaved.prompts && _autoSaved.prompts[mode] !== undefined)
      ? _autoSaved.prompts[mode]
      : modeConfig.promptDefault;
    // CLIP mode: pre-fill with current category names if no saved value
    if (mode === "clip" && (_autoSaved.prompts === undefined || _autoSaved.prompts[mode] === undefined) && categories && categories.length > 0) {
      promptValue = categories
        .map(function (c) {
          return c.name;
        })
        .join(", ");
    }

    if (modeConfig.promptType === "number") {
      html +=
        '<input type="number" class="auto-annotate-threshold" id="autoAnnotatePrompt"' +
        ' step="0.01" min="0" max="1" value="' +
        escapeHtml(String(promptValue)) +
        '">';
    } else {
      html +=
        '<textarea class="auto-annotate-prompt-textarea" id="autoAnnotatePrompt" rows="3">' +
        escapeHtml(String(promptValue)) +
        "</textarea>";
    }

    // ANNOTATE button + status
    html +=
      '<button class="auto-annotate-btn" id="autoAnnotateBtn">ANNOTATE</button>';
    html += '<div class="auto-annotate-status" id="autoAnnotateStatus"></div>';
    html += "</div>"; // .auto-annotate-section

    return html;
  };

  /**
   * D7: Wire event listeners for the auto-annotate section.
   * Called after `renderLabelsTab()` injects the HTML into the DOM.
   * @param {HTMLElement} container - the left panel content container
   * @param {Array} categories - current dataset categories
   */
  EditorView._initAutoAnnotateListeners = function (container, categories) {
    var self = this;

    function saveAutoSettings() {
      var m = editorState.autoAnnotateMode || "detect";
      var promptEl = container.querySelector("#autoAnnotatePrompt");
      var p = promptEl ? promptEl.value : "";
      try {
        var settings = JSON.parse(localStorage.getItem("mata-annotate-auto-settings") || "{}");
        settings.mode = m;
        settings.prompts = settings.prompts || {};
        settings.prompts[m] = p;
        localStorage.setItem("mata-annotate-auto-settings", JSON.stringify(settings));
      } catch(e) {}
    }

    var modeSelect = container.querySelector("#autoAnnotateModeSelect");
    if (modeSelect) {
      modeSelect.addEventListener("change", function () {
        editorState.autoAnnotateMode = modeSelect.value;
        saveAutoSettings();
        // Re-render the labels tab to swap the prompt widget
        var leftPanelContainer = document.getElementById("leftPanelContent");
        if (leftPanelContainer) {
          self.renderLabelsTab(leftPanelContainer);
        }
      });
    }

    var promptEl = container.querySelector("#autoAnnotatePrompt");
    if (promptEl) {
      promptEl.addEventListener("input", function () {
        saveAutoSettings();
      });
    }

    var btn = container.querySelector("#autoAnnotateBtn");
    if (btn) {
      btn.addEventListener("click", function () {
        EditorView.runAutoAnnotate();
      });
    }
  };

  /**
   * D7: Execute the auto-annotate request for the current mode.
   * Builds the request body, calls the appropriate /api/assist/ endpoint,
   * and converts the response candidates to draft annotations on the canvas.
   */
  EditorView.runAutoAnnotate = function () {
    var mode = editorState.autoAnnotateMode || "detect";
    var modeConfig = null;
    var i;
    for (i = 0; i < AUTO_ANNOTATE_MODES.length; i++) {
      if (AUTO_ANNOTATE_MODES[i].id === mode) {
        modeConfig = AUTO_ANNOTATE_MODES[i];
        break;
      }
    }
    if (!modeConfig) {
      return;
    }

    var promptEl = document.getElementById("autoAnnotatePrompt");
    var promptValue = promptEl ? promptEl.value.trim() : "";

    // Build request body per mode
    var imageMeta = getCurrentImageMeta();
    var body = {};

    if (mode === "detect") {
      if (!state.selectedDataset || !imageMeta) {
        EditorView._setAutoAnnotateStatus("error", "No image loaded.");
        return;
      }
      body.dataset = state.selectedDataset;
      body.image_filename = imageMeta.filename;
      var threshold = parseFloat(promptValue);
      if (!isNaN(threshold)) {
        body.threshold = threshold;
      }
    } else if (mode === "vlm") {
      if (!imageMeta) {
        EditorView._setAutoAnnotateStatus("error", "No image loaded.");
        return;
      }
      body.image_path =
        "/api/datasets/" +
        encodeURIComponent(state.selectedDataset || "") +
        "/images/" +
        encodeURIComponent(imageMeta.filename);
      body.prompt = promptValue || modeConfig.promptDefault;
      var vlmClasses = (state.classes || []).map(function (c) {
        return c.name;
      });
      if (vlmClasses.length) {
        body.class_names = vlmClasses;
      }
    } else if (mode === "clip") {
      if (!imageMeta) {
        EditorView._setAutoAnnotateStatus("error", "No image loaded.");
        return;
      }
      var classNames = promptValue
        ? promptValue
            .split(",")
            .map(function (s) {
              return s.trim();
            })
            .filter(Boolean)
        : [];
      if (!classNames.length) {
        EditorView._setAutoAnnotateStatus(
          "error",
          "Enter class names for CLIP mode.",
        );
        return;
      }
      body.image_path =
        "/api/datasets/" +
        encodeURIComponent(state.selectedDataset || "") +
        "/images/" +
        encodeURIComponent(imageMeta.filename);
      body.class_names = classNames;
      body.top_k = 5;
    }

    EditorView._setAutoAnnotateStatus("running", "");

    fetch(modeConfig.endpoint, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    })
      .then(function (resp) {
        if (resp.status === 501) {
          return resp.json().then(function (data) {
            var msg =
              (data && data.error) ||
              "Model not loaded. Start the server with the required --model flag.";
            EditorView._setAutoAnnotateStatus("error", msg);
            return null;
          });
        }
        if (!resp.ok) {
          return resp.json().then(function (data) {
            var msg =
              (data && data.error) || "Request failed (" + resp.status + ").";
            EditorView._setAutoAnnotateStatus("error", msg);
            return null;
          });
        }
        return resp.json();
      })
      .then(function (data) {
        if (!data) {
          return;
        }
        var candidates = data.candidates || data.suggestions || [];
        EditorView.addCandidatesToCanvas(candidates, mode);
        EditorView._setAutoAnnotateStatus(
          "complete",
          candidates.length + " candidate(s) added.",
        );
      })
      .catch(function (err) {
        EditorView._setAutoAnnotateStatus(
          "error",
          err.message || "Network error.",
        );
      });
  };

  /**
   * D7: Set the auto-annotate status indicator.
   * @param {string} status - "running" | "complete" | "error" | ""
   * @param {string} message - optional description to show
   */
  EditorView._setAutoAnnotateStatus = function (status, message) {
    var el = document.getElementById("autoAnnotateStatus");
    if (!el) {
      return;
    }
    el.dataset.status = status || "";
    if (status === "running") {
      el.textContent = "\u23F3 Running\u2026";
    } else if (status === "complete") {
      el.textContent = "\u2713 " + (message || "Done.");
    } else if (status === "error") {
      el.textContent = "\u26A0 " + (message || "Error.");
    } else {
      el.textContent = "";
    }
  };

  /**
   * D7: Add AI-generated annotation candidates as dashed-border draft
   * annotations on the canvas. Drafts are stored in
   * `state.draftCandidates` and rendered by renderCanvas().
   * @param {Array} candidates - array of {bbox_xywh, label, score} objects
   *   (or {label, score} for CLIP classify suggestions)
   * @param {string} mode - "detect" | "vlm" | "clip"
   */
  EditorView.addCandidatesToCanvas = function (candidates, mode) {
    if (!Array.isArray(candidates) || !candidates.length) {
      return;
    }
    // For detect/vlm modes: each candidate has bbox_xywh or bbox_xyxy
    // For clip mode: suggestions are classification scores (no bbox), skip
    var imageRecord = getCurrentImageRecord(false);
    if (!imageRecord) {
      return;
    }
    pushUndoState();
    candidates.forEach(function (candidate) {
      var bbox;
      if (
        Array.isArray(candidate.bbox_xywh) &&
        candidate.bbox_xywh.length === 4
      ) {
        bbox = candidate.bbox_xywh;
      } else if (
        Array.isArray(candidate.bbox_xyxy) &&
        candidate.bbox_xyxy.length === 4
      ) {
        bbox = xyxyToXywh(candidate.bbox_xyxy);
      } else {
        // No bbox (e.g. CLIP classify) — nothing to place on canvas
        return;
      }
      // Find or create category for the candidate label
      var labelName = candidate.label || candidate.category || "object";
      var category =
        state.coco &&
        state.coco.categories.find(function (c) {
          return c.name.toLowerCase() === String(labelName).toLowerCase();
        });
      if (!category && state.coco) {
        category = {
          id: nextId(state.coco.categories),
          name: String(labelName),
          supercategory: String(labelName),
        };
        state.coco.categories.push(category);
      }
      var categoryId = category ? category.id : 1;
      var ann = {
        id: nextId(state.coco ? state.coco.annotations : []),
        image_id: imageRecord.id,
        category_id: categoryId,
        bbox: [
          Math.round(bbox[0]),
          Math.round(bbox[1]),
          Math.round(bbox[2]),
          Math.round(bbox[3]),
        ],
        area: Math.round(bbox[2] * bbox[3]),
        iscrowd: 0,
        segmentation: [],
        // D7: flag as AI-draft for dashed rendering
        _draft: true,
        score:
          typeof candidate.score === "number" ? candidate.score : undefined,
      };
      if (state.coco) {
        state.coco.annotations.push(ann);
      }
    });
    scheduleSave("AI candidates added");
    renderAll();
  };

  var Router = {
    current: null, // "browse" | "edit"
    params: {}, // { dataset, imageIndex }
    _ignoreNext: false, // skip one hashchange after restoring hash on cancel

    /** Bootstrap: call once on page load to start responding to hash changes. */
    init: function () {
      var self = this;
      window.addEventListener("hashchange", function () {
        if (self._ignoreNext) {
          self._ignoreNext = false;
          return;
        }
        self.route();
      });
      this.route();
    },

    /** Parse the current hash and activate the matching view. */
    route: function () {
      var hash = location.hash || "#/browse";
      var parts;
      if (hash.startsWith("#/edit/")) {
        parts = hash.slice(7).split("/");
        this.params = {
          dataset: decodeURIComponent(parts[0] || ""),
          imageIndex: parseInt(parts[1] || "0", 10) || 0,
        };
        this.show("edit");
      } else {
        this.params = {};
        this.show("browse");
      }
    },

    /** Switch to the given view, handling dirty-state guard and lifecycle. */
    show: function (view) {
      if (this.current === view) {
        return;
      }

      // Guard: prompt before leaving an editor with unsaved changes
      if (this.current === "edit" && state.dirty) {
        var proceed = window.confirm(
          "You have unsaved annotations.\n\nLeave the editor? Changes that could not be saved will be lost.",
        );
        if (!proceed) {
          // Restore hash to the editor position without re-triggering route
          this._ignoreNext = true;
          location.hash =
            "#/edit/" +
            encodeURIComponent(state.selectedDataset || "") +
            "/" +
            Math.max(0, state.selectedImageIndex);
          return;
        }
      }

      // Unmount the current view
      if (this.current === "edit") {
        EditorView.unmount();
      }
      if (this.current === "browse") {
        BrowserView.unmount();
      }

      // Toggle view visibility via the HTML `hidden` attribute
      var browserEl = document.getElementById("browser-view");
      var editorEl = document.getElementById("editor-view");
      if (browserEl) {
        browserEl.hidden = view !== "browse";
      }
      if (editorEl) {
        editorEl.hidden = view !== "edit";
      }

      // E3: track editor mode on shell for responsive CSS targeting
      var shell = document.getElementById("shell");
      if (shell) {
        shell.classList.toggle("editor-mode", view === "edit");
      }

      this.current = view;

      // Mount the new view
      if (view === "browse") {
        BrowserView.mount();
      }
      if (view === "edit") {
        EditorView.mount(this.params);
      }
    },

    /** Programmatically navigate to a view by updating the location hash. */
    navigate: function (view, params) {
      params = params || {};
      if (view === "edit") {
        location.hash =
          "#/edit/" +
          encodeURIComponent(params.dataset || "") +
          "/" +
          (params.imageIndex || 0);
      } else {
        location.hash = "#/browse";
      }
    },
  };

  // ── Element references (mapped to B1 HTML IDs) ────────────────────────────────
  var elements = {
    shell: document.getElementById("shell"),
    sidebar: document.getElementById("browserSidebar"), // B1: was "sidebar"
    sidebarToggle: document.getElementById("sidebarToggle"),
    backdrop: document.getElementById("mobileNavBackdrop"), // B1: was "backdrop"
    datasetList: document.getElementById("datasetList"),
    datasetEmpty: document.getElementById("datasetEmpty"),
    datasetError: document.getElementById("datasetError"),
    datasetStatus: document.getElementById("datasetStatus"),
    imageList: null, // B1: replaced by thumbnail grid (C2)
    imageEmpty: null, // B1: replaced by thumbnail grid (C2)
    imageError: null, // B1: replaced by thumbnail grid (C2)
    imageStatus: null, // B1: replaced by thumbnail grid (C2)
    // C3: browser view search/filter/pagination elements
    imageSearch: document.getElementById("imageSearch"),
    sortSelect: document.getElementById("sortSelect"),
    thumbnailGrid: document.getElementById("thumbnailGrid"),
    browserPlaceholder: document.getElementById("browserPlaceholder"),
    loadingOverlay: document.getElementById("browserLoadingOverlay"),
    loadingLabel: document.getElementById("browserLoadingLabel"),
    loadingSubLabel: document.getElementById("browserLoadingSubLabel"),
    paginationInfo: document.getElementById("paginationInfo"),
    pagePrevBtn: document.getElementById("pagePrevBtn"),
    pageNextBtn: document.getElementById("pageNextBtn"),
    perPageSelect: document.getElementById("perPageSelect"),
    datasetCount: document.getElementById("datasetCount"),
    imageCount: document.getElementById("imageCount"),
    selectedIndex: document.getElementById("editorNavCounter"), // B1: was "selectedIndex"
    viewerTitle: document.getElementById("editorFilename"), // B1: was "viewerTitle"
    viewerSubtitle: null, // B1: in browser header (C2)
    progressText: null, // B1: in browser header (C2)
    progressFill: null, // B1: browse progress fill (C2)
    annotationCount: null, // B1: in left panel (D3)
    activeToolLabel: null, // B1: removed; shown via tool button active state
    imageCanvas: document.getElementById("imageCanvas"),
    imageLoader: document.getElementById("imageLoader"),
    canvasStack: document.getElementById("editorCanvasArea"), // B1: was "canvasStack"
    canvasPlaceholder: document.getElementById("canvasPlaceholder"),
    placeholderText: document.getElementById("placeholderText"),
    classList: null, // B1: in left panel (D3)
    classHint: document.getElementById("leftPanelHint"), // B1: was "classHint"
    classPicker: document.getElementById("classPicker"),
    classPickerHint: document.getElementById("classPickerHint"),
    classPickerList: document.getElementById("classPickerList"),
    classPickerError: document.getElementById("classPickerError"),
    classPickerClose: document.getElementById("classPickerClose"),
    bboxTool: document.querySelector("[data-tool='bbox']"), // B1: was id="bboxTool"
    polygonTool: document.querySelector("[data-tool='polygon']"), // B1: was id="polygonTool"
    selectTool: document.querySelector("[data-tool='select']"), // B1: new
    saveStatus: document.getElementById("saveStatus"),
    selectedLabel: null, // B1: in left panel (D3)
    toolbarHint: document.getElementById("leftPanelHint"), // B1: same el as classHint
    annotatedCount: document.getElementById("annotatedCount"), // C1: workspace status grid
  };

  var canvasContext = elements.imageCanvas.getContext("2d");

  function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function createEmptyCoco() {
    return {
      info: { description: "MATA annotate dataset", version: "1.0" },
      licenses: [],
      images: [],
      annotations: [],
      categories: [],
    };
  }

  function normaliseCoco(payload) {
    if (!payload || typeof payload !== "object") {
      return createEmptyCoco();
    }

    return {
      info:
        payload.info && typeof payload.info === "object"
          ? payload.info
          : createEmptyCoco().info,
      licenses: Array.isArray(payload.licenses) ? payload.licenses : [],
      images: Array.isArray(payload.images) ? payload.images : [],
      annotations: Array.isArray(payload.annotations)
        ? payload.annotations
        : [],
      categories: Array.isArray(payload.categories) ? payload.categories : [],
    };
  }

  function nextId(items) {
    var maxId = 0;
    items.forEach(function (item) {
      if (item && typeof item.id === "number" && item.id > maxId) {
        maxId = item.id;
      }
    });
    return maxId + 1;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function buildTag(text) {
    return '<span class="tag">' + escapeHtml(text) + "</span>";
  }

  function formatBytes(bytes) {
    if (!bytes) {
      return "0 B";
    }

    var units = ["B", "KB", "MB", "GB"];
    var value = bytes;
    var unitIndex = 0;

    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex += 1;
    }

    return value.toFixed(unitIndex === 0 ? 0 : 1) + " " + units[unitIndex];
  }

  // ── Type badge helpers (C1) ─────────────────────────────────────────────────

  function typeBadgeClass(type) {
    var t = (type || "").toLowerCase();
    if (t === "coco") {
      return "type-badge-coco";
    }
    if (t === "imagefolder") {
      return "type-badge-folder";
    }
    if (t === "voc") {
      return "type-badge-voc";
    }
    return "type-badge-empty";
  }

  function typeBadgeLabel(type) {
    var t = (type || "").toLowerCase();
    if (t === "coco") {
      return "COCO";
    }
    if (t === "imagefolder") {
      return "Folder";
    }
    if (t === "voc") {
      return "VOC";
    }
    return type ? escapeHtml(type) : "Empty";
  }

  function normaliseName(value) {
    return String(value || "")
      .replace(/\\/g, "/")
      .toLowerCase();
  }

  function basename(value) {
    var parts = normaliseName(value).split("/");
    return parts[parts.length - 1];
  }

  function namesMatch(left, right) {
    var normalLeft = normaliseName(left);
    var normalRight = normaliseName(right);
    return (
      normalLeft === normalRight ||
      basename(normalLeft) === basename(normalRight)
    );
  }

  function getCategoryColor(categoryId) {
    var numericId = Number(categoryId);
    if (!isFinite(numericId) || numericId <= 0) {
      return CATEGORY_COLORS[0];
    }
    // Respect custom color stored on the category object
    if (state.classes) {
      var cat = state.classes.find(function (c) { return c.id === numericId; });
      if (cat && cat.color) {
        return cat.color;
      }
    }
    return CATEGORY_COLORS[
      (Math.floor(numericId) - 1) % CATEGORY_COLORS.length
    ];
  }

  function xyxyToXywh(box) {
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]];
  }

  function polyBboxFromFlat(flat) {
    var xs = [];
    var ys = [];
    var i;
    for (i = 0; i < flat.length - 1; i += 2) {
      xs.push(flat[i]);
      ys.push(flat[i + 1]);
    }
    if (!xs.length) {
      return [0, 0, 0, 0];
    }
    var minX = Math.min.apply(null, xs);
    var minY = Math.min.apply(null, ys);
    var maxX = Math.max.apply(null, xs);
    var maxY = Math.max.apply(null, ys);
    return [minX, minY, maxX - minX, maxY - minY];
  }

  function segmentationToCanvasPoints(flat) {
    var points = [];
    var zoom = editorState.zoom;
    var i;
    for (i = 0; i + 1 < flat.length; i += 2) {
      points.push({
        x: flat[i] * state.view.scaleX * zoom,
        y: flat[i + 1] * state.view.scaleY * zoom,
      });
    }
    return points;
  }

  function hasPolygon(annotation) {
    return (
      Array.isArray(annotation.segmentation) &&
      annotation.segmentation.length > 0 &&
      Array.isArray(annotation.segmentation[0]) &&
      annotation.segmentation[0].length >= 6
    );
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function distance(a, b) {
    var dx = a.x - b.x;
    var dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function fetchJson(path) {
    return fetch(path, { headers: { Accept: "application/json" } }).then(
      function (response) {
        return response
          .json()
          .catch(function () {
            return null;
          })
          .then(function (payload) {
            if (!response.ok) {
              throw new Error(
                payload && payload.error ? payload.error : "Request failed.",
              );
            }
            return payload;
          });
      },
    );
  }

  function postJson(path, body) {
    return fetch(path, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    }).then(function (response) {
      return response
        .json()
        .catch(function () {
          return null;
        })
        .then(function (payload) {
          if (!response.ok) {
            throw new Error(
              payload && payload.error ? payload.error : "Request failed.",
            );
          }
          return payload;
        });
    });
  }

  function toggleSidebar(forceOpen) {
    var shouldOpen =
      typeof forceOpen === "boolean"
        ? forceOpen
        : !elements.shell.classList.contains("sidebar-open");
    elements.shell.classList.toggle("sidebar-open", shouldOpen);
    elements.sidebarToggle.setAttribute(
      "aria-expanded",
      shouldOpen ? "true" : "false",
    );
  }

  function applyLoadingState(isLoading, label, subLabel) {
    if (elements.sidebar) {
      elements.sidebar.classList.toggle("loading", isLoading);
    }
    if (elements.imageList) {
      elements.imageList.classList.toggle("loading", isLoading);
    }
    if (elements.loadingOverlay) {
      elements.loadingOverlay.hidden = !isLoading;
    }
    if (elements.loadingLabel && label) {
      elements.loadingLabel.textContent = label;
    }
    if (elements.loadingSubLabel) {
      elements.loadingSubLabel.textContent = subLabel || "";
    }
  }

  function updateCounts() {
    var current =
      state.selectedImageIndex >= 0 ? state.selectedImageIndex + 1 : 0;
    var total = state.images.length;
    var percent =
      total > 0 && current > 0 ? Math.round((current / total) * 100) : 0;

    elements.datasetCount.textContent = String(state.datasets.length);
    elements.imageCount.textContent = String(total);
    if (elements.annotatedCount) {
      if (
        state.datasetStats &&
        typeof state.datasetStats.total_annotated === "number"
      ) {
        elements.annotatedCount.textContent =
          state.datasetStats.total_annotated +
          " / " +
          (state.datasetStats.image_count || total || 0);
      } else {
        elements.annotatedCount.textContent = "0";
      }
    }
    if (elements.selectedIndex) {
      elements.selectedIndex.textContent = current + " of " + total;
    }
    if (elements.progressText) {
      elements.progressText.textContent = percent + "%";
    }
    if (elements.progressFill) {
      elements.progressFill.style.width = percent + "%";
    }
  }

  function updateSaveStatus(text, stateName) {
    elements.saveStatus.textContent = text;
    elements.saveStatus.dataset.state = stateName;
  }

  function setPlaceholder(title, text) {
    elements.canvasPlaceholder.hidden = false;
    elements.imageCanvas.style.display = "none";
    elements.viewerTitle.textContent = title;
    elements.placeholderText.textContent = text;
  }

  function canAnnotateCurrentDataset() {
    return state.selectedDatasetType !== "imagefolder";
  }

  function getCurrentImageMeta() {
    if (
      state.selectedImageIndex < 0 ||
      state.selectedImageIndex >= state.images.length
    ) {
      return null;
    }
    return state.images[state.selectedImageIndex];
  }

  function getCurrentImageRecord(createIfMissing) {
    var imageMeta = getCurrentImageMeta();
    var record;
    if (!imageMeta || !state.coco) {
      return null;
    }

    record =
      state.coco.images.find(function (item) {
        return namesMatch(item.file_name, imageMeta.filename);
      }) || null;

    if (!record && createIfMissing) {
      record = {
        id: nextId(state.coco.images),
        file_name: imageMeta.filename,
        width: imageMeta.width || state.view.imageWidth || 0,
        height: imageMeta.height || state.view.imageHeight || 0,
      };
      state.coco.images.push(record);
    }

    if (record && state.view.imageWidth && state.view.imageHeight) {
      if (!record.width) {
        record.width = state.view.imageWidth;
      }
      if (!record.height) {
        record.height = state.view.imageHeight;
      }
    }

    return record;
  }

  function getCurrentAnnotations() {
    var imageRecord = getCurrentImageRecord(false);
    if (!imageRecord || !state.coco) {
      return [];
    }
    return state.coco.annotations.filter(function (annotation) {
      return annotation.image_id === imageRecord.id;
    });
  }

  function getSelectedAnnotation() {
    return (
      getCurrentAnnotations().find(function (annotation) {
        return annotation.id === state.selectedAnnotationId;
      }) || null
    );
  }

  function getCategory(categoryId) {
    if (!state.coco) {
      return null;
    }
    return (
      state.coco.categories.find(function (category) {
        return category.id === categoryId;
      }) || null
    );
  }

  function ensureCategoriesFromStats() {
    var statsClasses;
    if (!state.coco || state.coco.categories.length > 0) {
      return;
    }
    statsClasses =
      state.datasetStats && Array.isArray(state.datasetStats.classes)
        ? state.datasetStats.classes
        : [];
    statsClasses.forEach(function (className, index) {
      state.coco.categories.push({
        id: index + 1,
        name: className,
        supercategory: className,
      });
    });
  }

  function refreshClasses() {
    if (!state.coco) {
      state.classes = [];
      return;
    }
    ensureCategoriesFromStats();
    state.classes = state.coco.categories.slice().sort(function (left, right) {
      return left.id - right.id;
    });
    if (state.lastUsedCategoryId === null && state.classes.length > 0) {
      state.lastUsedCategoryId = state.classes[0].id;
    }
  }

  function pushUndoState() {
    if (!state.coco) {
      return;
    }
    state.undoStack.push({
      coco: deepClone(state.coco),
      selectedAnnotationId: state.selectedAnnotationId,
      lastUsedCategoryId: state.lastUsedCategoryId,
    });
    if (state.undoStack.length > UNDO_LIMIT) {
      state.undoStack.shift();
    }
  }

  function markDirty(message) {
    state.dirty = true;
    updateSaveStatus(message || "Unsaved changes", "dirty");
  }

  function scheduleSave(message) {
    markDirty(message || "Unsaved changes");
    if (state.saveTimer) {
      window.clearTimeout(state.saveTimer);
    }
    state.saveTimer = window.setTimeout(function () {
      saveAnnotationsNow(false);
    }, SAVE_DEBOUNCE_MS);
  }

  function saveAnnotationsNow(force) {
    // Capture the image identity RIGHT NOW (before any state change can occur).
    // The caller may have changed state.selectedImageIndex by the time an async
    // timer fires, so we snapshot filename + image record here.
    var saveDataset = state.selectedDataset;
    var saveImageMeta = getCurrentImageMeta();
    var saveImageRecord = saveImageMeta ? getCurrentImageRecord(false) : null;

    if (!saveDataset || !state.coco) {
      return Promise.resolve();
    }
    if (!state.dirty && !force) {
      return Promise.resolve();
    }
    if (state.saveTimer) {
      window.clearTimeout(state.saveTimer);
      state.saveTimer = null;
    }
    if (state.isSaving) {
      state.pendingSave = true;
      return Promise.resolve();
    }

    // Per-image save: only send annotations for the current image.
    // The server merges them into the full on-disk COCO file so other images
    // are never lost even though state.coco.annotations may only hold a subset.
    if (!saveImageMeta || !saveImageRecord) {
      return Promise.resolve();
    }

    var imageAnnotations = state.coco.annotations.filter(function (a) {
      return a.image_id === saveImageRecord.id;
    });

    state.isSaving = true;
    updateSaveStatus("Saving annotations...", "saving");

    return postJson(
      "/api/datasets/" +
        encodeURIComponent(saveDataset) +
        "/annotations/image/" +
        encodeURIComponent(saveImageMeta.filename),
      { annotations: imageAnnotations, categories: state.coco.categories },
    )
      .then(function () {
        state.dirty = false;
        updateSaveStatus("Annotations saved", "saved");
      })
      .catch(function (error) {
        updateSaveStatus(error.message, "error");
        throw error;
      })
      .finally(function () {
        state.isSaving = false;
        if (state.pendingSave) {
          state.pendingSave = false;
          if (state.dirty) {
            scheduleSave("Queued save pending");
          }
        }
      });
  }

  function updateSelectedLabel() {
    if (!elements.selectedLabel) {
      return;
    } // Element not present in current view (implemented in D3)
    var selected = getSelectedAnnotation();
    var category;
    if (!selected) {
      elements.selectedLabel.textContent = "No selection";
      return;
    }
    category = getCategory(selected.category_id);
    elements.selectedLabel.textContent = category
      ? category.name
      : "Category " + selected.category_id;
  }

  // ── D4: selectTool — change the active annotation tool ───────────────────────
  function selectTool(id) {
    if (id === "undo") {
      undoLastAction();
      return;
    }
    if (id === "delete") {
      deleteSelectedAnnotation();
      return;
    }
    if (id === "ai") {
      // D7: scroll left panel to auto-annotate section
      var leftContent = document.getElementById("leftPanelContent");
      if (leftContent) {
        leftContent.scrollTop = leftContent.scrollHeight;
      }
      return;
    }
    state.activeTool = id;
    state.polygonVertices = [];
    state.lastPolygonClickTime = 0;
    state.interaction = null;
    updateToolbarState();
    renderCanvas();
  }

  function updateToolbarState() {
    // D4: generic active-class management for all mode tool buttons
    var palette = document.getElementById("editorToolPalette");
    if (palette) {
      palette.querySelectorAll("[data-tool]").forEach(function (btn) {
        var toolId = btn.getAttribute("data-tool");
        var isModeBtn =
          toolId === "select" || toolId === "bbox" || toolId === "polygon";
        var isActive = isModeBtn && toolId === state.activeTool;
        btn.classList.toggle("active", isActive);
        btn.setAttribute("aria-pressed", isActive ? "true" : "false");
      });
    }
    if (elements.activeToolLabel) {
      elements.activeToolLabel.textContent = state.activeTool + " tool";
    }

    if (elements.toolbarHint) {
      if (!state.selectedDataset) {
        elements.toolbarHint.textContent =
          "Select a dataset and image to start annotating.";
      } else if (!canAnnotateCurrentDataset()) {
        elements.toolbarHint.textContent =
          "This dataset uses image-folder classification. Annotation tools are disabled in classification mode.";
      } else if (state.activeTool === "polygon") {
        elements.toolbarHint.textContent =
          "Click to place vertices. Double-click (or Enter) to close the polygon and assign a class. Escape cancels.";
      } else {
        elements.toolbarHint.textContent =
          "Click and drag on the canvas to create a box. Click an existing box to select it, drag to move it, and drag a handle to resize it.";
      }
    }
  }

  function renderAll() {
    refreshClasses();
    renderClasses();
    renderClassPicker();
    updateSelectedLabel();
    updateToolbarState();
    updateAnnotationSummary();
    renderCanvas();
    // D3: full left-panel rebuild (layers list + class legend + annotation count)
    EditorView.renderLeftPanel();
  }

  // ── C3: Search / filter bar helpers ─────────────────────────────────────────
  // fetchImages(), BrowserView.renderGrid(), and BrowserView.renderPagination()
  // are declared later in this file. syncSearchBarUI() and initSearchBar() below
  // handle UI state sync and event wiring.

  /**
   * Sync the search bar UI elements (input value, active chip, sort select,
   * active split tab) to the current browserState.pagination values.
   * Call this when returning to the browser view to restore filter state.
   */
  function syncSearchBarUI() {
    var pg = browserState.pagination;
    var searchEl = elements.imageSearch;
    var sortEl = elements.sortSelect;
    var perPageEl = elements.perPageSelect;
    if (searchEl && searchEl.value !== (pg.search || "")) {
      searchEl.value = pg.search || "";
    }
    if (sortEl && sortEl.value !== (pg.sort || "name_asc")) {
      sortEl.value = pg.sort || "name_asc";
    }
    if (perPageEl && String(perPageEl.value) !== String(pg.perPage)) {
      perPageEl.value = String(pg.perPage);
    }
    // Sync filter chips active state
    var filterChipsEl = document.querySelector(".filter-chips");
    if (filterChipsEl) {
      filterChipsEl.querySelectorAll(".chip").forEach(function (chip) {
        var chipFilter = chip.getAttribute("data-filter");
        var isActive;
        if (
          pg.annotated === null ||
          pg.annotated === undefined ||
          pg.annotated === ""
        ) {
          isActive = chipFilter === "";
        } else {
          isActive = chipFilter === pg.annotated;
        }
        chip.classList.toggle("active", isActive);
      });
    }
    // Sync split tabs active state
    var splitTabsEl = document.getElementById("splitTabs");
    if (splitTabsEl) {
      splitTabsEl.querySelectorAll(".split-tab").forEach(function (tab) {
        var tabSplit = tab.getAttribute("data-split");
        var isActive =
          tabSplit === (pg.split || "") || (tabSplit === "" && !pg.split);
        tab.classList.toggle("active", isActive);
        tab.setAttribute("aria-selected", isActive ? "true" : "false");
      });
    }
  }

  /**
   * Wire up search, filter chip, sort, per-page, and pagination button events.
   * Called once during app initialisation.
   * C3: Search + Filter Bar
   */
  function initSearchBar() {
    var searchEl = elements.imageSearch;
    var sortEl = elements.sortSelect;
    var perPageEl = elements.perPageSelect;
    var prevBtn = elements.pagePrevBtn;
    var nextBtn = elements.pageNextBtn;
    var searchTimer = null;

    // ── Search input (300 ms debounce) ────────────────────────────────────────
    if (searchEl) {
      searchEl.addEventListener("input", function () {
        if (searchTimer) {
          window.clearTimeout(searchTimer);
        }
        searchTimer = window.setTimeout(function () {
          browserState.pagination.search = searchEl.value;
          browserState.pagination.page = 1;
          fetchImages();
        }, 300);
      });
    }

    // ── Filter chips: All / Annotated / Unannotated ───────────────────────────
    var filterChipsEl = document.querySelector(".filter-chips");
    if (filterChipsEl) {
      filterChipsEl.addEventListener("click", function (e) {
        var chip = e.target.closest(".chip");
        if (!chip) {
          return;
        }
        filterChipsEl.querySelectorAll(".chip").forEach(function (c) {
          c.classList.remove("active");
        });
        chip.classList.add("active");
        var filter = chip.getAttribute("data-filter");
        // data-filter="" → All (null); "true" → Annotated; "false" → Unannotated
        browserState.pagination.annotated = filter === "" ? null : filter;
        browserState.pagination.page = 1;
        fetchImages();
      });
    }

    // ── Sort dropdown ─────────────────────────────────────────────────────────
    if (sortEl) {
      sortEl.addEventListener("change", function () {
        browserState.pagination.sort = sortEl.value;
        browserState.pagination.page = 1;
        fetchImages();
      });
    }

    // ── Per-page dropdown ─────────────────────────────────────────────────────
    if (perPageEl) {
      perPageEl.addEventListener("change", function () {
        browserState.pagination.perPage = parseInt(perPageEl.value, 10) || 50;
        browserState.pagination.page = 1;
        fetchImages();
      });
    }

    // ── Pagination: prev / next buttons ───────────────────────────────────────
    if (prevBtn) {
      prevBtn.addEventListener("click", function () {
        if (browserState.pagination.page > 1) {
          browserState.pagination.page -= 1;
          fetchImages();
        }
      });
    }
    if (nextBtn) {
      nextBtn.addEventListener("click", function () {
        if (browserState.pagination.page < browserState.pagination.totalPages) {
          browserState.pagination.page += 1;
          fetchImages();
        }
      });
    }

    // ── Split tabs ────────────────────────────────────────────────────────────
    var splitTabsEl = document.getElementById("splitTabs");
    if (splitTabsEl) {
      splitTabsEl.addEventListener("click", function (e) {
        var tab = e.target.closest(".split-tab");
        if (!tab) {
          return;
        }
        splitTabsEl.querySelectorAll(".split-tab").forEach(function (t) {
          t.classList.remove("active");
          t.setAttribute("aria-selected", "false");
        });
        tab.classList.add("active");
        tab.setAttribute("aria-selected", "true");
        var split = tab.getAttribute("data-split") || null;
        browserState.pagination.split = split || null;
        browserState.pagination.page = 1;
        fetchImages();
      });
    }
  }

  // ── end C3 ────────────────────────────────────────────────────────────────

  function renderDatasets() {
    BrowserView.renderSidebar();
  }

  function renderImages() {
    if (!elements.imageList) {
      return;
    } // Element not present in current view layout (implemented in C2)
    elements.imageList.innerHTML = "";
    elements.imageError.hidden = true;
    if (!state.selectedDataset) {
      elements.imageEmpty.hidden = false;
      elements.imageEmpty.textContent = "No dataset selected yet.";
      elements.imageStatus.textContent = "Choose a dataset";
      return;
    }
    if (!state.images.length) {
      elements.imageEmpty.hidden = false;
      elements.imageEmpty.textContent =
        "This dataset does not contain any supported image files yet.";
      elements.imageStatus.textContent = "0 images";
      return;
    }

    elements.imageEmpty.hidden = true;
    elements.imageStatus.textContent = state.images.length + " images";
    state.images.forEach(function (image, index) {
      var button = document.createElement("button");
      var dimensions =
        image.width && image.height
          ? image.width + "x" + image.height
          : "size pending";
      var sizeLabel =
        typeof image.size_bytes === "number"
          ? formatBytes(image.size_bytes)
          : "unknown size";
      button.type = "button";
      button.className =
        "image-item" + (state.selectedImageIndex === index ? " active" : "");
      button.innerHTML =
        "<strong>" +
        escapeHtml(image.filename) +
        "</strong>" +
        '<div class="meta-row">' +
        buildTag(dimensions) +
        buildTag(sizeLabel) +
        (state.selectedImageIndex === index ? buildTag("active") : "") +
        "</div>";
      button.addEventListener("click", function () {
        selectImage(index);
      });
      elements.imageList.appendChild(button);
    });
  }

  /**
   * Return the index of an image in state.images[] by filename match.
   * Returns -1 when not found. Used by BrowserView.renderGrid for editor nav.
   */
  function findImageIndexByFilename(filename) {
    if (!state.images || !filename) {
      return -1;
    }
    for (var i = 0; i < state.images.length; i++) {
      if (state.images[i].filename === filename) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Fetch a page of images from the API using the current
   * browserState.pagination values, then render the grid and footer.
   * Supports page, per_page, sort, split, annotated, and search params.
   * C3: Called by initSearchBar() events, BrowserView.mount(), selectDataset().
   */
  function fetchImages() {
    var ds = state.selectedDataset;
    if (!ds) {
      BrowserView.renderGrid([]);
      BrowserView.renderPagination();
      return Promise.resolve();
    }

    var params = new URLSearchParams();
    params.set("page", String(browserState.pagination.page));
    params.set("per_page", String(browserState.pagination.perPage));
    if (browserState.pagination.sort) {
      params.set("sort", browserState.pagination.sort);
    }
    if (browserState.pagination.split) {
      params.set("split", browserState.pagination.split);
    }
    if (
      browserState.pagination.annotated !== null &&
      browserState.pagination.annotated !== undefined &&
      browserState.pagination.annotated !== ""
    ) {
      params.set("annotated", String(browserState.pagination.annotated));
    }
    if (browserState.pagination.search) {
      params.set("search", browserState.pagination.search);
    }

    return fetchJson(
      "/api/datasets/" +
        encodeURIComponent(ds) +
        "/images?" +
        params.toString(),
    )
      .then(function (data) {
        if (!data) {
          return;
        }
        browserState.pagination.total = data.total || 0;
        browserState.pagination.totalPages = data.total_pages || 0;
        BrowserView.renderGrid(data.images || []);
        BrowserView.renderPagination();
      })
      .catch(function () {
        BrowserView.renderGrid([]);
        BrowserView.renderPagination();
      });
  }

  function renderClasses() {
    var selected = getSelectedAnnotation();
    if (!elements.classList) {
      return;
    } // Class list not present in current view (implemented in D3)
    elements.classList.innerHTML = "";
    if (!state.classes.length) {
      var emptyButton = document.createElement("button");
      emptyButton.className = "chip-add";
      emptyButton.type = "button";
      emptyButton.textContent = "+ add class";
      emptyButton.addEventListener("click", function () {
        promptForCategory(false);
      });
      elements.classList.appendChild(emptyButton);
      elements.classHint.textContent =
        "Create a class first, then draw a box to assign it.";
      return;
    }

    state.classes.forEach(function (category, index) {
      var button = document.createElement("button");
      var isDefault = category.id === state.lastUsedCategoryId;
      var isSelected = selected && selected.category_id === category.id;
      button.type = "button";
      button.className =
        "chip-button" +
        (isDefault ? " active" : "") +
        (isSelected ? " selected" : "");
      button.innerHTML =
        '<span style="display:inline-flex;width:12px;height:12px;border-radius:50%;background:' +
        getCategoryColor(category.id) +
        '"></span>' +
        escapeHtml(category.name) +
        (index < 9
          ? ' <span class="hotkey">' + String(index + 1) + "</span>"
          : "");
      button.addEventListener("click", function () {
        handleCategorySelection(category.id);
      });
      elements.classList.appendChild(button);
    });

    var addButton = document.createElement("button");
    addButton.className = "chip-add";
    addButton.type = "button";
    addButton.textContent = "+ add class";
    addButton.addEventListener("click", function () {
      promptForCategory(false);
    });
    elements.classList.appendChild(addButton);
    elements.classHint.textContent =
      "Click a class chip to make it the default label, or while a box is selected, click a chip to reassign that box.";
  }

  function renderClassPicker() {
    elements.classPickerList.innerHTML = "";
    elements.classPickerError.hidden = true;
    if (!state.pendingAnnotation) {
      elements.classPicker.hidden = true;
      return;
    }

    elements.classPicker.hidden = false;
    if (!state.classes.length) {
      var createButton = document.createElement("button");
      createButton.type = "button";
      createButton.className = "picker-item";
      createButton.textContent = "+ create first class";
      createButton.addEventListener("click", function () {
        promptForCategory(true);
      });
      elements.classPickerList.appendChild(createButton);
      elements.classPickerHint.textContent =
        "Create a category before the new box can be committed.";
      return;
    }

    elements.classPickerHint.textContent =
      "Choose a category for the new " +
      (state.pendingAnnotation && state.pendingAnnotation.type === "polygon"
        ? "polygon"
        : "bounding box") +
      ". Press 1-9 for a quick shortcut.";
    state.classes.forEach(function (category, index) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "picker-item";
      button.innerHTML =
        '<span style="display:inline-flex;flex-shrink:0;width:12px;height:12px;border-radius:50%;background:' +
        getCategoryColor(category.id) +
        '"></span>' +
        '<span class="picker-label" title="' + escapeHtml(category.name) + '">' + escapeHtml(category.name) + '</span>' +
        (index < 9
          ? ' <span class="hotkey">' + String(index + 1) + "</span>"
          : "") +
        (category.id === state.lastUsedCategoryId
          ? ' <span class="tag">default</span>'
          : "");
      button.addEventListener("click", function () {
        assignPendingCategory(category.id);
      });
      elements.classPickerList.appendChild(button);
    });

    var addButton = document.createElement("button");
    addButton.type = "button";
    addButton.className = "picker-item picker-add-class";
    addButton.textContent = "+ add class";
    addButton.style.display = "none"; // hidden — use Classes section instead
    addButton.addEventListener("click", function () {
      promptForCategory(true);
    });
    elements.classPickerList.appendChild(addButton);
  }

  function updateAnnotationSummary() {
    var annotations = getCurrentAnnotations();
    var polyCount = annotations.filter(hasPolygon).length;
    var bboxCount = annotations.length - polyCount;
    var parts = [];
    if (bboxCount > 0) {
      parts.push(bboxCount + (bboxCount === 1 ? " box" : " boxes"));
    }
    if (polyCount > 0) {
      parts.push(polyCount + (polyCount === 1 ? " polygon" : " polygons"));
    }
    if (elements.annotationCount) {
      elements.annotationCount.textContent = parts.length
        ? parts.join(", ")
        : "0 boxes";
    }
  }

  function updateViewerCopy(extraText) {
    var imageMeta = getCurrentImageMeta();
    var bits;
    if (!imageMeta) {
      return;
    }
    bits = [state.selectedDataset];
    if (imageMeta.width && imageMeta.height) {
      bits.push(imageMeta.width + "x" + imageMeta.height);
    }
    if (typeof imageMeta.size_bytes === "number") {
      bits.push(formatBytes(imageMeta.size_bytes));
    }
    if (extraText) {
      bits.push(extraText);
    }
    if (elements.viewerTitle) {
      elements.viewerTitle.textContent = imageMeta.filename;
    }
    if (elements.viewerSubtitle) {
      elements.viewerSubtitle.textContent = bits.filter(Boolean).join("  |  ");
    }
  }

  function fitCanvasToImage() {
    var image = elements.imageLoader;
    var canvasArea = elements.canvasStack;
    var areaH = canvasArea ? canvasArea.clientHeight : 0;
    var maxWidth = Math.max(200, elements.canvasStack.clientWidth - 12);
    var maxHeight =
      areaH > 80
        ? Math.max(320, areaH - 20)
        : Math.max(320, Math.min(window.innerHeight * 0.82, 1200));
    var scale = Math.min(
      maxWidth / image.naturalWidth,
      maxHeight / image.naturalHeight,
      1,
    );
    var width = Math.max(1, Math.round(image.naturalWidth * scale));
    var height = Math.max(1, Math.round(image.naturalHeight * scale));
    var imageRecord;

    elements.imageCanvas.width = width;
    elements.imageCanvas.height = height;
    elements.imageCanvas.style.width = width + "px";
    elements.imageCanvas.style.height = height + "px";
    elements.imageCanvas.style.display = "block";
    elements.canvasPlaceholder.hidden = true;

    state.view.imageWidth = image.naturalWidth;
    state.view.imageHeight = image.naturalHeight;
    state.view.canvasWidth = width;
    state.view.canvasHeight = height;
    state.view.scaleX = width / image.naturalWidth;
    state.view.scaleY = height / image.naturalHeight;

    imageRecord = getCurrentImageRecord(true);
    if (imageRecord) {
      imageRecord.width = image.naturalWidth;
      imageRecord.height = image.naturalHeight;
    }
  }

  function imageToCanvasRect(bbox) {
    var zoom = editorState.zoom;
    return {
      x: bbox[0] * state.view.scaleX * zoom,
      y: bbox[1] * state.view.scaleY * zoom,
      width: bbox[2] * state.view.scaleX * zoom,
      height: bbox[3] * state.view.scaleY * zoom,
    };
  }

  function canvasToImagePoint(point) {
    var zoom = editorState.zoom;
    return {
      x: clamp(point.x / (state.view.scaleX * zoom), 0, state.view.imageWidth),
      y: clamp(point.y / (state.view.scaleY * zoom), 0, state.view.imageHeight),
    };
  }

  function getCanvasPoint(event) {
    var rect = elements.imageCanvas.getBoundingClientRect();
    return {
      x:
        (event.clientX - rect.left) * (elements.imageCanvas.width / rect.width),
      y:
        (event.clientY - rect.top) *
        (elements.imageCanvas.height / rect.height),
    };
  }

  function normaliseBox(startPoint, currentPoint) {
    var x1 = clamp(
      Math.min(startPoint.x, currentPoint.x),
      0,
      state.view.imageWidth,
    );
    var y1 = clamp(
      Math.min(startPoint.y, currentPoint.y),
      0,
      state.view.imageHeight,
    );
    var x2 = clamp(
      Math.max(startPoint.x, currentPoint.x),
      0,
      state.view.imageWidth,
    );
    var y2 = clamp(
      Math.max(startPoint.y, currentPoint.y),
      0,
      state.view.imageHeight,
    );
    return [x1, y1, x2, y2];
  }

  function drawLabel(text, x, y, color) {
    var label = text || "object";
    var textWidth;
    var labelX;
    var labelY;
    canvasContext.save();
    canvasContext.font = "12px Bahnschrift, Trebuchet MS, Segoe UI, sans-serif";
    textWidth = canvasContext.measureText(label).width;
    labelX = Math.max(6, x);
    labelY = Math.max(18, y - 8);
    canvasContext.fillStyle = color;
    canvasContext.fillRect(labelX - 4, labelY - 16, textWidth + 10, 18);
    canvasContext.fillStyle = "#ffffff";
    canvasContext.fillText(label, labelX + 1, labelY - 3);
    canvasContext.restore();
  }

  function getHandlePoints(rect) {
    return {
      nw: { x: rect.x, y: rect.y },
      n: { x: rect.x + rect.width / 2, y: rect.y },
      ne: { x: rect.x + rect.width, y: rect.y },
      e: { x: rect.x + rect.width, y: rect.y + rect.height / 2 },
      se: { x: rect.x + rect.width, y: rect.y + rect.height },
      s: { x: rect.x + rect.width / 2, y: rect.y + rect.height },
      sw: { x: rect.x, y: rect.y + rect.height },
      w: { x: rect.x, y: rect.y + rect.height / 2 },
    };
  }

  function drawHandles(annotation) {
    var rect = imageToCanvasRect(annotation.bbox);
    var points = getHandlePoints(rect);
    canvasContext.save();
    canvasContext.fillStyle = "#ffffff";
    canvasContext.strokeStyle = "#172331";
    canvasContext.lineWidth = 1.2;
    Object.keys(points).forEach(function (handleName) {
      var point = points[handleName];
      canvasContext.beginPath();
      canvasContext.rect(
        point.x - HANDLE_SIZE / 2,
        point.y - HANDLE_SIZE / 2,
        HANDLE_SIZE,
        HANDLE_SIZE,
      );
      canvasContext.fill();
      canvasContext.stroke();
    });
    canvasContext.restore();
  }

  function drawDraftBox(xyxy, color) {
    var rect = imageToCanvasRect(xyxyToXywh(xyxy));
    canvasContext.save();
    canvasContext.strokeStyle = color;
    canvasContext.fillStyle = color + "18";
    canvasContext.lineWidth = 2;
    canvasContext.setLineDash([8, 6]);
    canvasContext.fillRect(rect.x, rect.y, rect.width, rect.height);
    canvasContext.strokeRect(rect.x, rect.y, rect.width, rect.height);
    canvasContext.restore();
  }

  function drawPolygonVertexHandles(points) {
    canvasContext.save();
    canvasContext.fillStyle = "#ffffff";
    canvasContext.strokeStyle = "#172331";
    canvasContext.lineWidth = 1.2;
    points.forEach(function (p) {
      canvasContext.beginPath();
      canvasContext.arc(p.x, p.y, VERTEX_HANDLE_SIZE / 2, 0, Math.PI * 2);
      canvasContext.fill();
      canvasContext.stroke();
    });
    canvasContext.restore();
  }

  function drawPolygonAnnotation(annotation, isSelected) {
    var flat = annotation.segmentation && annotation.segmentation[0];
    var points;
    var color;
    var category;
    var label;
    var rect;
    var i;
    if (!flat || flat.length < 6) {
      return;
    }
    points = segmentationToCanvasPoints(flat);
    color = getCategoryColor(annotation.category_id);
    category = getCategory(annotation.category_id);
    label = category ? category.name : "Category " + annotation.category_id;
    if (typeof annotation.score === "number") {
      label += " " + annotation.score.toFixed(2);
    }

    canvasContext.save();
    canvasContext.beginPath();
    canvasContext.moveTo(points[0].x, points[0].y);
    for (i = 1; i < points.length; i += 1) {
      canvasContext.lineTo(points[i].x, points[i].y);
    }
    canvasContext.closePath();
    canvasContext.fillStyle = color + "30";
    canvasContext.strokeStyle = color;
    canvasContext.lineWidth = isSelected ? 3 : 2;
    canvasContext.fill();
    canvasContext.stroke();
    canvasContext.restore();

    if (annotation.bbox && annotation.bbox[2] > 0 && annotation.bbox[3] > 0) {
      rect = imageToCanvasRect(annotation.bbox);
      canvasContext.save();
      canvasContext.strokeStyle = color + "88";
      canvasContext.lineWidth = 1;
      canvasContext.setLineDash([4, 4]);
      canvasContext.strokeRect(rect.x, rect.y, rect.width, rect.height);
      canvasContext.restore();
      drawLabel(label, rect.x, rect.y, color);
    }

    if (isSelected) {
      drawPolygonVertexHandles(points);
    }
  }

  function drawInProgressPolygon(vertices, cursorPoint) {
    var canvasVerts;
    var color = CATEGORY_COLORS[0];
    var last;
    var cur;
    var i;
    if (!vertices.length) {
      return;
    }
    canvasVerts = vertices.map(function (v) {
      return {
        x: v.x * state.view.scaleX * editorState.zoom,
        y: v.y * state.view.scaleY * editorState.zoom,
      };
    });

    canvasContext.save();
    if (canvasVerts.length >= 2) {
      canvasContext.beginPath();
      canvasContext.moveTo(canvasVerts[0].x, canvasVerts[0].y);
      for (i = 1; i < canvasVerts.length; i += 1) {
        canvasContext.lineTo(canvasVerts[i].x, canvasVerts[i].y);
      }
      if (cursorPoint) {
        canvasContext.lineTo(
          cursorPoint.x * state.view.scaleX * editorState.zoom,
          cursorPoint.y * state.view.scaleY * editorState.zoom,
        );
      }
      if (canvasVerts.length >= 3) {
        canvasContext.closePath();
        canvasContext.fillStyle = color + "18";
        canvasContext.fill();
      }
      canvasContext.strokeStyle = color;
      canvasContext.lineWidth = 2;
      canvasContext.setLineDash([6, 4]);
      canvasContext.stroke();
      canvasContext.setLineDash([]);
    }

    canvasContext.fillStyle = "#ffffff";
    canvasContext.strokeStyle = color;
    canvasContext.lineWidth = 1.5;
    canvasVerts.forEach(function (v) {
      canvasContext.beginPath();
      canvasContext.arc(v.x, v.y, VERTEX_HANDLE_SIZE / 2, 0, Math.PI * 2);
      canvasContext.fill();
      canvasContext.stroke();
    });

    if (cursorPoint && canvasVerts.length > 0) {
      last = canvasVerts[canvasVerts.length - 1];
      cur = {
        x: cursorPoint.x * state.view.scaleX * editorState.zoom,
        y: cursorPoint.y * state.view.scaleY * editorState.zoom,
      };
      canvasContext.beginPath();
      canvasContext.moveTo(last.x, last.y);
      canvasContext.lineTo(cur.x, cur.y);
      canvasContext.strokeStyle = color + "aa";
      canvasContext.lineWidth = 1.5;
      canvasContext.setLineDash([4, 4]);
      canvasContext.stroke();
      canvasContext.setLineDash([]);
    }

    canvasContext.restore();
  }

  function renderCanvas() {
    var annotations;
    updateAnnotationSummary();
    // D3: sync layer row highlight to current selection (lightweight, no DOM rebuild)
    EditorView.syncLayersHighlight();
    // D5: sync annotation properties panel (lightweight coords update during drag)
    EditorView.syncPropsPanel();
    if (!elements.imageLoader.complete || !elements.imageLoader.naturalWidth) {
      return;
    }
    var _rw = Math.round(state.view.canvasWidth * editorState.zoom);
    var _rh = Math.round(state.view.canvasHeight * editorState.zoom);
    if (
      elements.imageCanvas.width !== _rw ||
      elements.imageCanvas.height !== _rh
    ) {
      elements.imageCanvas.width = _rw;
      elements.imageCanvas.height = _rh;
      elements.imageCanvas.style.width = _rw + "px";
      elements.imageCanvas.style.height = _rh + "px";
    }
    canvasContext.clearRect(0, 0, _rw, _rh);
    canvasContext.drawImage(elements.imageLoader, 0, 0, _rw, _rh);
    if (editorState.showAnnotations !== false) {
    annotations = getCurrentAnnotations();
    annotations.forEach(function (annotation) {
      var selectedIds = state.selectedAnnotationIds || [];
      var selected = selectedIds.length > 0
        ? selectedIds.indexOf(annotation.id) !== -1
        : annotation.id === state.selectedAnnotationId;
      if (hasPolygon(annotation)) {
        drawPolygonAnnotation(annotation, selected);
        return;
      }
      var rect = imageToCanvasRect(annotation.bbox);
      var category = getCategory(annotation.category_id);
      var color = getCategoryColor(annotation.category_id);
      var label = category
        ? category.name
        : "Category " + annotation.category_id;
      if (typeof annotation.score === "number") {
        label += " " + annotation.score.toFixed(2);
      }

      canvasContext.save();
      if (annotation._draft) {
        // D7: AI-draft annotations rendered with dashed border
        canvasContext.fillStyle = color + "18";
        canvasContext.strokeStyle = color;
        canvasContext.lineWidth = selected ? 2.5 : 1.5;
        canvasContext.setLineDash([6, 4]);
        canvasContext.fillRect(rect.x, rect.y, rect.width, rect.height);
        canvasContext.strokeRect(rect.x, rect.y, rect.width, rect.height);
        canvasContext.setLineDash([]);
      } else {
        canvasContext.fillStyle = color + "22";
        canvasContext.strokeStyle = color;
        canvasContext.lineWidth = selected ? 3 : 2;
        canvasContext.fillRect(rect.x, rect.y, rect.width, rect.height);
        canvasContext.strokeRect(rect.x, rect.y, rect.width, rect.height);
      }
      canvasContext.restore();
      drawLabel(label, rect.x, rect.y, color);
      if (selected) {
        drawHandles(annotation);
      }
    });

    if (state.activeTool === "polygon" && state.polygonVertices.length > 0) {
      drawInProgressPolygon(
        state.polygonVertices,
        state.interaction && state.interaction.currentPoint
          ? state.interaction.currentPoint
          : null,
      );
    }

    if (state.interaction && state.interaction.mode === "drawing") {
      drawDraftBox(
        normaliseBox(
          state.interaction.startPoint,
          state.interaction.currentPoint,
        ),
        "#0e6c78",
      );
    }
    if (
      state.pendingAnnotation &&
      Array.isArray(state.pendingAnnotation.xyxy)
    ) {
      drawDraftBox(state.pendingAnnotation.xyxy, "#cc6b2c");
    }
    if (
      state.pendingAnnotation &&
      state.pendingAnnotation.type === "polygon" &&
      Array.isArray(state.pendingAnnotation.segmentation)
    ) {
      var pendingFlat = state.pendingAnnotation.segmentation;
      var pendingBbox = polyBboxFromFlat(pendingFlat);
      var pendingRect = imageToCanvasRect(pendingBbox);
      canvasContext.save();
      canvasContext.strokeStyle = "#cc6b2c";
      canvasContext.lineWidth = 2;
      canvasContext.setLineDash([6, 4]);
      canvasContext.strokeRect(
        pendingRect.x,
        pendingRect.y,
        pendingRect.width,
        pendingRect.height,
      );
      canvasContext.setLineDash([]);
      canvasContext.restore();
    }
    } // end showAnnotations

  function hitTestHandle(annotation, canvasPoint) {
    var rect = imageToCanvasRect(annotation.bbox);
    var points = getHandlePoints(rect);
    var handleNames = Object.keys(points);
    var index;
    for (index = 0; index < handleNames.length; index += 1) {
      var handleName = handleNames[index];
      var point = points[handleName];
      if (
        Math.abs(canvasPoint.x - point.x) <= HANDLE_SIZE &&
        Math.abs(canvasPoint.y - point.y) <= HANDLE_SIZE
      ) {
        return handleName;
      }
    }
    return null;
  }

  function hitTestPolygonVertex(annotation, canvasPoint) {
    var flat = annotation.segmentation && annotation.segmentation[0];
    var points;
    var i;
    if (!flat || flat.length < 6) {
      return -1;
    }
    points = segmentationToCanvasPoints(flat);
    for (i = 0; i < points.length; i += 1) {
      if (
        Math.abs(canvasPoint.x - points[i].x) <= VERTEX_HANDLE_SIZE &&
        Math.abs(canvasPoint.y - points[i].y) <= VERTEX_HANDLE_SIZE
      ) {
        return i;
      }
    }
    return -1;
  }

  function hitTestPolygon(annotation, canvasPoint) {
    var flat = annotation.segmentation && annotation.segmentation[0];
    var points;
    var x;
    var y;
    var inside;
    var i;
    var j;
    var xi;
    var yi;
    var xj;
    var yj;
    if (!flat || flat.length < 6) {
      return false;
    }
    points = segmentationToCanvasPoints(flat);
    x = canvasPoint.x;
    y = canvasPoint.y;
    inside = false;
    for (i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
      xi = points[i].x;
      yi = points[i].y;
      xj = points[j].x;
      yj = points[j].y;
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
        inside = !inside;
      }
    }
    return inside;
  }

  function hitTestAnnotation(canvasPoint) {
    var annotations = getCurrentAnnotations().slice().reverse();
    return (
      annotations.find(function (annotation) {
        if (hasPolygon(annotation)) {
          return hitTestPolygon(annotation, canvasPoint);
        }
        var rect = imageToCanvasRect(annotation.bbox);
        return (
          canvasPoint.x >= rect.x &&
          canvasPoint.x <= rect.x + rect.width &&
          canvasPoint.y >= rect.y &&
          canvasPoint.y <= rect.y + rect.height
        );
      }) || null
    );
  }

  function resizeBox(originalBBox, handleName, imagePoint) {
    var x1 = originalBBox[0];
    var y1 = originalBBox[1];
    var x2 = originalBBox[0] + originalBBox[2];
    var y2 = originalBBox[1] + originalBBox[3];

    if (handleName.indexOf("w") !== -1) {
      x1 = clamp(imagePoint.x, 0, x2 - MIN_DRAW_SIZE);
    }
    if (handleName.indexOf("e") !== -1) {
      x2 = clamp(imagePoint.x, x1 + MIN_DRAW_SIZE, state.view.imageWidth);
    }
    if (handleName.indexOf("n") !== -1) {
      y1 = clamp(imagePoint.y, 0, y2 - MIN_DRAW_SIZE);
    }
    if (handleName.indexOf("s") !== -1) {
      y2 = clamp(imagePoint.y, y1 + MIN_DRAW_SIZE, state.view.imageHeight);
    }
    return [x1, y1, x2 - x1, y2 - y1];
  }

  function moveBox(originalBBox, deltaX, deltaY) {
    var width = originalBBox[2];
    var height = originalBBox[3];
    var nextX = clamp(
      originalBBox[0] + deltaX,
      0,
      state.view.imageWidth - width,
    );
    var nextY = clamp(
      originalBBox[1] + deltaY,
      0,
      state.view.imageHeight - height,
    );
    return [nextX, nextY, width, height];
  }

  function setCanvasCursor(cursor) {
    elements.imageCanvas.style.cursor =
      cursor || (canAnnotateCurrentDataset() ? "crosshair" : "default");
  }

  function openClassPicker(pendingData) {
    state.pendingAnnotation = pendingData;
    renderClassPicker();
    renderCanvas();
  }

  function closeClassPicker() {
    state.pendingAnnotation = null;
    state.polygonVertices = [];
    state.lastPolygonClickTime = 0;
    renderClassPicker();
    renderCanvas();
  }

  function promptForCategory(keepPickerOpen) {
    var name;
    var trimmed;
    var existing;
    if (!state.coco) {
      return;
    }
    name = window.prompt("Class name", "");
    if (!name) {
      return;
    }
    trimmed = name.trim();
    if (!trimmed) {
      return;
    }
    existing = state.coco.categories.find(function (category) {
      return category.name.toLowerCase() === trimmed.toLowerCase();
    });
    if (existing) {
      state.lastUsedCategoryId = existing.id;
      renderAll();
      return;
    }

    state.coco.categories.push({
      id: nextId(state.coco.categories),
      name: trimmed,
      supercategory: trimmed,
    });
    state.lastUsedCategoryId =
      state.coco.categories[state.coco.categories.length - 1].id;
    markDirty("New class added");
    renderAll();
    if (keepPickerOpen) {
      renderClassPicker();
    }
  }

  function closePendingPolygon() {
    var flat;
    if (state.polygonVertices.length < 3) {
      return;
    }
    flat = [];
    state.polygonVertices.forEach(function (p) {
      flat.push(
        clamp(Math.round(p.x), 0, state.view.imageWidth),
        clamp(Math.round(p.y), 0, state.view.imageHeight),
      );
    });
    state.polygonVertices = [];
    state.lastPolygonClickTime = 0;
    state.interaction = null;
    state.pendingAnnotation = { type: "polygon", segmentation: flat };
    renderClassPicker();
    renderCanvas();
  }

  function assignPendingCategory(categoryId) {
    var imageRecord;
    var xywh;
    var flat;
    var newAnnotation;
    if (!state.pendingAnnotation || !state.coco) {
      return;
    }
    imageRecord = getCurrentImageRecord(true);
    if (!imageRecord) {
      return;
    }
    pushUndoState();

    if (state.pendingAnnotation.type === "polygon") {
      flat = state.pendingAnnotation.segmentation;
      xywh = polyBboxFromFlat(flat);
      newAnnotation = {
        id: nextId(state.coco.annotations),
        image_id: imageRecord.id,
        category_id: categoryId,
        bbox: xywh,
        area: xywh[2] * xywh[3],
        iscrowd: 0,
        segmentation: [flat],
      };
    } else {
      xywh = xyxyToXywh(state.pendingAnnotation.xyxy);
      newAnnotation = {
        id: nextId(state.coco.annotations),
        image_id: imageRecord.id,
        category_id: categoryId,
        bbox: xywh,
        area: xywh[2] * xywh[3],
        iscrowd: 0,
        segmentation: [],
      };
    }

    state.coco.annotations.push(newAnnotation);
    state.selectedAnnotationId = newAnnotation.id;
    state.selectedAnnotationIds = [newAnnotation.id];
    state.lastUsedCategoryId = categoryId;
    state.pendingAnnotation = null;
    scheduleSave(
      newAnnotation.segmentation.length
        ? "Polygon added"
        : "Bounding box added",
    );
    renderAll();
  }

  function handleCategorySelection(categoryId) {
    var selected = getSelectedAnnotation();
    state.lastUsedCategoryId = categoryId;
    if (state.pendingAnnotation) {
      assignPendingCategory(categoryId);
      return;
    }
    if (selected) {
      pushUndoState();
      selected.category_id = categoryId;
      scheduleSave("Class updated");
    }
    renderAll();
  }

  function deleteSelectedAnnotation() {
    var index;
    if (!state.coco || state.selectedAnnotationId === null) {
      return;
    }
    index = state.coco.annotations.findIndex(function (annotation) {
      return annotation.id === state.selectedAnnotationId;
    });
    if (index === -1) {
      state.selectedAnnotationId = null;
      renderAll();
      return;
    }
    pushUndoState();
    state.coco.annotations.splice(index, 1);
    state.selectedAnnotationId = null;
    state.selectedAnnotationIds = [];
    scheduleSave("Bounding box removed");
    renderAll();
  }

  function undoLastAction() {
    var snapshot = state.undoStack.pop();
    if (!snapshot) {
      return;
    }
    state.coco = normaliseCoco(snapshot.coco);
    state.selectedAnnotationId = snapshot.selectedAnnotationId;
    state.lastUsedCategoryId = snapshot.lastUsedCategoryId;
    state.pendingAnnotation = null;
    scheduleSave("Undo applied");
    renderAll();
  }

  function applyCurrentImageLoad() {
    fitCanvasToImage();
    EditorView.renderBottomBar(); // D2: refresh zoom label after fit
    updateViewerCopy(getCurrentAnnotations().length + " boxes");
    renderAll();
  }

  function imageUrl(datasetName, fileName) {
    return (
      "/api/datasets/" +
      encodeURIComponent(datasetName) +
      "/images/" +
      encodeURIComponent(fileName)
    );
  }

  function selectImage(index, options) {
    var config = options || {};
    var imageMeta;
    if (index < 0 || index >= state.images.length || !state.selectedDataset) {
      return Promise.resolve();
    }
    // Save any unsaved work on the PREVIOUS image before changing the index.
    // This captures the correct image identity since selectedImageIndex has
    // not yet changed.
    if (
      state.dirty &&
      state.selectedImageIndex >= 0 &&
      state.selectedImageIndex !== index
    ) {
      saveAnnotationsNow(false);
    }
    state.selectedImageIndex = index;
    state.selectedAnnotationId = null;
    state.selectedAnnotationIds = [];
    state.interaction = null;
    state.pendingAnnotation = null;
    state.polygonVertices = [];
    state.lastPolygonClickTime = 0;
    // D4: reset tool to "select" when switching images
    state.activeTool = "select";
    // D2: reset zoom/pan when loading a new image
    editorState.zoom = 1;
    editorState.panX = 0;
    editorState.panY = 0;
    if (elements.canvasStack) {
      elements.canvasStack.scrollLeft = 0;
      elements.canvasStack.scrollTop = 0;
    }
    renderImages();
    updateCounts();
    imageMeta = getCurrentImageMeta();
    elements.viewerTitle.textContent = imageMeta.filename;
    if (elements.viewerSubtitle) {
      elements.viewerSubtitle.textContent =
        "Loading image from " + state.selectedDataset + ".";
    }
    // Update placeholder title to show loading state
    var placeholderTitle = document.getElementById("canvasPlaceholderTitle");
    if (placeholderTitle) {
      placeholderTitle.textContent = "Loading\u2026";
    }
    // D1: keep top-bar filename + counter in sync
    // Compute position relative to the active nav context (split filter).
    (function () {
      var _navSplit = state.editorNavSplit;
      var _navIndices = [];
      if (_navSplit) {
        for (var _ni = 0; _ni < state.images.length; _ni++) {
          if ((state.images[_ni].split || null) === _navSplit) {
            _navIndices.push(_ni);
          }
        }
      }
      if (!_navIndices.length) {
        _navIndices = state.images.map(function (_, i) {
          return i;
        });
      }
      var _navPos = _navIndices.indexOf(index);
      if (_navPos < 0) {
        _navPos = 0;
      } else {
        _navPos = _navIndices.indexOf(index);
      }
      EditorView.renderTopBar(
        state.selectedDataset,
        _navPos,
        _navIndices.length,
        imageMeta.filename,
      );
    })();
    elements.canvasPlaceholder.hidden = false;
    elements.imageCanvas.style.display = "none";
    elements.placeholderText.textContent =
      "Loading " + imageMeta.filename + "...";

    return new Promise(function (resolve) {
      // Capture dataset + filename in case state changes mid-flight.
      var loadDataset = state.selectedDataset;
      var loadFilename = imageMeta.filename;

      function finishLoad() {
        applyCurrentImageLoad();
        if (!config.silentFocus) {
          elements.imageCanvas.focus();
        }
        resolve();
      }

      elements.imageLoader.onload = function () {
        // Fetch per-image annotations lazily — avoids loading the full 500MB+
        // COCO annotations blob upfront.  We clear any stale annotations for
        // this image and replace them with the freshly fetched ones.
        fetchJson(
          "/api/datasets/" +
            encodeURIComponent(loadDataset) +
            "/annotations/image/" +
            encodeURIComponent(loadFilename),
        )
          .then(function (data) {
            // Only apply if the user hasn't navigated away while fetching.
            if (
              state.selectedDataset !== loadDataset ||
              !state.images[state.selectedImageIndex] ||
              state.images[state.selectedImageIndex].filename !== loadFilename
            ) {
              return;
            }
            if (data && typeof data.image_id === "number") {
              // Remove stale entries for this image, then append the fresh ones.
              state.coco.annotations = state.coco.annotations.filter(
                function (a) {
                  return a.image_id !== data.image_id;
                },
              );
              if (Array.isArray(data.annotations)) {
                Array.prototype.push.apply(
                  state.coco.annotations,
                  data.annotations,
                );
              }
            }
          })
          .catch(function () {
            // Silently ignore — editor will show empty annotations for this image.
          })
          .finally(function () {
            finishLoad();
          });
      };
      elements.imageLoader.onerror = function () {
        setPlaceholder(
          "Image failed to load",
          "The server returned an unreadable image response for " +
            imageMeta.filename +
            ".",
        );
        resolve();
      };
      elements.imageLoader.src =
        imageUrl(state.selectedDataset, imageMeta.filename) +
        "?t=" +
        Date.now();
    });
  }

  function loadDatasetArtifacts(datasetName) {
    // Fetch stats and annotations in parallel.
    // Do NOT fetch /images here — that would return the full unPaginated list
    // for large datasets (COCO ~118k images = minutes).  Instead we derive
    // state.images from the COCO annotations doc, which is already loaded.
    // The thumbnail browser grid is populated separately by fetchImages() using
    // true server-side pagination.
    return Promise.all([
      fetchJson(
        "/api/datasets/" + encodeURIComponent(datasetName) + "/stats",
      ).catch(function () {
        return null;
      }),
      fetchJson(
        "/api/datasets/" + encodeURIComponent(datasetName) + "/annotations",
      ).catch(function () {
        return createEmptyCoco();
      }),
    ]);
  }

  function selectDataset(datasetName, options) {
    var config = options || {};
    var savePromise = Promise.resolve();
    if (!datasetName) {
      return Promise.resolve();
    }
    if (
      state.dirty &&
      state.selectedDataset &&
      state.selectedDataset !== datasetName
    ) {
      savePromise = saveAnnotationsNow(true).catch(function () {
        return null;
      });
    }

    return savePromise.then(function () {
      state.selectedDataset = datasetName;
      state.selectedImageIndex = -1;
      state.selectedAnnotationId = null;
      state.datasetStats = null;
      state.selectedDatasetType = null;
      state.editorNavSplit = null;
      state.images = [];
      state.coco = createEmptyCoco();
      state.classes = [];
      state.pendingAnnotation = null;
      state.interaction = null;
      state.undoStack = [];
      state.polygonVertices = [];
      state.lastPolygonClickTime = 0;
      state.dirty = false;
      updateSaveStatus("Loaded", "idle");
      renderDatasets();
      renderImages();
      renderClasses();
      updateCounts();
      setPlaceholder(
        "Loading dataset",
        "Fetching image list and annotations for " + datasetName + ".",
      );
      applyLoadingState(
        true,
        "Loading \u201c" + datasetName + "\u201d\u2026",
        "Fetching annotations and image list",
      );

      return loadDatasetArtifacts(datasetName)
        .then(function (results) {
          // results[0] = stats (or null), results[1] = coco annotations
          state.datasetStats = results[0] || null;
          state.selectedDatasetType =
            state.datasetStats && state.datasetStats.type
              ? state.datasetStats.type
              : null;
          state.coco = normaliseCoco(results[1]);
          // Build state.images from COCO metadata — no filesystem scan needed.
          // Each entry needs at minimum {filename} for editor navigation.
          var cocoImages =
            state.coco && state.coco.images ? state.coco.images : [];
          state.images = cocoImages.map(function (img) {
            return {
              filename: img.file_name
                ? img.file_name.indexOf("/") !== -1
                  ? img.file_name.split("/").pop()
                  : img.file_name
                : String(img.id),
              width: img.width || 0,
              height: img.height || 0,
              annotation_count: 0,
            };
          });
          // For non-COCO or empty datasets, state.images stays empty here;
          // the browser grid will still populate via fetchImages() below.
          refreshClasses();
          renderClasses();
          renderImages();
          renderDatasets();
          updateCounts();
          // C2: update browser header, split tabs, and thumbnail grid
          BrowserView.renderHeader(state.datasetStats, datasetName);
          BrowserView.renderSplitTabs(state.datasetStats);
          browserState.pagination.page = 1;
          browserState.pagination.split = null;
          fetchImages();
          if (state.images.length) {
            return selectImage(0, { silentFocus: config.silentFocus });
          }

          // Non-COCO dataset (e.g. ImageFolder): no image list from annotations.
          // Fetch filenames in the background using server-side pagination so the
          // editor prev/next navigation works.  For these datasets there are
          // typically far fewer images than a COCO dataset so this is fast.
          fetchJson(
            "/api/datasets/" +
              encodeURIComponent(datasetName) +
              "/images?page=1&per_page=9999",
          )
            .then(function (data) {
              var raw =
                data && Array.isArray(data.images)
                  ? data.images
                  : Array.isArray(data)
                    ? data
                    : [];
              if (raw.length && datasetName === state.selectedDataset) {
                state.images = raw;
                renderImages();
                updateCounts();
                // Trigger editor on the first image now that we have filenames
                selectImage(0, { silentFocus: config.silentFocus });
              }
            })
            .catch(function () {
              // Silently ignore — the grid still works without editor nav
            });

          if (elements.viewerTitle) {
            elements.viewerTitle.textContent = datasetName;
          }
          if (elements.viewerSubtitle) {
            elements.viewerSubtitle.textContent =
              "Dataset loaded, but there are no supported image files to display yet.";
          }
          setPlaceholder(
            datasetName,
            "Add JPG, PNG, BMP, TIFF, or WEBP files to this dataset to populate the canvas.",
          );
          return null;
        })
        .catch(function (error) {
          state.images = [];
          state.coco = createEmptyCoco();
          renderImages();
          renderClasses();
          if (elements.imageError) {
            elements.imageError.hidden = false;
            elements.imageError.textContent = error.message;
          }
          setPlaceholder(
            "Unable to load dataset",
            "The selected dataset could not be opened through the API.",
          );
        })
        .finally(function () {
          applyLoadingState(false);
          if (window.innerWidth <= 860) {
            toggleSidebar(false);
          }
        });
    });
  }

  function moveSelection(offset) {
    var nextIndex;
    if (!state.images.length) {
      return;
    }
    nextIndex =
      state.selectedImageIndex < 0 ? 0 : state.selectedImageIndex + offset;
    if (nextIndex < 0 || nextIndex >= state.images.length) {
      return;
    }
    selectImage(nextIndex);
  }

  /**
   * Start a background rescan for *name* and poll until done, then update
   * the sidebar with the real image count.
   */
  function rescanDataset(name) {
    if (state.rescanStatus[name] === "running") {
      return; // Already polling — ignore duplicate click
    }
    state.rescanStatus[name] = "running";
    renderDatasets();

    fetch("/api/datasets/" + encodeURIComponent(name) + "/rescan", {
      method: "POST",
    }).catch(function () {
      // Network error — clear status and let the user retry
      delete state.rescanStatus[name];
      renderDatasets();
    });

    function poll() {
      fetchJson("/api/datasets/" + encodeURIComponent(name) + "/rescan")
        .then(function (status) {
          if (status.status === "done") {
            // Update the matching entry in state.datasets with fresh values
            var cache = status.cache || {};
            state.datasets = state.datasets.map(function (d) {
              if (d.name !== name) return d;
              return {
                name: d.name,
                image_count:
                  cache.image_count != null ? cache.image_count : d.image_count,
                has_annotations:
                  cache.has_annotations != null
                    ? cache.has_annotations
                    : d.has_annotations,
                type: cache.type || d.type,
                cache_valid: true,
              };
            });
            delete state.rescanStatus[name];
            renderDatasets();
          } else if (status.status === "error") {
            delete state.rescanStatus[name];
            renderDatasets();
          } else {
            // still running
            window.setTimeout(poll, 2000);
          }
        })
        .catch(function () {
          delete state.rescanStatus[name];
          renderDatasets();
        });
    }

    // Give server a beat to start the thread before first poll
    window.setTimeout(poll, 1000);
  }

  function loadDatasets() {
    elements.datasetStatus.textContent = "Loading";
    elements.datasetEmpty.hidden = false;
    elements.datasetEmpty.textContent = "Loading datasets...";
    return fetchJson("/api/datasets")
      .then(function (datasets) {
        state.datasets = Array.isArray(datasets) ? datasets : [];
        renderDatasets();
        if (state.datasets.length > 0) {
          return selectDataset(state.datasets[0].name, { silentFocus: true });
        }
        setPlaceholder(
          "No datasets available",
          "Create or point the server at a dataset directory to begin browsing images.",
        );
        return null;
      })
      .catch(function (error) {
        elements.datasetStatus.textContent = "Error";
        elements.datasetError.hidden = false;
        elements.datasetError.textContent = error.message;
        elements.datasetEmpty.hidden = true;
        setPlaceholder(
          "Unable to load datasets",
          "The sidebar could not reach /api/datasets. Check the server and refresh.",
        );
      })
      .finally(function () {
        updateCounts();
      });
  }

  function handleCanvasPointerDown(event) {
    var canvasPoint;
    var imagePoint;
    var selected;
    var handle;
    var hitAnnotation;
    var now;
    var elapsed;
    var vertexIdx;

    // D8: Space+drag pan or middle-click pan — handled before annotation guards
    if (editorState.panMode || event.button === 1) {
      var _pc = elements.canvasStack;
      state.interaction = {
        mode: "panning",
        pointerId: event.pointerId,
        startScrollLeft: _pc ? _pc.scrollLeft : 0,
        startScrollTop: _pc ? _pc.scrollTop : 0,
        startClientX: event.clientX,
        startClientY: event.clientY,
      };
      elements.imageCanvas.setPointerCapture(event.pointerId);
      setCanvasCursor("grabbing");
      return;
    }

    if (
      !elements.imageLoader.complete ||
      !elements.imageLoader.naturalWidth ||
      !canAnnotateCurrentDataset() ||
      state.pendingAnnotation
    ) {
      return;
    }

    elements.imageCanvas.focus();
    canvasPoint = getCanvasPoint(event);
    imagePoint = canvasToImagePoint(canvasPoint);

    if (state.activeTool === "polygon") {
      now = Date.now();
      elapsed = now - state.lastPolygonClickTime;
      state.lastPolygonClickTime = now;

      if (elapsed < 350 && state.polygonVertices.length > 0) {
        closePendingPolygon();
        return;
      }

      selected = getSelectedAnnotation();
      if (
        selected &&
        hasPolygon(selected) &&
        state.polygonVertices.length === 0
      ) {
        vertexIdx = hitTestPolygonVertex(selected, canvasPoint);
        if (vertexIdx >= 0) {
          state.interaction = {
            mode: "dragging_poly_vertex",
            pointerId: event.pointerId,
            annotationId: selected.id,
            vertexIndex: vertexIdx,
            mutated: false,
          };
          elements.imageCanvas.setPointerCapture(event.pointerId);
          setCanvasCursor("crosshair");
          return;
        }
      }

      if (state.polygonVertices.length === 0) {
        hitAnnotation = hitTestAnnotation(canvasPoint);
        if (hitAnnotation) {
          state.selectedAnnotationId = hitAnnotation.id;
          state.selectedAnnotationIds = [hitAnnotation.id];
          state.interaction = null;
          renderClasses();
          renderCanvas();
          updateSelectedLabel();
          if (editorState.leftPanelTab === "attributes") { EditorView.renderLeftPanel(); }
          return;
        }
        state.selectedAnnotationId = null;
        state.selectedAnnotationIds = [];
      }

      state.polygonVertices.push(imagePoint);
      if (!state.interaction || state.interaction.mode !== "drawing_polygon") {
        state.interaction = {
          mode: "drawing_polygon",
          currentPoint: imagePoint,
        };
      }
      renderCanvas();
      return;
    }

    selected = getSelectedAnnotation();
    handle =
      selected && !hasPolygon(selected)
        ? hitTestHandle(selected, canvasPoint)
        : null;

    if (handle && selected) {
      state.interaction = {
        mode: "resizing",
        pointerId: event.pointerId,
        handle: handle,
        startPoint: imagePoint,
        originalBBox: selected.bbox.slice(),
        annotationId: selected.id,
        mutated: false,
      };
      elements.imageCanvas.setPointerCapture(event.pointerId);
      setCanvasCursor(RESIZE_CURSOR[handle]);
      return;
    }

    hitAnnotation = hitTestAnnotation(canvasPoint);
    if (hitAnnotation) {
      state.selectedAnnotationId = hitAnnotation.id;
      state.selectedAnnotationIds = [hitAnnotation.id];
      state.interaction = {
        mode: "selecting",
        pointerId: event.pointerId,
        startPoint: imagePoint,
        originalBBox: hitAnnotation.bbox.slice(),
        annotationId: hitAnnotation.id,
        mutated: false,
      };
      elements.imageCanvas.setPointerCapture(event.pointerId);
      setCanvasCursor("move");
      renderClasses();
      renderCanvas();
      updateSelectedLabel();
      if (editorState.leftPanelTab === "attributes") { EditorView.renderLeftPanel(); }
      return;
    }

    state.selectedAnnotationId = null;
    state.selectedAnnotationIds = [];
    state.interaction = {
      mode: "drawing",
      pointerId: event.pointerId,
      startPoint: imagePoint,
      currentPoint: imagePoint,
    };
    elements.imageCanvas.setPointerCapture(event.pointerId);
    setCanvasCursor("crosshair");
    renderClasses();
    renderCanvas();
    updateSelectedLabel();
  }

  function handleCanvasPointerMove(event) {
    var canvasPoint;
    var imagePoint;
    var selected;
    var handle;
    var annotation;
    var ann;
    var flat;

    // D8: Pan drag — scroll the container instead of moving panX/panY
    if (state.interaction && state.interaction.mode === "panning") {
      if (state.interaction.pointerId !== event.pointerId) {
        return;
      }
      var _pm = elements.canvasStack;
      if (_pm) {
        _pm.scrollLeft =
          state.interaction.startScrollLeft -
          (event.clientX - state.interaction.startClientX);
        _pm.scrollTop =
          state.interaction.startScrollTop -
          (event.clientY - state.interaction.startClientY);
      }
      return;
    }

    if (!elements.imageLoader.complete || !elements.imageLoader.naturalWidth) {
      return;
    }

    canvasPoint = getCanvasPoint(event);
    imagePoint = canvasToImagePoint(canvasPoint);

    if (state.activeTool === "polygon" && state.polygonVertices.length > 0) {
      if (!state.interaction) {
        state.interaction = { mode: "drawing_polygon" };
      }
      state.interaction.currentPoint = imagePoint;
      renderCanvas();
      return;
    }

    if (
      state.interaction &&
      state.interaction.mode === "dragging_poly_vertex"
    ) {
      if (state.interaction.pointerId !== event.pointerId) {
        return;
      }
      ann =
        state.coco &&
        state.coco.annotations.find(function (a) {
          return a.id === state.interaction.annotationId;
        });
      if (ann && hasPolygon(ann)) {
        if (!state.interaction.mutated) {
          pushUndoState();
          state.interaction.mutated = true;
        }
        flat = ann.segmentation[0];
        var vi = state.interaction.vertexIndex;
        flat[vi * 2] = clamp(imagePoint.x, 0, state.view.imageWidth);
        flat[vi * 2 + 1] = clamp(imagePoint.y, 0, state.view.imageHeight);
        ann.bbox = polyBboxFromFlat(flat);
        ann.area = ann.bbox[2] * ann.bbox[3];
        renderCanvas();
      }
      return;
    }

    if (!state.interaction) {
      selected = getSelectedAnnotation();
      handle =
        selected && !hasPolygon(selected)
          ? hitTestHandle(selected, canvasPoint)
          : null;
      if (handle) {
        setCanvasCursor(RESIZE_CURSOR[handle]);
        return;
      }
      if (selected && hasPolygon(selected)) {
        if (hitTestPolygonVertex(selected, canvasPoint) >= 0) {
          setCanvasCursor("crosshair");
          return;
        }
      }
      if (hitTestAnnotation(canvasPoint)) {
        setCanvasCursor("move");
        return;
      }
      setCanvasCursor();
      return;
    }

    if (state.interaction.pointerId !== event.pointerId) {
      return;
    }

    if (state.interaction.mode === "drawing") {
      state.interaction.currentPoint = imagePoint;
      renderCanvas();
      return;
    }

    annotation = getSelectedAnnotation();
    if (!annotation) {
      return;
    }

    if (state.interaction.mode === "selecting") {
      if (distance(state.interaction.startPoint, imagePoint) >= 2) {
        state.interaction.mode = "moving";
      } else {
        return;
      }
    }

    if (state.interaction.mode === "moving") {
      if (!state.interaction.mutated) {
        pushUndoState();
        state.interaction.mutated = true;
      }
      if (hasPolygon(annotation)) {
        var dx = imagePoint.x - state.interaction.startPoint.x;
        var dy = imagePoint.y - state.interaction.startPoint.y;
        var origFlat = state.interaction.originalFlat;
        if (!origFlat) {
          origFlat = annotation.segmentation[0].slice();
          state.interaction.originalFlat = origFlat;
          state.interaction.originalBBox = annotation.bbox.slice();
        }
        var newFlat = origFlat.slice();
        for (var fi = 0; fi + 1 < newFlat.length; fi += 2) {
          newFlat[fi] = clamp(origFlat[fi] + dx, 0, state.view.imageWidth);
          newFlat[fi + 1] = clamp(
            origFlat[fi + 1] + dy,
            0,
            state.view.imageHeight,
          );
        }
        annotation.segmentation[0] = newFlat;
        annotation.bbox = polyBboxFromFlat(newFlat);
        annotation.area = annotation.bbox[2] * annotation.bbox[3];
      } else {
        annotation.bbox = moveBox(
          state.interaction.originalBBox,
          imagePoint.x - state.interaction.startPoint.x,
          imagePoint.y - state.interaction.startPoint.y,
        );
      }
      renderClasses();
      renderCanvas();
      return;
    }

    if (state.interaction.mode === "resizing") {
      if (!state.interaction.mutated) {
        pushUndoState();
        state.interaction.mutated = true;
      }
      annotation.bbox = resizeBox(
        state.interaction.originalBBox,
        state.interaction.handle,
        imagePoint,
      );
      renderClasses();
      renderCanvas();
    }
  }

  function handleCanvasPointerUp(event) {
    var xyxy;
    var width;
    var height;
    var didMutate;
    var actionLabel;

    // D8: End pan drag
    if (state.interaction && state.interaction.mode === "panning") {
      if (state.interaction.pointerId !== event.pointerId) {
        return;
      }
      state.interaction = null;
      // Restore cursor: grab if still in pan mode, otherwise default
      setCanvasCursor(editorState.panMode ? "grab" : undefined);
      return;
    }

    if (
      state.interaction &&
      state.interaction.mode === "dragging_poly_vertex"
    ) {
      if (state.interaction.pointerId !== event.pointerId) {
        return;
      }
      didMutate = state.interaction.mutated;
      state.interaction = null;
      if (didMutate) {
        scheduleSave("Polygon vertex moved");
      }
      renderCanvas();
      return;
    }

    if (!state.interaction || state.interaction.pointerId !== event.pointerId) {
      return;
    }

    if (state.interaction.mode === "drawing_polygon") {
      state.interaction = null;
      return;
    }

    if (state.interaction.mode === "drawing") {
      xyxy = normaliseBox(
        state.interaction.startPoint,
        state.interaction.currentPoint,
      );
      width = xyxy[2] - xyxy[0];
      height = xyxy[3] - xyxy[1];
      state.interaction = null;
      if (width >= MIN_DRAW_SIZE && height >= MIN_DRAW_SIZE) {
        openClassPicker({ xyxy: xyxy });
      } else {
        renderCanvas();
      }
      return;
    }

    if (
      state.interaction.mode === "moving" ||
      state.interaction.mode === "resizing"
    ) {
      didMutate = state.interaction.mutated;
      actionLabel =
        state.interaction.mode === "moving"
          ? "Annotation moved"
          : "Bounding box resized";
      state.interaction = null;
      if (didMutate) {
        scheduleSave(actionLabel);
      }
      renderClasses();
      renderCanvas();
      updateSelectedLabel();
      return;
    }

    state.interaction = null;
    renderCanvas();
    updateSelectedLabel();
  }

  function handleKeyDown(event) {
    var target = event.target;
    if (target && /input|textarea|select/i.test(target.tagName)) {
      return;
    }
    // E2: Browser view keyboard shortcuts (Arrow grid nav + Enter to open)
    if (Router.current === "browse") {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        var cards = Array.from(
          document.querySelectorAll("#thumbnailGrid .thumb-card"),
        );
        if (!cards.length) {
          return;
        }
        var idx = cards.indexOf(document.activeElement);
        if (idx < 0) {
          cards[0].focus();
          cards[0].scrollIntoView({ block: "nearest" });
          return;
        }
        var next = event.key === "ArrowRight" ? idx + 1 : idx - 1;
        if (next >= 0 && next < cards.length) {
          cards[next].focus();
          cards[next].scrollIntoView({ block: "nearest" });
        }
      } else if (event.key === "Enter") {
        var focused = document.activeElement;
        if (focused && focused.classList.contains("thumb-card")) {
          event.preventDefault();
          focused.click();
        }
      }
      return;
    }
    // D2: only process canvas shortcuts when the editor view is active
    if (Router.current !== "edit") {
      return;
    }

    // D8: Space key activates pan mode (drag-to-pan)
    if (event.key === " " && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      if (!editorState.panMode) {
        editorState.panMode = true;
        setCanvasCursor("grab");
      }
      return;
    }

    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveAnnotationsNow(true);
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      undoLastAction();
      return;
    }
    // E2: Ctrl+Y — Redo (Coming Soon, no-op)
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
      event.preventDefault();
      return;
    }
    if (state.pendingAnnotation && /^[1-9]$/.test(event.key)) {
      var shortcutIndex = Number(event.key) - 1;
      if (state.classes[shortcutIndex]) {
        event.preventDefault();
        assignPendingCategory(state.classes[shortcutIndex].id);
      }
      return;
    }
    // E2: Arrow keys navigate images (prev/next) in editor view
    if (event.key === "ArrowRight") {
      event.preventDefault();
      EditorView.navigateImage(1);
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      EditorView.navigateImage(-1);
      return;
    }
    if (event.key.toLowerCase() === "v") {
      selectTool("select");
      return;
    }
    if (event.key.toLowerCase() === "b") {
      selectTool("bbox");
      return;
    }
    if (event.key.toLowerCase() === "p") {
      selectTool("polygon");
      return;
    }
    if (event.key.toLowerCase() === "a") {
      selectTool("ai");
      return;
    }
    // H: toggle annotation visibility
    if (event.key.toLowerCase() === "h") {
      editorState.showAnnotations = !editorState.showAnnotations;
      var visBtn = document.getElementById("annotationVisibilityBtn");
      if (visBtn) {
        visBtn.classList.toggle("is-active", !editorState.showAnnotations);
        visBtn.title = editorState.showAnnotations ? "Hide annotations (H)" : "Show annotations (H)";
      }
      renderCanvas();
      return;
    }
    if (event.key === "Enter") {
      if (state.activeTool === "polygon" && state.polygonVertices.length >= 3) {
        event.preventDefault();
        closePendingPolygon();
      }
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      if (state.pendingAnnotation) {
        return;
      }
      event.preventDefault();
      deleteSelectedAnnotation();
      return;
    }
    if (event.key === "Escape") {
      if (state.pendingAnnotation) {
        closeClassPicker();
        return;
      }
      if (state.activeTool === "polygon" && state.polygonVertices.length > 0) {
        state.polygonVertices = [];
        state.lastPolygonClickTime = 0;
        state.interaction = null;
        renderCanvas();
        return;
      }
      if (state.interaction) {
        state.interaction = null;
        renderCanvas();
        return;
      }
      if (elements.shell.classList.contains("sidebar-open")) {
        toggleSidebar(false);
        return;
      }
      // E3: close mobile editor left-panel overlay on Escape
      var editorPanel = document.getElementById("editorLeftPanel");
      var editorBackdrop = document.getElementById("mobileEditorBackdrop");
      if (editorPanel && editorPanel.classList.contains("open")) {
        editorPanel.classList.remove("open");
        if (editorBackdrop) {
          editorBackdrop.classList.remove("open");
        }
        var panelToggle = document.getElementById("editorPanelToggle");
        if (panelToggle) {
          panelToggle.setAttribute("aria-expanded", "false");
        }
        return;
      }
      if (state.selectedAnnotationId !== null) {
        state.selectedAnnotationId = null;
        renderClasses();
        renderCanvas();
        updateSelectedLabel();
      }
    }
  }

  function handleBeforeUnload(event) {
    if (!state.dirty) {
      return;
    }
    event.preventDefault();
    event.returnValue = "";
  }

  // Note: sidebarToggle and mobileNavBackdrop click events are handled by
  // the inline script in index.html (B1). Do not add duplicate listeners here.

  // D4: tool palette click listeners now wired via EditorView.initToolPalette()

  // ── D1: Editor top-bar navigation (Prev / Next with wrap + auto-save) ────────
  var editorPrevBtn = document.getElementById("editorPrevBtn");
  if (editorPrevBtn) {
    editorPrevBtn.addEventListener("click", function () {
      EditorView.navigateImage(-1);
    });
  }

  var editorNextBtn = document.getElementById("editorNextBtn");
  if (editorNextBtn) {
    editorNextBtn.addEventListener("click", function () {
      EditorView.navigateImage(1);
    });
  }

  // ── Annotation visibility toggle ─────────────────────────────────────────────
  var annotationVisibilityBtn = document.getElementById("annotationVisibilityBtn");
  if (annotationVisibilityBtn) {
    annotationVisibilityBtn.addEventListener("click", function () {
      editorState.showAnnotations = !editorState.showAnnotations;
      annotationVisibilityBtn.classList.toggle("is-active", !editorState.showAnnotations);
      annotationVisibilityBtn.title = editorState.showAnnotations
        ? "Hide annotations (H)"
        : "Show annotations (H)";
      renderCanvas();
    });
  }

  // ── Mark as Reviewed toggle ───────────────────────────────────────────────────
  var reviewToggleBtn = document.getElementById("reviewToggleBtn");
  if (reviewToggleBtn) {
    reviewToggleBtn.addEventListener("click", function () {
      var imageRecord = getCurrentImageRecord(true);
      if (!imageRecord || !state.selectedDataset) { return; }
      imageRecord.reviewed = !imageRecord.reviewed;
      reviewToggleBtn.classList.toggle("is-active", !!imageRecord.reviewed);
      reviewToggleBtn.title = imageRecord.reviewed ? "Unmark reviewed" : "Mark as reviewed";
      // Persist to backend
      var dataset = state.selectedDataset;
      var filename = getCurrentImageMeta() ? getCurrentImageMeta().filename : null;
      if (filename) {
        fetch(
          "/api/datasets/" + encodeURIComponent(dataset) +
          "/images/" + encodeURIComponent(filename) + "/reviewed",
          {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reviewed: !!imageRecord.reviewed }),
          }
        ).catch(function () {});
      }
    });
  }

  // ── Overflow menu: Move to split ─────────────────────────────────────────────
  function moveCurrentImageToSplit(targetSplit) {
    var imageMeta = getCurrentImageMeta();
    if (!imageMeta || !state.selectedDataset) {
      return;
    }
    if (window._closeOverflowDropdown) { window._closeOverflowDropdown(); }
    fetch(
      "/api/datasets/" + encodeURIComponent(state.selectedDataset) + "/move",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: imageMeta.filename,
          target_split: targetSplit,
        }),
      }
    )
      .then(function (resp) { return resp.json(); })
      .then(function (data) {
        if (data && data.error) {
          alert("Move failed: " + data.error);
        } else {
          // Navigate to next image, then refresh
          EditorView.navigateImage(1);
        }
      })
      .catch(function () { alert("Move request failed."); });
  }

  var overflowMoveTrain = document.getElementById("overflowMoveTrain");
  if (overflowMoveTrain) {
    overflowMoveTrain.addEventListener("click", function () { moveCurrentImageToSplit("train"); });
  }
  var overflowMoveValid = document.getElementById("overflowMoveValid");
  if (overflowMoveValid) {
    overflowMoveValid.addEventListener("click", function () { moveCurrentImageToSplit("val"); });
  }
  var overflowMoveTest = document.getElementById("overflowMoveTest");
  if (overflowMoveTest) {
    overflowMoveTest.addEventListener("click", function () { moveCurrentImageToSplit("test"); });
  }

  // ── D2: Zoom helper — zoom toward container centre (scroll-based) ────────────
  function zoomFromCenter(oldZoom, newZoom) {
    var container = elements.canvasStack;
    if (!container) {
      editorState.zoom = newZoom;
      return;
    }
    var scaleX = state.view.scaleX;
    var scaleY = state.view.scaleY;
    var vx = container.clientWidth / 2;
    var vy = container.clientHeight / 2;
    var imgX, imgY;
    if (scaleX > 0 && scaleY > 0 && oldZoom > 0) {
      imgX = (container.scrollLeft + vx) / (scaleX * oldZoom);
      imgY = (container.scrollTop + vy) / (scaleY * oldZoom);
    }
    editorState.zoom = newZoom;
    renderCanvas();
    if (imgX !== undefined) {
      container.scrollLeft = Math.round(imgX * scaleX * newZoom - vx);
      container.scrollTop = Math.round(imgY * scaleY * newZoom - vy);
    }
  }

  // ── D8: Pan clamp — no-op; browser scroll handles clamping ───────────────────
  function clampPan() {
    // no-op: container overflow:auto clamps scroll automatically
  }

  // ── D1: Bottom-bar zoom controls (updated to 10% steps, max 500%) ────────────
  var zoomInBtn = document.getElementById("zoomInBtn");
  if (zoomInBtn) {
    zoomInBtn.addEventListener("click", function () {
      var oldZoom = editorState.zoom;
      var newZoom = Math.min(oldZoom * 1.1, 5.0);
      zoomFromCenter(oldZoom, newZoom); // includes renderCanvas + scroll anchor
      EditorView.renderBottomBar();
    });
  }

  var zoomOutBtn = document.getElementById("zoomOutBtn");
  if (zoomOutBtn) {
    zoomOutBtn.addEventListener("click", function () {
      var oldZoom = editorState.zoom;
      var newZoom = Math.max(oldZoom / 1.1, 0.1);
      zoomFromCenter(oldZoom, newZoom); // includes renderCanvas + scroll anchor
      EditorView.renderBottomBar();
    });
  }

  var zoomResetBtn = document.getElementById("zoomResetBtn");
  if (zoomResetBtn) {
    zoomResetBtn.addEventListener("click", function () {
      editorState.zoom = 1;
      EditorView.renderBottomBar();
      renderCanvas();
      var _rc = elements.canvasStack;
      if (_rc) {
        _rc.scrollLeft = 0;
        _rc.scrollTop = 0;
      }
    });
  }

  // ── D8: Brightness / Contrast popover ────────────────────────────────────────
  (function () {
    var brightnessBtn = document.getElementById("brightnessBtn");
    var brightnessPopover = document.getElementById("brightnessPopover");
    var brightnessSlider = document.getElementById("brightnessSlider");
    var contrastSlider = document.getElementById("contrastSlider");
    var brightnessReset = document.getElementById("brightnessReset");

    if (brightnessBtn && brightnessPopover) {
      // Toggle popover open/close
      brightnessBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        var nowHidden = !brightnessPopover.hidden;
        brightnessPopover.hidden = nowHidden;
        brightnessBtn.classList.toggle("active", !nowHidden);
        brightnessBtn.setAttribute(
          "aria-expanded",
          nowHidden ? "false" : "true",
        );
      });

      // Close on click outside
      document.addEventListener("click", function (e) {
        if (
          !brightnessPopover.hidden &&
          !brightnessBtn.contains(e.target) &&
          !brightnessPopover.contains(e.target)
        ) {
          brightnessPopover.hidden = true;
          brightnessBtn.classList.remove("active");
          brightnessBtn.setAttribute("aria-expanded", "false");
        }
      });
    }

    if (brightnessSlider) {
      brightnessSlider.addEventListener("input", function () {
        editorState.brightness = Number(this.value);
        var valEl = document.getElementById("brightnessValue");
        if (valEl) {
          valEl.textContent = this.value + "%";
        }
        EditorView.applyBrightness();
      });
    }

    if (contrastSlider) {
      contrastSlider.addEventListener("input", function () {
        editorState.contrast = Number(this.value);
        var valEl = document.getElementById("contrastValue");
        if (valEl) {
          valEl.textContent = this.value + "%";
        }
        EditorView.applyBrightness();
      });
    }

    if (brightnessReset) {
      brightnessReset.addEventListener("click", function () {
        editorState.brightness = 100;
        editorState.contrast = 100;
        if (brightnessSlider) {
          brightnessSlider.value = 100;
        }
        if (contrastSlider) {
          contrastSlider.value = 100;
        }
        var bValEl = document.getElementById("brightnessValue");
        if (bValEl) {
          bValEl.textContent = "100%";
        }
        var cValEl = document.getElementById("contrastValue");
        if (cValEl) {
          cValEl.textContent = "100%";
        }
        EditorView.applyBrightness();
      });
    }
  })();

  elements.classPickerClose.addEventListener("click", function () {
    closeClassPicker();
  });

  elements.imageCanvas.addEventListener("pointerdown", handleCanvasPointerDown);
  elements.imageCanvas.addEventListener("pointermove", handleCanvasPointerMove);
  elements.imageCanvas.addEventListener("pointerup", handleCanvasPointerUp);
  elements.imageCanvas.addEventListener("pointercancel", handleCanvasPointerUp);

  window.addEventListener("keydown", handleKeyDown);

  // D8: keyup — release Space pan mode
  window.addEventListener("keyup", function (event) {
    if (event.key === " " && editorState.panMode) {
      editorState.panMode = false;
      // If not actively panning, restore cursor to default tool cursor
      if (!state.interaction || state.interaction.mode !== "panning") {
        setCanvasCursor(undefined);
      }
    }
  });

  window.addEventListener("beforeunload", handleBeforeUnload);
  window.addEventListener("resize", function () {
    if (window.innerWidth > 860) {
      toggleSidebar(false);
    }
    if (elements.imageLoader.complete && elements.imageLoader.naturalWidth) {
      fitCanvasToImage();
      renderCanvas();
    }
  });

  window._annotate = {
    state: state,
    browserState: browserState,
    editorState: editorState,
    Router: Router,
    BrowserView: BrowserView,
    EditorView: EditorView,
    fetchImages: fetchImages,
    renderPagination: BrowserView.renderPagination,
    syncSearchBarUI: syncSearchBarUI,
    // D2: expose canvas helpers for testing
    renderCanvas: renderCanvas,
    imageToCanvasRect: imageToCanvasRect,
    canvasToImagePoint: canvasToImagePoint,
    segmentationToCanvasPoints: segmentationToCanvasPoints,
    // D3: expose panel helpers for testing
    renderLeftPanel: function () {
      EditorView.renderLeftPanel();
    },
    // D9: expose attributes/raw-data tab helpers for testing
    renderAttributesTab: function () {
      var container = document.getElementById("leftPanelContent");
      if (container) {
        EditorView.renderAttributesTab(container);
      }
    },
    renderRawDataTab: function () {
      var container = document.getElementById("leftPanelContent");
      if (container) {
        EditorView.renderRawDataTab(container);
      }
    },
    // D4: expose tool selection for testing
    selectTool: selectTool,
    TOOLS: TOOLS,
    // D5: expose annotation properties helpers for testing
    renderAnnotationProperties: function () {
      EditorView.renderAnnotationProperties();
    },
    syncPropsPanel: function () {
      EditorView.syncPropsPanel();
    },
    // D7: expose auto-annotate helpers for testing
    runAutoAnnotate: function () {
      EditorView.runAutoAnnotate();
    },
    addCandidatesToCanvas: function (candidates, mode) {
      EditorView.addCandidatesToCanvas(candidates, mode);
    },
    setAutoAnnotateStatus: function (status, message) {
      EditorView._setAutoAnnotateStatus(status, message);
    },
    AUTO_ANNOTATE_MODES: AUTO_ANNOTATE_MODES,
    // D8: expose zoom/pan/brightness helpers for testing
    clampPan: clampPan,
    applyBrightness: function () {
      EditorView.applyBrightness();
    },
  };
  // C3: wire up search/filter/sort events once on page load
  initSearchBar();
  // D3: wire up left-panel tab switching once on page load
  EditorView.initLeftPanel();
  // D4: wire up tool palette click delegation once on page load
  EditorView.initToolPalette();