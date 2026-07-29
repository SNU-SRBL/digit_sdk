"use strict";

const $ = (id) => document.getElementById(id);
const view = $("view");
const viewCtx = view.getContext("2d", {willReadFrequently: true});
const sourceCanvas = document.createElement("canvas");
const sourceCtx = sourceCanvas.getContext("2d", {willReadFrequently: true});
const differenceCanvas = document.createElement("canvas");
const differenceCtx = differenceCanvas.getContext("2d", {willReadFrequently: true});
const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d", {willReadFrequently: true});
const overlayCanvas = document.createElement("canvas");
const overlayCtx = overlayCanvas.getContext("2d", {willReadFrequently: true});

const state = {
  samples: [],
  index: 0,
  tool: "brush",
  drawing: false,
  dirty: false,
  lastPoint: null,
  polygon: [],
  undo: [],
  redo: [],
  autosaveTimer: null,
  backgroundAvailable: false,
};

function setStatus(message, kind = "idle") {
  $("save-status").textContent = message;
  $("save-status").dataset.state = kind;
}

function setCanvasSize(width, height) {
  for (const canvas of [view, sourceCanvas, differenceCanvas, maskCanvas, overlayCanvas]) {
    canvas.width = width;
    canvas.height = height;
  }
  applyZoom();
}

function applyZoom() {
  const scale = Number($("zoom").value) / 100;
  view.style.width = `${view.width * scale}px`;
  view.style.height = `${view.height * scale}px`;
  $("zoom-value").value = `${$("zoom").value}%`;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${url}`));
    image.src = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
  });
}

function makeDifference(background) {
  const width = sourceCanvas.width;
  const height = sourceCanvas.height;
  differenceCtx.clearRect(0, 0, width, height);
  differenceCtx.drawImage(background, 0, 0, width, height);
  const base = sourceCtx.getImageData(0, 0, width, height);
  const bg = differenceCtx.getImageData(0, 0, width, height);
  const output = differenceCtx.createImageData(width, height);
  for (let i = 0; i < output.data.length; i += 4) {
    output.data[i] = Math.min(255, Math.abs(base.data[i] - bg.data[i]) * 3);
    output.data[i + 1] = Math.min(255, Math.abs(base.data[i + 1] - bg.data[i + 1]) * 3);
    output.data[i + 2] = Math.min(255, Math.abs(base.data[i + 2] - bg.data[i + 2]) * 3);
    output.data[i + 3] = 255;
  }
  differenceCtx.putImageData(output, 0, 0);
}

function rebuildOverlay() {
  const width = maskCanvas.width;
  const height = maskCanvas.height;
  const mask = maskCtx.getImageData(0, 0, width, height);
  const overlay = overlayCtx.createImageData(width, height);
  const alpha = Math.round(Number($("opacity").value) * 2.55);
  for (let i = 0; i < mask.data.length; i += 4) {
    if (mask.data[i] >= 128) {
      overlay.data[i] = 255;
      overlay.data[i + 1] = 54;
      overlay.data[i + 2] = 48;
      overlay.data[i + 3] = alpha;
    }
  }
  overlayCtx.putImageData(overlay, 0, 0);
}

function render() {
  if (!view.width) return;
  const showDifference = $("difference").checked && state.backgroundAvailable;
  viewCtx.clearRect(0, 0, view.width, view.height);
  viewCtx.drawImage(showDifference ? differenceCanvas : sourceCanvas, 0, 0);
  rebuildOverlay();
  viewCtx.drawImage(overlayCanvas, 0, 0);
  if (state.polygon.length) {
    viewCtx.save();
    viewCtx.strokeStyle = "#fff";
    viewCtx.fillStyle = "#ff665f";
    viewCtx.lineWidth = 1.5;
    viewCtx.setLineDash([5, 4]);
    viewCtx.beginPath();
    state.polygon.forEach((point, index) => {
      if (index === 0) viewCtx.moveTo(point.x, point.y);
      else viewCtx.lineTo(point.x, point.y);
    });
    viewCtx.stroke();
    viewCtx.setLineDash([]);
    for (const point of state.polygon) {
      viewCtx.beginPath();
      viewCtx.arc(point.x, point.y, 2.6, 0, Math.PI * 2);
      viewCtx.fill();
    }
    viewCtx.restore();
  }
}

function pointFromEvent(event) {
  const bounds = view.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) * view.width / bounds.width,
    y: (event.clientY - bounds.top) * view.height / bounds.height,
  };
}

function snapshot() {
  return maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
}

function pushUndo() {
  state.undo.push(snapshot());
  if (state.undo.length > 40) state.undo.shift();
  state.redo = [];
  updateHistoryButtons();
}

function updateHistoryButtons() {
  $("undo").disabled = state.undo.length === 0;
  $("redo").disabled = state.redo.length === 0;
}

function markDirty() {
  state.dirty = true;
  setStatus("Unsaved changes", "dirty");
  clearTimeout(state.autosaveTimer);
  state.autosaveTimer = setTimeout(() => saveMask(false), 900);
}

function stroke(from, to) {
  maskCtx.save();
  maskCtx.strokeStyle = state.tool === "eraser" ? "#000" : "#fff";
  maskCtx.lineWidth = Number($("brush-size").value);
  maskCtx.lineCap = "round";
  maskCtx.lineJoin = "round";
  maskCtx.beginPath();
  maskCtx.moveTo(from.x, from.y);
  maskCtx.lineTo(to.x, to.y);
  maskCtx.stroke();
  maskCtx.restore();
  render();
}

function finishPolygon() {
  if (state.polygon.length < 3) {
    state.polygon = [];
    render();
    return;
  }
  pushUndo();
  maskCtx.save();
  maskCtx.fillStyle = "#fff";
  maskCtx.beginPath();
  maskCtx.moveTo(state.polygon[0].x, state.polygon[0].y);
  for (const point of state.polygon.slice(1)) maskCtx.lineTo(point.x, point.y);
  maskCtx.closePath();
  maskCtx.fill();
  maskCtx.restore();
  state.polygon = [];
  markDirty();
  render();
}

async function saveMask(explicit = true) {
  clearTimeout(state.autosaveTimer);
  if (!state.dirty) {
    if (explicit) setStatus("No changes to save", "saved");
    return true;
  }
  const annotator = $("annotator").value.trim();
  if (!annotator) {
    setStatus("Enter an annotator name to save", "error");
    if (explicit) $("annotator").focus();
    return false;
  }
  localStorage.setItem("digit-annotator", annotator);
  setStatus("Saving…", "saving");
  const blob = await new Promise((resolve) => maskCanvas.toBlob(resolve, "image/png"));
  const sample = state.samples[state.index];
  try {
    const response = await fetch(
      `${sample.image_url.replace(/\/image$/, "/mask")}?annotator=${encodeURIComponent(annotator)}`,
      {method: "PUT", headers: {"Content-Type": "image/png"}, body: blob},
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Save failed");
    sample.saved = true;
    sample.mask_url = sample.image_url.replace(/\/image$/, "/mask");
    sample.annotator = payload.annotator;
    sample.annotation_revision = payload.annotation_revision;
    state.dirty = false;
    setStatus(`Saved revision ${payload.annotation_revision}`, "saved");
    await refreshStatus();
    updateSampleHeader();
    return true;
  } catch (error) {
    setStatus(error.message, "error");
    return false;
  }
}

async function refreshStatus() {
  const response = await fetch("/api/status", {cache: "no-store"});
  const status = await response.json();
  $("queue-status").textContent = `${status.completed}/${status.total} labeled`;
}

function updateSampleHeader() {
  const sample = state.samples[state.index];
  $("sample-id").textContent = sample.sample_id;
  const saved = sample.saved ? `saved r${sample.annotation_revision}` : "draft";
  $("sample-meta").textContent = `${state.index + 1} / ${state.samples.length} · ${sample.session_id} · ${saved}`;
  $("previous").disabled = state.index === 0;
  $("next").disabled = state.index === state.samples.length - 1;
}

async function loadSample(index) {
  state.index = index;
  const sample = state.samples[index];
  setStatus("Loading image…");
  const image = await loadImage(sample.image_url);
  setCanvasSize(image.naturalWidth, image.naturalHeight);
  sourceCtx.drawImage(image, 0, 0);
  maskCtx.fillStyle = "#000";
  maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
  if (sample.mask_url) {
    const mask = await loadImage(sample.mask_url);
    maskCtx.drawImage(mask, 0, 0);
  }
  state.backgroundAvailable = false;
  $("difference").disabled = !sample.background_url;
  if (sample.background_url) {
    try {
      const background = await loadImage(sample.background_url);
      makeDifference(background);
      state.backgroundAvailable = true;
      $("difference").disabled = false;
    } catch (error) {
      console.warn(error);
    }
  }
  state.undo = [];
  state.redo = [];
  state.polygon = [];
  state.dirty = false;
  updateHistoryButtons();
  updateSampleHeader();
  setStatus(sample.saved ? `Saved revision ${sample.annotation_revision}` : "Not labeled");
  render();
}

async function navigate(offset) {
  if (state.dirty && !(await saveMask(true))) return;
  const index = Math.max(0, Math.min(state.samples.length - 1, state.index + offset));
  if (index !== state.index) await loadSample(index);
}

function selectTool(tool) {
  if (state.polygon.length && tool !== "polygon") state.polygon = [];
  state.tool = tool;
  document.querySelectorAll("#tools button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tool === tool);
  });
  render();
}

view.addEventListener("pointerdown", (event) => {
  const point = pointFromEvent(event);
  if (state.tool === "polygon") {
    if (event.detail >= 2) finishPolygon();
    else {
      state.polygon.push(point);
      render();
    }
    return;
  }
  pushUndo();
  state.drawing = true;
  state.lastPoint = point;
  view.setPointerCapture(event.pointerId);
  stroke(point, point);
});

view.addEventListener("pointermove", (event) => {
  if (!state.drawing) return;
  const point = pointFromEvent(event);
  stroke(state.lastPoint, point);
  state.lastPoint = point;
});

function finishStroke(event) {
  if (!state.drawing) return;
  state.drawing = false;
  if (view.hasPointerCapture(event.pointerId)) view.releasePointerCapture(event.pointerId);
  markDirty();
}
view.addEventListener("pointerup", finishStroke);
view.addEventListener("pointercancel", finishStroke);

$("undo").addEventListener("click", () => {
  if (!state.undo.length) return;
  state.redo.push(snapshot());
  maskCtx.putImageData(state.undo.pop(), 0, 0);
  updateHistoryButtons();
  markDirty();
  render();
});

$("redo").addEventListener("click", () => {
  if (!state.redo.length) return;
  state.undo.push(snapshot());
  maskCtx.putImageData(state.redo.pop(), 0, 0);
  updateHistoryButtons();
  markDirty();
  render();
});

document.querySelectorAll("#tools button").forEach((button) => {
  button.addEventListener("click", () => selectTool(button.dataset.tool));
});
$("brush-size").addEventListener("input", () => {
  $("brush-size-value").value = `${$("brush-size").value} px`;
});
$("opacity").addEventListener("input", () => {
  $("opacity-value").value = `${$("opacity").value}%`;
  render();
});
$("zoom").addEventListener("input", applyZoom);
$("difference").addEventListener("change", render);
$("save").addEventListener("click", () => saveMask(true));
$("previous").addEventListener("click", () => navigate(-1));
$("next").addEventListener("click", () => navigate(1));

document.addEventListener("keydown", (event) => {
  const editingText = event.target instanceof HTMLInputElement && event.target.type !== "range";
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    (event.shiftKey ? $("redo") : $("undo")).click();
  } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
    event.preventDefault();
    $("redo").click();
  } else if (!editingText && event.key === "ArrowLeft") navigate(-1);
  else if (!editingText && event.key === "ArrowRight") navigate(1);
  else if (!editingText && event.key.toLowerCase() === "b") selectTool("brush");
  else if (!editingText && event.key.toLowerCase() === "e") selectTool("eraser");
  else if (!editingText && event.key.toLowerCase() === "p") selectTool("polygon");
  else if (!editingText && event.key === "Enter" && state.tool === "polygon") finishPolygon();
  else if (!editingText && event.key === "Escape" && state.polygon.length) {
    state.polygon = [];
    render();
  }
});

window.addEventListener("beforeunload", (event) => {
  if (state.dirty) {
    event.preventDefault();
    event.returnValue = "";
  }
});

async function start() {
  $("annotator").value = localStorage.getItem("digit-annotator") || "";
  const response = await fetch("/api/samples", {cache: "no-store"});
  const payload = await response.json();
  state.samples = payload.samples;
  if (!state.samples.length) throw new Error("Annotation queue is empty");
  $("queue-status").textContent = `${payload.status.completed}/${payload.status.total} labeled`;
  const firstIncomplete = state.samples.findIndex((sample) => !sample.saved);
  await loadSample(firstIncomplete >= 0 ? firstIncomplete : 0);
}

start().catch((error) => setStatus(error.message, "error"));
