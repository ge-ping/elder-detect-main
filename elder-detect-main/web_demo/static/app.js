"use strict";

const state = { profile: null, metric: "cadence_spm", previewUrl: null };

const metricMeta = {
  cadence_spm: { label: "步频", unit: "步/分", decimals: 1 },
  step_time_cv: { label: "步时变异", unit: "CV", decimals: 3 },
  trunk_sway_rms_deg: { label: "躯干摆动", unit: "° RMS", decimals: 2 },
  emos_proxy: { label: "eMOS 代理", unit: "proxy", decimals: 3 },
};

const byId = (id) => document.getElementById(id);
const numberText = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";

function setText(id, value) {
  const element = byId(id);
  if (element) element.textContent = value;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

function baselineForMetric(history, key) {
  if (key === "cadence_spm") {
    return history.map((item) => ({
      median: item.baseline_median, low: item.baseline_low, high: item.baseline_high,
    }));
  }
  const anchor = history.slice(0, 14).map((item) => Number(item[key])).filter(Number.isFinite);
  const sorted = [...anchor].sort((a, b) => a - b);
  const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
  const deviations = sorted.map((value) => Math.abs(value - median)).sort((a, b) => a - b);
  const mad = deviations.length ? deviations[Math.floor(deviations.length / 2)] : 0;
  const scale = Math.max(mad * 1.4826, Math.abs(median) * 0.035, 0.002);
  return history.map(() => ({ median, low: median - 2 * scale, high: median + 2 * scale }));
}

function svgLineChart(containerId, history, key, isDistance = false) {
  const container = byId(containerId);
  if (!container || !history.length) return;
  const width = 720;
  const height = 270;
  const margin = { top: 16, right: 18, bottom: 34, left: 48 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const baselines = baselineForMetric(history, key);
  const values = isDistance ? history.map((item) => Number(item.distance)) : history.map((item) => Number(item[key]));
  const candidates = isDistance
    ? [...values, 0, 3]
    : [...values, ...baselines.flatMap((item) => [item.low, item.high])];
  let minValue = Math.min(...candidates.filter(Number.isFinite));
  let maxValue = Math.max(...candidates.filter(Number.isFinite));
  const padding = Math.max((maxValue - minValue) * 0.15, Math.abs(maxValue) * 0.04, 0.02);
  minValue -= padding;
  maxValue += padding;
  if (isDistance) minValue = 0;
  const x = (index) => margin.left + (index / Math.max(history.length - 1, 1)) * plotWidth;
  const y = (value) => margin.top + (1 - (value - minValue) / Math.max(maxValue - minValue, 1e-9)) * plotHeight;
  const path = (series) => series.map((value, index) => `${index ? "L" : "M"}${x(index).toFixed(1)},${y(value).toFixed(1)}`).join(" ");
  const axisRows = Array.from({ length: 5 }, (_, index) => minValue + (maxValue - minValue) * index / 4);
  const xTicks = [0, Math.floor((history.length - 1) / 2), history.length - 1];
  const meta = isDistance ? { label: "偏离距离", unit: "", decimals: 2 } : metricMeta[key];

  let band = "";
  let median = "";
  if (!isDistance) {
    const upper = baselines.map((item, index) => `${x(index).toFixed(1)},${y(item.high).toFixed(1)}`);
    const lower = [...baselines].reverse().map((item, reverseIndex) => {
      const index = baselines.length - 1 - reverseIndex;
      return `${x(index).toFixed(1)},${y(item.low).toFixed(1)}`;
    });
    band = `<polygon class="baseline-band" points="${[...upper, ...lower].join(" ")}"></polygon>`;
    median = `<path class="median-line" d="${path(baselines.map((item) => item.median))}"></path>`;
  }
  const threshold = isDistance
    ? `<line class="threshold" x1="${margin.left}" x2="${width - margin.right}" y1="${y(3)}" y2="${y(3)}"></line><text x="${width - margin.right - 4}" y="${y(3) - 6}" text-anchor="end">提示阈值 3</text>`
    : "";
  const grid = axisRows.map((value) => `<g><line class="grid" x1="${margin.left}" x2="${width - margin.right}" y1="${y(value)}" y2="${y(value)}"></line><text x="${margin.left - 8}" y="${y(value) + 3}" text-anchor="end">${numberText(value, meta.decimals)}</text></g>`).join("");
  const labels = xTicks.map((index) => `<text x="${x(index)}" y="${height - 8}" text-anchor="middle">${history[index].date.slice(5)}</text>`).join("");
  const circles = values.map((value, index) => `<circle class="point" cx="${x(index)}" cy="${y(value)}" r="3.2" tabindex="0" aria-label="${history[index].date}，${meta.label}${numberText(value, meta.decimals)}${meta.unit}"><title>${history[index].date} · ${meta.label} ${numberText(value, meta.decimals)} ${meta.unit}</title></circle>`).join("");
  const lineClass = isDistance ? "distance-line" : "data-line";

  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${containerId}-title ${containerId}-desc"><title id="${containerId}-title">${meta.label}时间趋势</title><desc id="${containerId}-desc">共${history.length}个观测日，可聚焦数据点读取具体数值。</desc>${grid}${band}${median}${threshold}<line class="axis" x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}"></line><line class="axis" x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}"></line><path class="${lineClass}" d="${path(values)}"></path>${circles}${labels}</svg>`;
}

function renderTables(profile) {
  const latest = [...profile.history].slice(-8).reverse();
  byId("history-body").innerHTML = latest.map((item) => `<tr><td>${escapeHtml(item.date)}</td><td>${numberText(item.cadence_spm, 1)}</td><td>${numberText(item.step_time_cv, 3)}</td><td>${numberText(item.trunk_sway_rms_deg, 2)}°</td><td>${numberText(item.distance, 2)}</td><td>${escapeHtml(item.quality)}</td></tr>`).join("");
  const key = state.metric;
  const baselines = baselineForMetric(profile.history, key);
  const meta = metricMeta[key];
  byId("chart-data-body").innerHTML = profile.history.map((item, index) => `<tr><td>${escapeHtml(item.date)}</td><td>${numberText(item[key], meta.decimals)} ${meta.unit}</td><td>${numberText(baselines[index].low, meta.decimals)}–${numberText(baselines[index].high, meta.decimals)}</td><td>${numberText(item.distance, 2)}</td></tr>`).join("");
}

function renderProfile(profile) {
  state.profile = profile;
  setText("profile-name", profile.person.name);
  setText("profile-age", `${profile.person.age}岁`);
  setText("profile-context", profile.person.living_context || `编号 ${profile.person.id}`);
  setText("mode-badge", profile.mode_label);
  byId("mode-badge").className = `badge ${profile.data_mode === "sample" ? "badge-sample" : "badge-real"}`;
  setText("baseline-progress-text", `${profile.baseline.valid_days} / ${profile.baseline.required_days} 有效日`);
  byId("baseline-progress-bar").style.width = `${Math.min(profile.baseline.progress * 100, 100)}%`;
  setText("metric-cadence", numberText(profile.metrics.cadence_spm, 1));
  setText("metric-step-cv", numberText(profile.metrics.step_time_cv, 3));
  setText("metric-sway", numberText(profile.metrics.trunk_sway_rms_deg, 2));
  setText("metric-emos", numberText(profile.metrics.emos_proxy, 3));
  setText("metric-deviation", numberText(profile.metrics.deviation_index, 2));
  setText("distance-value", numberText(profile.metrics.distance, 2));
  setText("quality-pill", `数据质量${profile.quality.status}`);
  setText("psychological-label", profile.person.psychological.label);
  setText("psychological-role", profile.person.psychological.role);
  setText("baseline-method", profile.baseline.method);
  setText("baseline-anchor", `${profile.baseline.anchor_start} 至 ${profile.baseline.anchor_end}`);
  setText("risk-title", profile.risk.level);
  setText("risk-message", profile.risk.message);
  renderCharts();
  renderTables(profile);
}

function renderCharts() {
  if (!state.profile) return;
  const meta = metricMeta[state.metric];
  setText("chart-caption", `${meta.label}，单位：${meta.unit}`);
  byId("baseline-chart").setAttribute("aria-label", `每日${meta.label}与个人基线变化图`);
  svgLineChart("baseline-chart", state.profile.history, state.metric, false);
  svgLineChart("deviation-chart", state.profile.history, "distance", true);
  renderTables(state.profile);
}

async function loadProfile() {
  const response = await fetch("/api/profile", { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error("示例画像加载失败");
  renderProfile(await response.json());
}

function showSelectedVideo(file) {
  setText("selected-file", file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB` : "尚未选择文件");
  const preview = byId("video-preview");
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  if (!file) {
    preview.hidden = true;
    preview.removeAttribute("src");
    return;
  }
  state.previewUrl = URL.createObjectURL(file);
  preview.src = state.previewUrl;
  preview.hidden = false;
}

function setFormStatus(message, type = "") {
  const status = byId("form-status");
  status.textContent = message;
  status.className = `form-status${type ? ` is-${type}` : ""}`;
}

async function submitVideo(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const file = byId("video-input").files[0];
  if (!file) {
    setFormStatus("请先选择步行视频。", "error");
    byId("video-input").focus();
    return;
  }
  const button = byId("analyze-button");
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  setText("selected-file", `${file.name} · 正在处理`);
  setFormStatus("正在上传并检查姿态分析环境……");
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: new FormData(form) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "视频处理失败");
    renderProfile(payload.profile);
    const notice = payload.processing.warning || payload.processing.message;
    setFormStatus(notice, payload.processing.warning ? "error" : "success");
    setText("selected-file", `${payload.upload.filename} · 处理完成`);
  } catch (error) {
    setFormStatus(error.message || "视频处理失败，请重试。", "error");
  } finally {
    button.disabled = false;
    button.removeAttribute("aria-busy");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  byId("observed-at").value = new Date().toISOString().slice(0, 10);
  byId("video-input").addEventListener("change", (event) => showSelectedVideo(event.target.files[0]));
  byId("upload-form").addEventListener("submit", submitVideo);
  byId("metric-select").addEventListener("change", (event) => {
    state.metric = event.target.value;
    renderCharts();
  });
  const dropzone = byId("dropzone");
  ["dragenter", "dragover"].forEach((type) => dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach((type) => dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  }));
  dropzone.addEventListener("drop", (event) => {
    const files = event.dataTransfer.files;
    if (!files.length) return;
    const transfer = new DataTransfer();
    transfer.items.add(files[0]);
    byId("video-input").files = transfer.files;
    showSelectedVideo(files[0]);
  });
  loadProfile().catch((error) => setFormStatus(error.message, "error"));
});

window.addEventListener("beforeunload", () => {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
});
