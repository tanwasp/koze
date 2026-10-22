// Koze: Kreol speech to text in the browser with onnxruntime-web.
// models.json (served from this site) lists the model sizes with a SHA-256 for every
// part. Parts are downloaded once, checked against those hashes, kept in the Cache API,
// and run with WebGPU when available, otherwise WebAssembly.

const ORT_DIR = new URL("vendor/ort/", location.href).href;
const SR = 16000;
const MAX_CHUNK_S = 30;
const HISTORY_KEY = "koze-history";
const MODEL_KEY = "koze-model";
const SPEED_KEY = "koze-speed";
// Recommend a bigger model only if it would transcribe a minute of audio in under ~15 s.
const MAX_RTF = 0.25;

const $ = (id) => document.getElementById(id);
const ui = {
  pill: $("pill"), status: $("status"), stage: $("stage"), setup: $("setup"), setupTitle: $("setupTitle"),
  setupText: $("setupText"), models: $("models"), device: $("device"), cancelPick: $("cancelPick"),
  download: $("download"), downloadLabel: $("downloadLabel"), bar: $("bar"), barLabel: $("barLabel"),
  mic: $("mic"), micLabel: $("micLabel"), timer: $("timer"), wave: $("wave"), pick: $("pick"), file: $("file"),
  modelBtn: $("modelBtn"), modelName: $("modelName"), modelSpeed: $("modelSpeed"),
  out: $("out"), meta: $("meta"), copy: $("copy"), copyLabel: $("copyLabel"), history: $("history"),
};

let ort, session, vocab, backend;
let models = [], current = null, device = null, cached = {}, downloading = false;

// ------------------------------------------------------------------ model

async function init() {
  try {
    models = (await (await fetch("models.json", { cache: "no-cache" })).json()).models;
    if (!models.length) throw new Error("empty");
  } catch {
    return setStatus("No model published", "err");
  }
  [device] = await Promise.all([detectDevice(), refreshCached()]);
  const saved = models.find((m) => m.id === readJSON(MODEL_KEY));
  const start = saved && cached[saved.id] ? saved : models.find((m) => cached[m.id]);
  if (start) return loadModel(start);
  showPicker();
}

async function refreshCached() {
  await Promise.all(models.map(async (m) => (cached[m.id] = await isCached(m))));
}

function cacheName(m) { return `koze-model-${m.sha256.slice(0, 12)}`; }
function partUrl(m, path) { return new URL(m.base + path, location.href).href; }

async function isCached(m) {
  try {
    if (!(await caches.has(cacheName(m)))) return false;
    const cache = await caches.open(cacheName(m));
    const hits = await Promise.all(m.parts.map((p) => cache.match(partUrl(m, p.path))));
    return hits.every(Boolean);
  } catch {
    return false;
  }
}

async function sha256Hex(bytes) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

// Parts already in the cache were checked when they were stored. A part fresh from the
// network must match the hash in models.json before it is used or cached.
async function fetchModelBytes(m) {
  let cache = null;
  try { cache = await caches.open(cacheName(m)); } catch { /* private mode: download without caching */ }
  const buf = new Uint8Array(m.bytes);
  let offset = 0;
  showBar(0, m.bytes);
  for (const part of m.parts) {
    const url = partUrl(m, part.path);
    const start = offset;
    let res = cache && (await cache.match(url));
    const fresh = !res;
    if (fresh) {
      res = await fetch(url, { credentials: "omit", referrerPolicy: "no-referrer" });
      if (!res.ok) throw new Error(`Download failed for ${part.path} (${res.status})`);
    }
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (offset + value.length > start + part.bytes) throw new Error(`${part.path} is larger than expected`);
      buf.set(value, offset);
      offset += value.length;
      showBar(offset / m.bytes, m.bytes);
    }
    const bytes = buf.subarray(start, offset);
    if (fresh) {
      if (bytes.length !== part.bytes || (await sha256Hex(bytes)) !== part.sha256) {
        throw new Error(`${part.path} failed its integrity check, so it was not used`);
      }
      if (cache) {
        try { await cache.put(url, new Response(bytes.slice())); } catch { cache = null; }
      }
    }
  }
  if (offset !== m.bytes) throw new Error("Model download was incomplete. Reload to retry.");
  return buf;
}

async function fetchVocab(m) {
  const bytes = new Uint8Array(await (await fetch(partUrl(m, m.vocab.path), { credentials: "omit" })).arrayBuffer());
  if ((await sha256Hex(bytes)) !== m.vocab.sha256) throw new Error("The vocabulary file failed its integrity check");
  const v = JSON.parse(new TextDecoder().decode(bytes));
  return { pieces: v.pieces, control: new Set(v.control) };
}

async function loadModel(m) {
  ui.mic.disabled = ui.pick.disabled = ui.modelBtn.disabled = true;
  ui.setup.hidden = !!(cached[m.id] && !session);
  try {
    downloading = !cached[m.id];
    setStatus(downloading ? "Downloading…" : "Loading model…", "busy");
    if (!ort) {
      ort = await import(`${ORT_DIR}ort.webgpu.min.mjs`);
      ort.env.wasm.wasmPaths = ORT_DIR;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    }
    const [bytes, v] = await Promise.all([fetchModelBytes(m), fetchVocab(m)]);
    cached[m.id] = await isCached(m);
    setStatus("Preparing model…", "busy");
    if (!ui.setup.hidden) ui.barLabel.textContent = "Preparing the model…";
    const old = session;
    session = null;
    await old?.release?.();
    session = await createSession(bytes);
    vocab = v;
    current = m;
    writeJSON(MODEL_KEY, m.id);
    ui.setup.hidden = true;
    ui.bar.hidden = true;
    $("card").textContent = `(${m.card.replace(/_/g, " ")})`;
    setStatus(`Ready · ${m.params} · ${backend === "webgpu" ? "GPU" : "CPU"}`, "ok");
    ui.pill.title = backend === "webgpu" ? "Running on your GPU (WebGPU)" : "Running on your CPU (WebAssembly)";
    showModelChip();
    setStatus("Checking speed…", "busy");
    await speedCheck(m);
    setStatus(`Ready · ${m.params} · ${backend === "webgpu" ? "GPU" : "CPU"}`, "ok");
  } catch (e) {
    setStatus("Couldn't load model", "err");
    showPicker(`${e.message || e}. Try again, or pick another size.`);
    console.error(e);
  } finally {
    ui.mic.disabled = ui.pick.disabled = ui.modelBtn.disabled = !session;
  }
}

async function createSession(bytes) {
  // onnxruntime silently drops WebGPU when there is no adapter, so check first.
  const tries = device?.webgpu ? [["webgpu", "wasm"], ["wasm"]] : [["wasm"]];
  let last;
  for (const eps of tries) {
    try {
      const s = await ort.InferenceSession.create(bytes, { executionProviders: eps, graphOptimizationLevel: "all" });
      backend = eps[0];
      return s;
    } catch (e) {
      last = e;
      console.warn(`Backend ${eps[0]} failed, trying the next one`, e);
    }
  }
  throw last;
}

// ------------------------------------------------------------------ choosing a size

// Browsers deliberately reveal little: deviceMemory is rounded and capped at 8 GB (and
// missing in Safari and Firefox), and WebGPU names the GPU family but not its memory.
// So the hardware only rules things out; the speed check after loading decides.
async function detectDevice() {
  const ua = navigator.userAgent;
  const d = {
    webgpu: false, gpu: "",
    memGB: navigator.deviceMemory ?? null,
    threads: navigator.hardwareConcurrency || null,
    mobile: navigator.userAgentData?.mobile ?? (/Android|iPhone|iPad|Mobile/i.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1)),
  };
  try {
    const a = await navigator.gpu?.requestAdapter();
    const info = a && (a.info || (await a.requestAdapterInfo?.()));
    if (a && !(a.isFallbackAdapter || info?.isFallbackAdapter)) {
      d.webgpu = true;
      d.gpu = [info?.vendor, info?.architecture].filter(Boolean).join(" ");
    }
  } catch { /* no WebGPU */ }
  return d;
}

function speedFor(m) {
  // Seconds of compute per second of audio, measured for whichever size ran here last,
  // scaled by file size for the others (compute grows roughly with parameter count).
  const s = readJSON(SPEED_KEY);
  if (!s || s.backend !== (device.webgpu ? "webgpu" : "wasm")) return null;
  const ran = models.find((x) => x.id === s.id);
  if (!ran) return null;
  return { rtf: s.rtf * (m.bytes / ran.bytes), measured: ran.id === m.id };
}

function recommend() {
  const small = models[0];
  if (models.length === 1) return { id: small.id, why: "" };
  const big = models[models.length - 1];
  const pick = (m, why) => ({ id: m.id, why });
  if (device.mobile) return pick(small, "Phones and tablets tend to run out of memory with the larger model.");
  if (!device.webgpu) return pick(small, "This browser has no WebGPU, so the model would run on the CPU and the larger one would be slow.");
  if (device.memGB != null && device.memGB < 8) return pick(small, `This device reports ${device.memGB} GB of memory, which is tight for the larger model.`);
  const sp = speedFor(big);
  if (sp) {
    const perMin = Math.max(1, Math.round(sp.rtf * 60));
    return sp.rtf <= MAX_RTF
      ? pick(big, `From the speed check, ${big.name} should take about ${perMin} s per minute of audio here.`)
      : pick(small, `From the speed check, ${big.name} would take about ${perMin} s per minute of audio here, so ${small.name} is the better fit.`);
  }
  return pick(big, `WebGPU works and this isn't a phone, so ${big.name} should run well. A quick speed check after loading confirms it.`);
}

function deviceLine() {
  const bits = [];
  bits.push(device.webgpu ? `<b>GPU</b> ${esc(device.gpu || "found")} (WebGPU)` : "<b>GPU</b> not available to the browser");
  bits.push(device.memGB != null ? `<b>Memory</b> ${device.memGB >= 8 ? "8 GB or more" : `about ${device.memGB} GB`} (browsers cap this at 8)`
    : "<b>Memory</b> not reported by this browser");
  if (device.threads) bits.push(`<b>CPU</b> ${device.threads} threads`);
  return bits.join(" · ");
}

function showPicker(error) {
  const rec = recommend();
  const selected = (current && current.id) || rec.id;
  ui.setupTitle.textContent = session ? "Change model" : "Choose a model";
  if (error) ui.setupText.replaceChildren(span(error, "err"));
  else ui.setupText.textContent = rec.why ? `Recommended for this device: ${models.find((m) => m.id === rec.id).name}. ${rec.why}`
    : "It downloads once and then stays on this device.";
  ui.models.replaceChildren(...models.map((m) => {
    const label = document.createElement("label");
    label.className = "model";
    const sp = speedFor(m);
    const speed = sp ? ` · ${sp.measured ? "" : "about "}${fmtSpeed(sp.rtf)}` : "";
    label.innerHTML = `<input type="radio" name="model" value="${esc(m.id)}">
      <span class="m-name">${esc(m.name)} · ${esc(m.params)}</span>
      <span class="m-size">${fmtBytes(m.bytes)}</span>
      <span class="m-blurb">${esc(m.blurb)}${speed}</span>`;
    const name = label.querySelector(".m-name");
    if (m.id === rec.id && models.length > 1) name.append(span("Recommended", "tag tag-rec"));
    if (cached[m.id]) {
      name.append(span("On this device", "tag tag-have"));
      if (m.id !== current?.id) {
        const rm = document.createElement("button");
        rm.className = "remove";
        rm.type = "button";
        rm.textContent = "Remove";
        rm.onclick = async (e) => {
          e.preventDefault();
          await caches.delete(cacheName(m)).catch(() => {});
          cached[m.id] = false;
          showPicker();
        };
        label.querySelector(".m-size").replaceWith(rm);
      }
    }
    const input = label.querySelector("input");
    input.checked = m.id === selected;
    input.onchange = updateDownload;
    return label;
  }));
  ui.device.innerHTML = deviceLine();
  ui.cancelPick.hidden = !session;
  ui.download.hidden = false;
  ui.bar.hidden = true;
  ui.setup.hidden = false;
  updateDownload();
}

function chosen() {
  const id = ui.models.querySelector("input:checked")?.value;
  return models.find((m) => m.id === id) || models[0];
}

function updateDownload() {
  const m = chosen();
  ui.downloadLabel.textContent = m.id === current?.id ? `Keep ${m.name}`
    : cached[m.id] ? `Use ${m.name}` : `Download ${m.name} · ${fmtBytes(m.bytes)}`;
}

ui.download.onclick = () => {
  const m = chosen();
  if (m.id === current?.id) return (ui.setup.hidden = true);
  ui.download.hidden = ui.cancelPick.hidden = true;
  loadModel(m);
};
ui.cancelPick.onclick = () => { ui.setup.hidden = true; };
ui.modelBtn.onclick = () => showPicker();

function showModelChip() {
  ui.modelBtn.hidden = false;
  ui.modelName.textContent = `${current.name} · ${current.params}`;
  const sp = speedFor(current);
  ui.modelSpeed.textContent = sp?.measured ? fmtSpeed(sp.rtf) : "speed not measured";
}

// Time the model on 6 s of quiet noise. On the GPU the first run compiles shaders, so
// time a second one unless the first was already slow. This also warms up the session.
async function speedCheck(m) {
  const n = 6 * SR;
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = (Math.random() - 0.5) * 0.02;
  let secs = 0;
  try {
    for (let i = 0; i < 2 && (i === 0 || secs <= 2); i++) {
      const t0 = performance.now();
      const r = await session.run({ waveform: new ort.Tensor("float32", x, [1, n]) });
      if (r.logits.location === "gpu-buffer") await r.logits.getData();
      r.logits.dispose?.();
      secs = (performance.now() - t0) / 1000;
    }
  } catch (e) {
    return console.warn("Speed check failed", e);
  }
  writeJSON(SPEED_KEY, { id: m.id, backend, rtf: secs / 6, at: new Date().toISOString() });
  showModelChip();
}

// ------------------------------------------------------------------ audio

async function decodeTo16k(arrayBuffer) {
  const ctx = new AudioContext();
  let audio;
  try {
    audio = await ctx.decodeAudioData(arrayBuffer);
  } finally {
    ctx.close();
  }
  const length = Math.ceil(audio.duration * SR);
  const off = new OfflineAudioContext(1, length, SR);
  const src = off.createBufferSource();
  src.buffer = audio;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0);
}

// Split at the quietest 50 ms frame between 60% and 100% of the chunk limit.
function splitOnPauses(wav, sr = SR, maxS = MAX_CHUNK_S) {
  const n = wav.length, maxLen = Math.floor(maxS * sr);
  if (n <= maxLen) return [[0, n]];
  const hop = Math.floor(0.05 * sr);
  const spans = [];
  let start = 0;
  while (n - start > maxLen) {
    const lo = start + Math.floor(0.6 * maxLen), hi = start + maxLen;
    let best = hi, bestE = Infinity;
    for (let f = lo; f + hop <= hi; f += hop) {
      let e = 0;
      for (let i = f; i < f + hop; i++) e += wav[i] * wav[i];
      if (e < bestE) { bestE = e; best = f; }
    }
    spans.push([start, best]);
    start = best;
  }
  spans.push([start, n]);
  return spans;
}

// Same decoding as the omnilingual-asr pipeline: argmax per frame, collapse
// repeats, drop control symbols, join pieces, "▁" becomes a space.
function ctcGreedy(data, frames, vocabSize, voc) {
  let out = "", prev = -1;
  for (let t = 0; t < frames; t++) {
    let best = 0, bestV = -Infinity;
    const base = t * vocabSize;
    for (let v = 0; v < vocabSize; v++) {
      const x = data[base + v];
      if (x > bestV) { bestV = x; best = v; }
    }
    if (best !== prev && !voc.control.has(best)) out += voc.pieces[best] ?? "";
    prev = best;
  }
  return out.replaceAll("▁", " ").trim();
}

async function transcribe(wav, label) {
  ui.mic.disabled = ui.pick.disabled = true;
  ui.out.textContent = "";
  ui.out.classList.add("working");
  ui.copy.disabled = true;
  const seconds = wav.length / SR;
  const spans = splitOnPauses(wav);
  const t0 = performance.now();
  const parts = [];
  try {
    for (const [i, [a, b]] of spans.entries()) {
      ui.meta.textContent = spans.length > 1 ? `Transcribing part ${i + 1} of ${spans.length}…` : "Transcribing…";
      const input = new ort.Tensor("float32", wav.slice(a, b), [1, b - a]);
      const res = await session.run({ waveform: input });
      const logits = res.logits;
      const [, frames, vocabSize] = logits.dims;
      const data = logits.location === "gpu-buffer" ? await logits.getData() : logits.data;
      parts.push(ctcGreedy(data, frames, vocabSize, vocab));
      ui.out.textContent = parts.join(" ");
      logits.dispose?.();
    }
    const secs = (performance.now() - t0) / 1000;
    const text = parts.join(" ").trim();
    ui.out.textContent = text || "(nothing recognised)";
    ui.meta.textContent = `${fmtTime(seconds)} of audio · transcribed in ${secs.toFixed(1)} s on ${backend === "webgpu" ? "GPU" : "CPU"}`;
    ui.copy.disabled = !text;
    if (text) saveHistory({ text, label, seconds: +seconds.toFixed(1), at: new Date().toISOString() });
  } catch (e) {
    ui.meta.innerHTML = "";
    ui.meta.append(span(`Transcription failed: ${e.message || e}`, "err"));
    console.error(e);
  } finally {
    ui.out.classList.remove("working");
    ui.mic.disabled = ui.pick.disabled = false;
  }
}

// ------------------------------------------------------------------ recording

let recorder, chunks, tick, startedAt;

ui.mic.onclick = async () => {
  if (recorder && recorder.state === "recording") return recorder.stop();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch (e) {
    ui.meta.replaceChildren(span(`Microphone blocked (${e.message}). Allow it in the address bar and try again.`, "err"));
    return;
  }
  chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    clearInterval(tick);
    wave.stop();
    ui.stage.classList.remove("recording");
    ui.pick.disabled = false;
    ui.micLabel.textContent = "Tap to talk";
    ui.mic.setAttribute("aria-label", "Start recording");
    const blob = new Blob(chunks, { type: recorder.mimeType });
    try {
      const wav = await decodeTo16k(await blob.arrayBuffer());
      if (wav.length < SR * 0.3) return (ui.meta.textContent = "That was too short. Hold on a little longer.");
      await transcribe(wav, "recording");
    } catch (e) {
      ui.meta.textContent = `Could not read the recording: ${e.message}`;
    }
  };
  recorder.start();
  wave.start(stream);
  startedAt = Date.now();
  ui.stage.classList.add("recording");
  ui.pick.disabled = true;
  ui.micLabel.textContent = "Listening… tap to stop";
  ui.mic.setAttribute("aria-label", "Stop recording");
  ui.timer.textContent = "0:00";
  tick = setInterval(() => (ui.timer.textContent = fmtTime((Date.now() - startedAt) / 1000)), 250);
};

ui.pick.onclick = () => ui.file.click();
ui.file.onchange = async () => {
  const f = ui.file.files[0];
  ui.file.value = "";
  if (!f) return;
  try {
    const wav = await decodeTo16k(await f.arrayBuffer());
    await transcribe(wav, f.name);
  } catch (e) {
    ui.meta.textContent = `Could not read ${f.name}: ${e.message}. Try converting it to MP3 or WAV.`;
  }
};

ui.copy.onclick = async () => {
  try { await navigator.clipboard.writeText(ui.out.textContent); ui.copyLabel.textContent = "Copied"; } catch { /* ignore */ }
  setTimeout(() => (ui.copyLabel.textContent = "Copy"), 1200);
};

// ------------------------------------------------------------------ bits

function setStatus(text, state = "busy") {
  ui.status.textContent = text;
  ui.pill.dataset.state = state;
}
function showBar(frac, total) {
  ui.bar.hidden = false;
  ui.bar.querySelector(".progress > div").style.width = `${(frac * 100).toFixed(1)}%`;
  const mb = (x) => Math.round(x / 2 ** 20);
  ui.barLabel.textContent = `${mb(frac * total)} of ${mb(total)} MB`;
  setStatus(downloading ? `Downloading ${Math.round(frac * 100)}%` : "Loading model…", "busy");
}
// rtf: seconds of compute per second of audio.
function fmtSpeed(rtf) {
  const x = 1 / rtf;
  return x >= 1.5 ? `${Math.round(x)}× real time` : `${(rtf * 60).toFixed(0)} s per minute of audio`;
}
function fmtBytes(b) { return b >= 1e9 ? `${(b / 2 ** 30).toFixed(1)} GB` : `${Math.round(b / 2 ** 20)} MB`; }
function esc(t) { return String(t).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`); }
function readJSON(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } }
function writeJSON(key, v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* storage off */ } }
function fmtTime(s) {
  s = Math.floor(s);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// Live level bars while recording; a quiet dotted line otherwise.
const wave = (() => {
  const canvas = ui.wave, ctx = canvas.getContext("2d");
  const BARS = 48;
  let ac, analyser, raf, levels = new Array(BARS).fill(0);
  function size() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function draw() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement);
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, css.getPropertyValue("--accent-a").trim());
    grad.addColorStop(1, css.getPropertyValue("--accent-b").trim());
    const step = w / BARS, bw = Math.max(2, step * 0.42);
    for (let i = 0; i < BARS; i++) {
      const bh = Math.max(3, levels[i] * h * 0.95);
      ctx.fillStyle = analyser ? grad : css.getPropertyValue("--fg-3").trim();
      const x = i * step + (step - bw) / 2, y = (h - bh) / 2;
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x, y, bw, bh, bw / 2) : ctx.rect(x, y, bw, bh);
      ctx.fill();
    }
  }
  function loop() {
    const buf = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(buf);
    let sum = 0;
    for (const v of buf) sum += ((v - 128) / 128) ** 2;
    const rms = Math.min(1, Math.sqrt(sum / buf.length) * 4);
    levels = [...levels.slice(1), rms];
    draw();
    raf = requestAnimationFrame(loop);
  }
  size();
  draw();
  window.addEventListener("resize", () => { size(); draw(); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", draw);
  return {
    start(stream) {
      try {
        ac = new AudioContext();
        analyser = ac.createAnalyser();
        analyser.fftSize = 1024;
        ac.createMediaStreamSource(stream).connect(analyser);
        loop();
      } catch { analyser = null; }
    },
    stop() {
      cancelAnimationFrame(raf);
      ac?.close();
      ac = analyser = null;
      levels = new Array(BARS).fill(0);
      draw();
    },
  };
})();
function span(text, cls) { const s = document.createElement("span"); s.className = cls; s.textContent = text; return s; }

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
}
function saveHistory(item) {
  const items = [item, ...loadHistory()].slice(0, 50);
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items)); } catch { /* storage off */ }
  renderHistory(items);
}
function renderHistory(items = loadHistory()) {
  ui.history.replaceChildren(...items.map((h) => {
    const li = document.createElement("li");
    li.textContent = h.text;
    const meta = document.createElement("small");
    const when = new Date(h.at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    meta.textContent = `${when} · ${h.label === "recording" ? "Recording" : h.label} · ${fmtTime(h.seconds)}`;
    li.append(meta);
    return li;
  }));
}

renderHistory();
init();
