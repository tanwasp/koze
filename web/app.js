// Koze: Kreol speech to text in the browser with onnxruntime-web.
// The model is downloaded once in parts, kept in the Cache API, and run with
// WebGPU when available, otherwise WebAssembly.

const ORT_VERSION = "1.27.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const SR = 16000;
const MAX_CHUNK_S = 30;
const HISTORY_KEY = "koze-history";

const $ = (id) => document.getElementById(id);
const ui = {
  status: $("status"), download: $("download"), bar: $("bar"), mic: $("mic"), timer: $("timer"),
  pick: $("pick"), file: $("file"), out: $("out"), meta: $("meta"), copy: $("copy"), history: $("history"),
};

let ort, session, vocab, manifest, backend;

// ------------------------------------------------------------------ model

async function init() {
  try {
    manifest = await (await fetch("model/manifest.json", { cache: "no-cache" })).json();
  } catch {
    return setStatus("No model is published with this site yet.", "err");
  }
  $("card").textContent = `(${manifest.card})`;
  const mb = Math.round(manifest.bytes / 2 ** 20);
  if (await isCached()) {
    await loadModel();
  } else {
    setStatus(`The model is ${mb} MB. It downloads once and then stays on this device.`);
    ui.download.textContent = `Download model (${mb} MB)`;
    ui.download.hidden = false;
    ui.download.onclick = () => { ui.download.hidden = true; loadModel(); };
  }
}

async function openCache() {
  return caches.open(`koze-model-${manifest.sha256.slice(0, 12)}`);
}

async function isCached() {
  try {
    const cache = await openCache();
    const hits = await Promise.all(manifest.parts.map((p) => cache.match(p)));
    return hits.every(Boolean);
  } catch {
    return false;
  }
}

async function fetchModelBytes() {
  let cache = null;
  try { cache = await openCache(); } catch { /* private mode: download without caching */ }
  const buf = new Uint8Array(manifest.bytes);
  let offset = 0;
  showBar(0);
  for (const part of manifest.parts) {
    let res = cache && (await cache.match(part));
    if (!res) {
      res = await fetch(part);
      if (!res.ok) throw new Error(`Download failed for ${part} (${res.status})`);
      if (cache) {
        try { await cache.put(part, res.clone()); } catch { cache = null; }
      }
    }
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf.set(value, offset);
      offset += value.length;
      showBar(offset / manifest.bytes);
      setStatus(`Loading model… ${Math.round(offset / 2 ** 20)} / ${Math.round(manifest.bytes / 2 ** 20)} MB`);
    }
  }
  if (offset !== manifest.bytes) throw new Error("Model download was incomplete. Reload to retry.");
  return buf;
}

async function loadModel() {
  try {
    ort = await import(`${ORT_BASE}ort.webgpu.min.mjs`);
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    const [bytes, v] = await Promise.all([fetchModelBytes(), fetch(manifest.vocab).then((r) => r.json())]);
    vocab = { pieces: v.pieces, control: new Set(v.control) };
    setStatus("Preparing the model…");
    session = await createSession(bytes);
    hideBar();
    setStatus(`Ready. Running on ${backend === "webgpu" ? "your GPU (WebGPU)" : "your CPU (WebAssembly)"}.`, "ok");
    ui.mic.disabled = ui.pick.disabled = false;
  } catch (e) {
    hideBar();
    setStatus(`Could not load the model: ${e.message || e}`, "err");
    console.error(e);
  }
}

async function createSession(bytes) {
  // onnxruntime silently drops WebGPU when there is no adapter, so check first.
  let gpu = false;
  try { gpu = !!(navigator.gpu && (await navigator.gpu.requestAdapter())); } catch { /* no WebGPU */ }
  const tries = gpu ? [["webgpu", "wasm"], ["wasm"]] : [["wasm"]];
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
  ui.copy.disabled = true;
  const seconds = wav.length / SR;
  const spans = splitOnPauses(wav);
  const t0 = performance.now();
  const parts = [];
  try {
    for (const [i, [a, b]] of spans.entries()) {
      ui.meta.textContent = `Transcribing ${spans.length > 1 ? `part ${i + 1} of ${spans.length}` : ""}…`;
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
    ui.meta.textContent = `${seconds.toFixed(1)} s of audio in ${secs.toFixed(1)} s on ${backend}.`;
    ui.copy.disabled = !text;
    if (text) saveHistory({ text, label, seconds: +seconds.toFixed(1), at: new Date().toISOString() });
  } catch (e) {
    ui.meta.innerHTML = "";
    ui.meta.append(span(`Transcription failed: ${e.message || e}`, "err"));
    console.error(e);
  } finally {
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
    return setStatus(`Microphone blocked: ${e.message}. Allow it in the address bar and try again.`, "err");
  }
  chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    clearInterval(tick);
    ui.mic.classList.remove("on");
    ui.mic.textContent = "Tap to talk";
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
  startedAt = Date.now();
  ui.mic.classList.add("on");
  ui.mic.textContent = "Tap to stop";
  ui.timer.textContent = "0:00";
  tick = setInterval(() => {
    const s = Math.floor((Date.now() - startedAt) / 1000);
    ui.timer.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }, 250);
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
  try { await navigator.clipboard.writeText(ui.out.textContent); ui.copy.textContent = "Copied"; } catch { /* ignore */ }
  setTimeout(() => (ui.copy.textContent = "Copy"), 1200);
};

// ------------------------------------------------------------------ bits

function setStatus(text, cls = "") {
  ui.status.textContent = text;
  ui.status.className = cls;
}
function showBar(frac) {
  ui.bar.style.display = "block";
  ui.bar.firstElementChild.style.width = `${Math.round(frac * 100)}%`;
}
function hideBar() { ui.bar.style.display = "none"; }
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
    meta.textContent = `${new Date(h.at).toLocaleString()} · ${h.label} · ${h.seconds} s`;
    li.append(meta);
    return li;
  }));
}

renderHistory();
init();
