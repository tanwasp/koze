// Optional data collection: after a transcription the person can correct the text, rate
// it, and share the clip. Nothing leaves the device until they press Share and have
// agreed to the consent text.
//
// collect.json (written by tools/build_site.py when the site is built with a Supabase
// project) holds the project URL and its public key. With the rules in
// supabase/schema.sql that key can only add clips and withdraw one with its token.
// Without collect.json this whole feature stays hidden.

const SR = 16000;
const MAX_SHARE_S = 300;
const CONSENT_KEY = "koze-consent";
const SHARED_KEY = "koze-shared";
// Bump when the consent text in index.html changes, so people see and agree to the new one.
const CONSENT_VERSION = "1";

const $ = (id) => document.getElementById(id);
const ui = {
  form: $("contrib"), audio: $("clipAudio"), fix: $("fix"), resetFix: $("resetFix"), larmoni: $("larmoni"),
  rating: $("rating"), issuesWrap: $("issuesWrap"), issues: $("issues"), languages: $("languages"),
  comment: $("comment"), uploadChecks: $("uploadChecks"), permission: $("permission"), ownVoice: $("ownVoice"),
  share: $("share"), status: $("shareStatus"), consentOpen: $("consentOpen"),
  dialog: $("consent"), cAge: $("cAge"), cAgree: $("cAgree"), cOpen: $("cOpen"), revoke: $("consentRevoke"),
  sAge: $("sAge"), sGender: $("sGender"), sRegion: $("sRegion"), sFirst: $("sFirst"),
  sharedWrap: $("sharedWrap"), shared: $("shared"),
};

let config = null;
let clip = null; // { wav, source, seconds, modelText, model, backend, url }
let sending = false;

export async function initCollect() {
  try {
    const res = await fetch("collect.json", { cache: "no-cache" });
    if (!res.ok) return;
    const c = await res.json();
    if (!/^https:\/\//.test(c.url) || !c.key) return;
    config = { url: c.url.replace(/\/+$/, ""), key: c.key };
  } catch {
    return;
  }
  renderShared();
}

// Called by app.js after each transcription.
export function offerClip({ wav, source, modelText, model, backend }) {
  if (!config) return;
  if (clip?.url) URL.revokeObjectURL(clip.url);
  const seconds = wav.length / SR;
  const blob = encodeWav(wav);
  clip = { wav, blob, source, seconds, modelText, model, backend, url: URL.createObjectURL(blob) };
  ui.form.reset();
  ui.audio.src = clip.url;
  ui.fix.value = modelText;
  ui.uploadChecks.hidden = source !== "upload";
  ui.issuesWrap.hidden = true;
  ui.form.hidden = false;
  ui.share.disabled = false;
  if (seconds > MAX_SHARE_S) {
    ui.share.disabled = true;
    setStatus(`Clips over ${MAX_SHARE_S / 60} minutes can't be shared. Trim it and upload the part you want to share.`, "err");
  } else {
    setStatus("");
  }
}

ui.resetFix.onclick = () => { if (clip) ui.fix.value = clip.modelText; };
ui.rating.onchange = () => { ui.issuesWrap.hidden = rating() === 3 || rating() == null; };

ui.form.onsubmit = async (e) => {
  e.preventDefault();
  if (!clip || sending) return;
  const text = ui.fix.value.trim();
  if (!text) return setStatus("Write what was said first. If nobody spoke, there's nothing to share.", "err");
  if (clip.source === "upload" && !ui.permission.checked) {
    return setStatus("Please confirm that everyone who speaks in this audio agreed to share it.", "err");
  }
  if (!currentConsent() && !(await askConsent())) return;
  await share(text);
};

ui.consentOpen.onclick = () => askConsent();

async function share(text) {
  const consent = currentConsent();
  if (!consent) return;
  sending = true;
  ui.share.disabled = true;
  setStatus("Sharing…");
  const id = crypto.randomUUID();
  const token = randomHex(32);
  const ownVoice = clip.source === "recording" || ui.ownVoice.checked;
  const row = {
    id,
    audio_path: `${id}.wav`,
    source: clip.source,
    duration_s: +clip.seconds.toFixed(2),
    model_id: clip.model.id,
    model_card: clip.model.card,
    backend: clip.backend,
    model_text: clip.modelText,
    text: ui.fix.value, // as typed: no trimming or spelling changes beyond the empty check
    spelling: ui.larmoni.checked ? "larmoni" : "free",
    languages: checked(ui.languages),
    rating: rating(),
    issues: rating() === 3 ? [] : checked(ui.issues),
    comment: ui.comment.value.trim() || null,
    speaker: ownVoice && Object.keys(consent.speaker).length ? consent.speaker : null,
    permission_confirmed: true,
    consent_version: consent.version,
    open_release: consent.open,
    delete_token_hash: await sha256Hex(new TextEncoder().encode(token)),
  };
  try {
    // Audio first: a row never points at missing audio, and orphaned audio is cleaned up later.
    await api(`/storage/v1/object/clips/${id}.wav`, { body: clip.blob, headers: { "content-type": "audio/wav", "x-upsert": "false" } });
    await api("/rest/v1/clips", { body: JSON.stringify(row), headers: { "content-type": "application/json", prefer: "return=minimal" } });
    saveShared({ id, token, text, seconds: row.duration_s, at: new Date().toISOString() });
    setStatus("Shared. Thank you! You can withdraw it below at any time.", "ok");
  } catch (e) {
    setStatus(`Couldn't share: ${e.message}. Try again in a moment.`, "err");
    ui.share.disabled = false;
    console.error(e);
  } finally {
    sending = false;
  }
}

async function api(path, { body, headers }) {
  const res = await fetch(config.url + path, {
    method: "POST",
    body,
    credentials: "omit",
    referrerPolicy: "no-referrer",
    headers: { apikey: config.key, authorization: `Bearer ${config.key}`, ...headers },
  });
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).message || ""; } catch { /* not JSON */ }
    throw new Error(`the server said ${res.status}${detail ? ` (${detail})` : ""}`);
  }
  return res;
}

// ------------------------------------------------------------------ consent

function currentConsent() {
  const c = readJSON(CONSENT_KEY);
  return c && c.version === CONSENT_VERSION ? c : null;
}

function askConsent() {
  const c = currentConsent();
  ui.cAge.checked = ui.cAgree.checked = !!c;
  ui.cOpen.checked = !!c?.open;
  const s = c?.speaker || {};
  ui.sAge.value = s.age || "";
  ui.sGender.value = s.gender || "";
  ui.sRegion.value = s.region || "";
  ui.sFirst.value = s.first_language || "";
  ui.revoke.hidden = !c;
  ui.dialog.returnValue = "";
  ui.dialog.showModal();
  return new Promise((resolve) => {
    ui.dialog.addEventListener("close", () => {
      const v = ui.dialog.returnValue;
      if (v === "revoke") {
        try { localStorage.removeItem(CONSENT_KEY); } catch { /* storage off */ }
        setStatus("Sharing is off. Shared clips stay until you withdraw them below.");
        return resolve(false);
      }
      if (v !== "ok") return resolve(false);
      const speaker = {};
      if (ui.sAge.value) speaker.age = ui.sAge.value;
      if (ui.sGender.value) speaker.gender = ui.sGender.value;
      if (ui.sRegion.value) speaker.region = ui.sRegion.value;
      if (ui.sFirst.value) speaker.first_language = ui.sFirst.value;
      writeJSON(CONSENT_KEY, { version: CONSENT_VERSION, at: new Date().toISOString(), open: ui.cOpen.checked, speaker });
      resolve(true);
    }, { once: true });
  });
}

// ------------------------------------------------------------------ shared clips

function loadShared() { return readJSON(SHARED_KEY) || []; }

function saveShared(item) {
  writeJSON(SHARED_KEY, [item, ...loadShared()].slice(0, 200));
  renderShared();
}

function renderShared() {
  const items = loadShared();
  ui.sharedWrap.hidden = !items.length;
  ui.shared.replaceChildren(...items.map((s) => {
    const li = document.createElement("li");
    li.textContent = s.text;
    const meta = document.createElement("small");
    const when = new Date(s.at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    meta.textContent = `${when} · ${Math.round(s.seconds)} s · `;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "link danger";
    btn.textContent = "Withdraw";
    btn.onclick = () => withdraw(s, btn);
    meta.append(btn);
    li.append(meta);
    return li;
  }));
}

async function withdraw(s, btn) {
  if (!config || !confirm("Withdraw this clip? Its audio and text will be deleted.")) return;
  btn.disabled = true;
  btn.textContent = "Withdrawing…";
  try {
    const res = await api("/rest/v1/rpc/withdraw_clip", {
      body: JSON.stringify({ clip_id: s.id, token: s.token }),
      headers: { "content-type": "application/json" },
    });
    // false means it was already gone (withdrawn before, or removed by the maintainer).
    await res.json();
    writeJSON(SHARED_KEY, loadShared().filter((x) => x.id !== s.id));
    renderShared();
  } catch (e) {
    btn.disabled = false;
    btn.textContent = "Withdraw (failed, try again)";
    console.error(e);
  }
}

// ------------------------------------------------------------------ bits

// 16 kHz mono 16-bit PCM: exactly what the model heard, and what training pipelines expect.
function encodeWav(wav) {
  const n = wav.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, SR, true); v.setUint32(28, SR * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const x = Math.max(-1, Math.min(1, wav[i]));
    v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

function rating() {
  const r = ui.rating.querySelector("input:checked");
  return r ? +r.value : null;
}
function checked(group) { return [...group.querySelectorAll("input:checked")].map((i) => i.value); }
function setStatus(text, state = "") { ui.status.textContent = text; ui.status.dataset.state = state; }
function randomHex(bytes) { return Array.from(crypto.getRandomValues(new Uint8Array(bytes)), (b) => b.toString(16).padStart(2, "0")).join(""); }
async function sha256Hex(bytes) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}
function readJSON(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } }
function writeJSON(key, v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* storage off */ } }
