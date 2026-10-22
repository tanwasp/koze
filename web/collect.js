// Contributor accounts and clip sharing. Anyone can transcribe; after a transcription a
// signed-in contributor can correct the text, rate it and share the clip. Nothing leaves
// the device until they press Share, and only after they've agreed to the consent text.
//
// collect.json (written by tools/build_site.py when the site is built with a Supabase
// project) holds the project URL and its public key. Sign-in is a passwordless email code
// against Supabase Auth; supabase/schema.sql scopes every request to the signed-in person.
// Without collect.json this whole feature stays hidden.

const SR = 16000;
const MAX_SHARE_S = 300;
const SESSION_KEY = "koze-session";
// Bump when the consent text in index.html changes, so everyone sees and agrees to the new one.
const CONSENT_VERSION = "1";
const RESEND_S = 60;

const $ = (id) => document.getElementById(id);
const ui = {
  form: $("contrib"), audio: $("clipAudio"), fix: $("fix"), resetFix: $("resetFix"), larmoni: $("larmoni"),
  rating: $("rating"), issuesWrap: $("issuesWrap"), issues: $("issues"), languages: $("languages"),
  comment: $("comment"), uploadChecks: $("uploadChecks"), permission: $("permission"), ownVoice: $("ownVoice"),
  share: $("share"), status: $("shareStatus"), invite: $("invite"), inviteBtn: $("inviteBtn"),
  signInBtn: $("signInBtn"), accountBtn: $("accountBtn"),
  signin: $("signin"), emailForm: $("emailForm"), email: $("email"), sendCode: $("sendCode"), codeForm: $("codeForm"),
  code: $("code"), codeEmail: $("codeEmail"), resend: $("resend"), otherEmail: $("otherEmail"), signinStatus: $("signinStatus"),
  account: $("account"), roleTag: $("roleTag"), acctEmail: $("acctEmail"), stats: $("stats"), nameForm: $("nameForm"),
  displayName: $("displayName"), myClips: $("myClips"), editConsent: $("editConsent"), signOut: $("signOut"),
  deleteAccount: $("deleteAccount"), acctStatus: $("acctStatus"),
  dialog: $("consent"), cAge: $("cAge"), cAgree: $("cAgree"), cOpen: $("cOpen"), revoke: $("consentRevoke"),
  sAge: $("sAge"), sGender: $("sGender"), sRegion: $("sRegion"), sFirst: $("sFirst"),
};

let config = null;
let session = null; // { access_token, refresh_token, expires_at, user: { id, email } }
let profile = null;
let clip = null; // { wav, blob, source, seconds, modelText, model, backend, url }
let sending = false;
let resendTimer;

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
  session = readJSON(SESSION_KEY);
  if (session) await loadProfile().catch(() => {});
  renderAuth();
}

// Called by app.js after each transcription.
export function offerClip({ wav, source, modelText, model, backend }) {
  if (!config) return;
  if (clip?.url) URL.revokeObjectURL(clip.url);
  const blob = encodeWav(wav);
  clip = { blob, source, seconds: wav.length / SR, modelText, model, backend, url: URL.createObjectURL(blob) };
  ui.form.reset();
  ui.audio.src = clip.url;
  ui.fix.value = modelText;
  ui.uploadChecks.hidden = source !== "upload";
  ui.issuesWrap.hidden = true;
  ui.share.disabled = false;
  if (clip.seconds > MAX_SHARE_S) {
    ui.share.disabled = true;
    setStatus(ui.status, `Clips over ${MAX_SHARE_S / 60} minutes can't be shared. Trim it and upload the part you want to share.`, "err");
  } else {
    setStatus(ui.status, "");
  }
  renderAuth();
}

function renderAuth() {
  const signedIn = !!session;
  ui.signInBtn.hidden = signedIn;
  ui.accountBtn.hidden = !signedIn;
  if (signedIn) ui.accountBtn.textContent = (profile?.display_name || session.user.email || "?").trim()[0].toUpperCase();
  ui.form.hidden = !signedIn || !clip;
  ui.invite.hidden = signedIn || !clip;
}

// ------------------------------------------------------------------ sharing

ui.resetFix.onclick = () => { if (clip) ui.fix.value = clip.modelText; };
ui.rating.onchange = () => { ui.issuesWrap.hidden = rating() === 3 || rating() == null; };

ui.form.onsubmit = async (e) => {
  e.preventDefault();
  if (!clip || sending) return;
  if (!ui.fix.value.trim()) return setStatus(ui.status, "Write what was said first. If nobody spoke, there's nothing to share.", "err");
  if (clip.source === "upload" && !ui.permission.checked) {
    return setStatus(ui.status, "Please confirm that everyone who speaks in this audio agreed to share it.", "err");
  }
  if (!session && !(await signIn())) return;
  if (profile?.consent_version !== CONSENT_VERSION && !(await askConsent())) return;
  await share();
};

async function share() {
  sending = true;
  ui.share.disabled = true;
  setStatus(ui.status, "Sharing…");
  const id = crypto.randomUUID();
  const path = `${session.user.id}/${id}.wav`;
  const row = {
    id,
    audio_path: path,
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
    own_voice: clip.source === "recording" || ui.ownVoice.checked,
    permission_confirmed: true,
  };
  try {
    // Audio first, so a row never points at missing audio. Orphaned audio is cleaned up later.
    await api(`/storage/v1/object/clips/${path}`, { body: clip.blob, headers: { "content-type": "audio/wav", "x-upsert": "false" } });
    await api("/rest/v1/clips", { body: JSON.stringify(row), headers: { "content-type": "application/json", prefer: "return=minimal" } });
    setStatus(ui.status, "Shared. Thank you! You'll find it in your account.", "ok");
  } catch (e) {
    setStatus(ui.status, `Couldn't share: ${e.message}.`, "err");
    ui.share.disabled = false;
    console.error(e);
  } finally {
    sending = false;
  }
}

// ------------------------------------------------------------------ Supabase calls

async function api(path, { method = "POST", body, headers = {}, auth = true } = {}) {
  const h = { apikey: config.key, ...headers };
  if (auth) {
    const token = await accessToken();
    if (!token) throw new Error("you're signed out. Sign in and try again");
    h.authorization = `Bearer ${token}`;
  }
  const res = await fetch(config.url + path, { method, body, headers: h, credentials: "omit", referrerPolicy: "no-referrer" });
  if (res.status === 401 && auth) {
    endSession();
    throw new Error("your sign-in expired. Sign in again");
  }
  if (!res.ok) {
    let detail = "";
    try { const j = await res.json(); detail = j.msg || j.message || j.error_description || ""; } catch { /* not JSON */ }
    throw new Error(detail || `the server said ${res.status}`);
  }
  return res;
}

async function json(path, opts) {
  const res = await api(path, opts);
  return res.status === 204 ? null : res.json();
}

// Access tokens last an hour; refresh a minute early. Another tab may have refreshed
// already (refresh tokens are single use), so re-read storage first.
async function accessToken() {
  if (!session) return null;
  const stored = readJSON(SESSION_KEY);
  if (stored?.refresh_token && stored.refresh_token !== session.refresh_token) session = stored;
  if (session.expires_at - 60 > Date.now() / 1000) return session.access_token;
  try {
    const s = await json("/auth/v1/token?grant_type=refresh_token", {
      body: JSON.stringify({ refresh_token: session.refresh_token }),
      headers: { "content-type": "application/json" }, auth: false,
    });
    startSession(s);
    return session.access_token;
  } catch {
    endSession();
    return null;
  }
}

function startSession(s) {
  session = {
    access_token: s.access_token, refresh_token: s.refresh_token,
    expires_at: s.expires_at || Math.floor(Date.now() / 1000) + s.expires_in,
    user: { id: s.user.id, email: s.user.email },
  };
  writeJSON(SESSION_KEY, session);
}

function endSession() {
  session = profile = null;
  try { localStorage.removeItem(SESSION_KEY); } catch { /* storage off */ }
  renderAuth();
}

async function loadProfile() {
  const rows = await json(`/rest/v1/profiles?select=*&id=eq.${session.user.id}`, { method: "GET" });
  profile = rows[0] || null;
}

// ------------------------------------------------------------------ sign in

ui.signInBtn.onclick = ui.inviteBtn.onclick = () => signIn();
for (const b of document.querySelectorAll("[data-close]")) b.onclick = () => b.closest("dialog").close();

// Resolves true once signed in (and agreed to the consent terms), false if they gave up.
function signIn() {
  ui.emailForm.hidden = false;
  ui.codeForm.hidden = true;
  setStatus(ui.signinStatus, "");
  ui.signin.showModal();
  return new Promise((resolve) => ui.signin.addEventListener("close", () => resolve(!!session), { once: true }));
}

ui.emailForm.onsubmit = async (e) => {
  e.preventDefault();
  await sendCode(ui.email.value.trim());
};
ui.resend.onclick = () => sendCode(ui.codeEmail.textContent);
ui.otherEmail.onclick = () => {
  ui.codeForm.hidden = true;
  ui.emailForm.hidden = false;
  setStatus(ui.signinStatus, "");
};

async function sendCode(email) {
  ui.sendCode.disabled = ui.resend.disabled = true;
  setStatus(ui.signinStatus, "Sending…");
  try {
    await api("/auth/v1/otp", {
      body: JSON.stringify({ email, create_user: true }), headers: { "content-type": "application/json" }, auth: false,
    });
    ui.codeEmail.textContent = email;
    ui.emailForm.hidden = true;
    ui.codeForm.hidden = false;
    ui.code.value = "";
    ui.code.focus();
    setStatus(ui.signinStatus, "Check your inbox (and spam folder). The code works for an hour.");
    cooldown();
  } catch (e) {
    setStatus(ui.signinStatus, `Couldn't send a code: ${e.message}.`, "err");
    ui.resend.disabled = false;
  } finally {
    ui.sendCode.disabled = false;
  }
}

function cooldown() {
  let left = RESEND_S;
  clearInterval(resendTimer);
  ui.resend.disabled = true;
  ui.resend.textContent = `Send a new code (${left})`;
  resendTimer = setInterval(() => {
    left -= 1;
    ui.resend.textContent = left > 0 ? `Send a new code (${left})` : "Send a new code";
    if (left <= 0) { clearInterval(resendTimer); ui.resend.disabled = false; }
  }, 1000);
}

ui.codeForm.onsubmit = async (e) => {
  e.preventDefault();
  setStatus(ui.signinStatus, "Checking…");
  try {
    const s = await json("/auth/v1/verify", {
      body: JSON.stringify({ type: "email", email: ui.codeEmail.textContent, token: ui.code.value.trim() }),
      headers: { "content-type": "application/json" }, auth: false,
    });
    startSession(s);
    await loadProfile();
    renderAuth();
    ui.signin.close();
  } catch (e) {
    setStatus(ui.signinStatus, `That didn't work: ${e.message}.`, "err");
  }
};

// ------------------------------------------------------------------ consent

ui.editConsent.onclick = () => askConsent();

function askConsent() {
  const agreed = profile?.consent_version === CONSENT_VERSION;
  ui.cAge.checked = ui.cAgree.checked = agreed;
  ui.cOpen.checked = !!profile?.open_release;
  const s = profile?.speaker || {};
  ui.sAge.value = s.age || "";
  ui.sGender.value = s.gender || "";
  ui.sRegion.value = s.region || "";
  ui.sFirst.value = s.first_language || "";
  ui.revoke.hidden = !profile?.consent_version;
  ui.dialog.returnValue = "";
  ui.dialog.showModal();
  return new Promise((resolve) => {
    ui.dialog.addEventListener("close", async () => {
      const v = ui.dialog.returnValue;
      try {
        if (v === "revoke") {
          await api("/rest/v1/rpc/withdraw_consent", { body: "{}", headers: { "content-type": "application/json" } });
          await loadProfile();
          setStatus(ui.status, "Sharing is off. Clips you already shared stay until you delete them from your account.");
          return resolve(false);
        }
        if (v !== "ok") return resolve(false);
        const speaker = {};
        if (ui.sAge.value) speaker.age = ui.sAge.value;
        if (ui.sGender.value) speaker.gender = ui.sGender.value;
        if (ui.sRegion.value) speaker.region = ui.sRegion.value;
        if (ui.sFirst.value) speaker.first_language = ui.sFirst.value;
        await api("/rest/v1/rpc/give_consent", {
          body: JSON.stringify({ version: CONSENT_VERSION, open: ui.cOpen.checked, details: speaker }),
          headers: { "content-type": "application/json" },
        });
        await loadProfile();
        resolve(true);
      } catch (e) {
        setStatus(ui.status, `Couldn't save your consent: ${e.message}.`, "err");
        resolve(false);
      }
    }, { once: true });
  });
}

// ------------------------------------------------------------------ account

ui.accountBtn.onclick = async () => {
  ui.acctEmail.textContent = `Signed in as ${session.user.email}`;
  ui.displayName.value = profile?.display_name || "";
  ui.roleTag.hidden = !profile || profile.role === "contributor";
  ui.roleTag.textContent = profile?.role === "admin" ? "Admin" : "Collector";
  setStatus(ui.acctStatus, "");
  ui.account.showModal();
  await renderClips();
};

async function renderClips() {
  ui.stats.textContent = "";
  ui.myClips.replaceChildren(span("Loading…", "muted"));
  let rows;
  try {
    rows = await json(`/rest/v1/clips?select=id,created_at,text,model_text,duration_s,audio_path&user_id=eq.${session.user.id}&order=created_at.desc&limit=1000`, { method: "GET" });
  } catch (e) {
    return ui.myClips.replaceChildren(span(`Couldn't load your clips: ${e.message}.`, "err"));
  }
  const mins = rows.reduce((a, r) => a + r.duration_s, 0) / 60;
  const fixed = rows.filter((r) => r.text !== r.model_text).length;
  ui.stats.replaceChildren(stat(rows.length, rows.length === 1 ? "clip" : "clips"), stat(mins.toFixed(1), "minutes"), stat(fixed, "corrected"));
  if (!rows.length) return ui.myClips.replaceChildren(span("Nothing shared yet.", "muted"));
  ui.myClips.replaceChildren(...rows.map((r) => {
    const li = document.createElement("li");
    li.textContent = r.text;
    const meta = document.createElement("small");
    const when = new Date(r.created_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    meta.textContent = `${when} · ${Math.round(r.duration_s)} s · `;
    const play = button("Play", "link");
    play.onclick = () => playClip(r, li, play);
    const del = button("Delete", "link danger");
    del.onclick = () => deleteClip(r, del);
    meta.append(play, " · ", del);
    li.append(meta);
    return li;
  }));
}

async function playClip(r, li, btn) {
  btn.disabled = true;
  try {
    const blob = await (await api(`/storage/v1/object/authenticated/clips/${r.audio_path}`, { method: "GET" })).blob();
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = URL.createObjectURL(blob);
    li.append(audio);
    audio.play().catch(() => {});
    btn.nextSibling?.remove(); // the " · " after it
    btn.remove();
  } catch (e) {
    btn.disabled = false;
    setStatus(ui.acctStatus, `Couldn't play it: ${e.message}.`, "err");
  }
}

async function deleteClip(r, btn) {
  if (!confirm("Delete this clip? Its audio and text will be removed.")) return;
  btn.disabled = true;
  try {
    await api(`/storage/v1/object/clips/${r.audio_path}`, { method: "DELETE" });
    await api(`/rest/v1/clips?id=eq.${r.id}`, { method: "DELETE" });
    await renderClips();
  } catch (e) {
    btn.disabled = false;
    setStatus(ui.acctStatus, `Couldn't delete it: ${e.message}.`, "err");
  }
}

ui.nameForm.onsubmit = async (e) => {
  e.preventDefault();
  try {
    await api(`/rest/v1/profiles?id=eq.${session.user.id}`, {
      method: "PATCH", body: JSON.stringify({ display_name: ui.displayName.value.trim() || null }),
      headers: { "content-type": "application/json", prefer: "return=minimal" },
    });
    await loadProfile();
    renderAuth();
    setStatus(ui.acctStatus, "Saved.", "ok");
  } catch (err) {
    setStatus(ui.acctStatus, `Couldn't save: ${err.message}.`, "err");
  }
};

ui.signOut.onclick = async () => {
  try { await api("/auth/v1/logout", { method: "POST" }); } catch { /* signed out locally anyway */ }
  endSession();
  ui.account.close();
};

ui.deleteAccount.onclick = async () => {
  if (!confirm("Delete your account and every clip you shared? This can't be undone.")) return;
  setStatus(ui.acctStatus, "Deleting…");
  try {
    const rows = await json(`/rest/v1/clips?select=audio_path&user_id=eq.${session.user.id}`, { method: "GET" });
    if (rows.length) {
      await api("/storage/v1/object/clips", {
        method: "DELETE", body: JSON.stringify({ prefixes: rows.map((r) => r.audio_path) }),
        headers: { "content-type": "application/json" },
      });
    }
    await api("/rest/v1/rpc/delete_my_account", { body: "{}", headers: { "content-type": "application/json" } });
    endSession();
    ui.account.close();
  } catch (e) {
    setStatus(ui.acctStatus, `Couldn't delete your account: ${e.message}.`, "err");
  }
};

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
function setStatus(el, text, state = "") { el.textContent = text; el.dataset.state = state; }
function span(text, cls) { const s = document.createElement("span"); s.className = cls; s.textContent = text; return s; }
function button(text, cls) { const b = document.createElement("button"); b.type = "button"; b.className = cls; b.textContent = text; return b; }
function stat(n, label) {
  const d = document.createElement("div");
  d.className = "stat";
  d.append(span(String(n), "n"), span(label, "l"));
  return d;
}
function readJSON(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } }
function writeJSON(key, v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* storage off */ } }
