// Kreol text to speech with Meta's MMS VITS model (facebook/mms-tts-mfe), exported to
// ONNX by tools/export_tts.py. Nothing here touches the page, so Node can import it
// (tools/check_tts_tokenizer.mjs checks tokenize() against the Hugging Face tokenizer).

// Same steps as transformers' VitsTokenizer with normalize on: lowercase, drop every
// character the model has no token for, trim, then put the blank token (0) between characters.
// One addition: a letter the model lacks is first tried without its accent, so "été"
// reads as "ete" instead of "t".
export function tokenize(text, table) {
  const vocab = table.vocab;
  const keys = Object.keys(vocab);
  let lowered = "";
  const s = Array.from(text);
  for (let i = 0; i < s.length;) {
    const hit = table.normalize && keys.find((k) => s.slice(i, i + Array.from(k).length).join("") === k);
    if (hit) { lowered += hit; i += Array.from(hit).length; continue; }
    lowered += table.normalize ? s[i].toLowerCase() : s[i];
    i++;
  }
  let kept = "";
  for (const c of lowered) {
    if (Object.hasOwn(vocab, c)) kept += c;
    else for (const b of foldAccent(c)) if (Object.hasOwn(vocab, b)) kept += b;
  }
  const chars = Array.from(kept.trim());
  if (!chars.length) return [];
  const ids = chars.map((c) => vocab[c]);
  if (!table.add_blank) return ids;
  const out = new Array(ids.length * 2 + 1).fill(table.blank_id ?? 0);
  ids.forEach((id, i) => (out[2 * i + 1] = id));
  return out;
}

export function foldAccent(c) {
  return c.normalize("NFKD").replace(/\p{M}/gu, "");
}

// The model is happiest with a sentence at a time, and long inputs cost memory, so read
// sentence by sentence, splitting anything still too long at commas, then at spaces.
export function sentences(text, max = 200) {
  const out = [];
  for (const sentence of text.split(/(?<=[.!?;:…])\s+|\n+/)) {
    let rest = sentence.trim();
    while (rest.length > max) {
      const head = rest.slice(0, max);
      let cut = Math.max(head.lastIndexOf(", "), head.lastIndexOf(" — "));
      if (cut < max * 0.4) cut = head.lastIndexOf(" ");
      if (cut < 1) cut = max;
      out.push(rest.slice(0, cut + 1).trim());
      rest = rest.slice(cut + 1).trim();
    }
    if (rest) out.push(rest);
  }
  return out;
}

export function createVoice(ort, session, table) {
  const scalar = (x) => new ort.Tensor("float32", new Float32Array([x]), []);
  return {
    sampleRate: table.sampling_rate,
    // rate: 1 is the model's normal pace; below 1 is slower.
    async synth(text, rate = 1) {
      const ids = tokenize(text, table);
      if (!ids.length) return null;
      const res = await session.run({
        input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]),
        noise_scale: scalar(table.noise_scale),
        noise_scale_duration: scalar(table.noise_scale_duration),
        speaking_rate: scalar(table.speaking_rate * rate),
      });
      return res.waveform.data;
    },
  };
}

// 16-bit PCM WAV, for saving what was read.
export function wavBlob(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const x = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}
