// Check web/tts.js turns text into the same token ids as the Hugging Face tokenizer.
//   node tools/check_tts_tokenizer.mjs build/tts
import { readFileSync } from "node:fs";
import { join } from "node:path";

// web/ has no package.json, so load the module from its source rather than its path.
const src = readFileSync(new URL("../web/tts.js", import.meta.url), "utf8");
const { tokenize } = await import(`data:text/javascript,${encodeURIComponent(src)}`);

const dir = process.argv[2];
const table = JSON.parse(readFileSync(join(dir, "tts.json"), "utf8"));
const cases = JSON.parse(readFileSync(join(dir, "tokenizer-cases.json"), "utf8"));
let bad = 0;
for (const { text, ids } of cases) {
  const got = tokenize(text, table);
  if (JSON.stringify(got) !== JSON.stringify(ids)) {
    bad++;
    console.error(`MISMATCH ${JSON.stringify(text)}\n  python ${JSON.stringify(ids)}\n  js     ${JSON.stringify(got)}`);
  }
}
console.log(`${cases.length - bad} of ${cases.length} tokenizer cases match`);
process.exit(bad ? 1 : 0);
