#!/usr/bin/env node
// Rewrite Docusaurus-style front matter to Blume's shape, in place.
// - sidebar_position: N  ->  sidebar:\n  order: N
// - ensures a title (falls back to the first H1, then the file name)
import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const ROOT = new URL("../../docs", import.meta.url).pathname;
const SKIP = ["plans/", "benchmarks/"];

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (entry.startsWith(".")) continue;
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.mdx?$/.test(p)) out.push(p);
  }
  return out;
}

let changed = 0;

for (const file of walk(ROOT)) {
  if (SKIP.some((s) => file.includes(`/docs/${s}`))) continue;
  let text = readFileSync(file, "utf8");
  const m = text.match(/^---\n([\s\S]*?)\n---\n/);
  const body = m ? text.slice(m[0].length) : text;
  const lines = m ? m[1].split("\n") : [];

  const pos = lines.find((l) => /^sidebar_position:\s*\d+/.test(l));
  const hasTitle = lines.some((l) => /^title:/.test(l));
  const hasDesc = lines.some((l) => /^description:/.test(l));
  const h1 = body.match(/^#\s+(.+)$/m)?.[1]?.trim();

  const kept = lines.filter((l) => !/^sidebar_position:/.test(l));
  if (pos && !kept.some((l) => /^sidebar:/.test(l))) {
    kept.push("sidebar:", `  order: ${pos.match(/\d+/)[0]}`);
  }
  if (!hasTitle) {
    kept.unshift(`title: ${JSON.stringify(h1 ?? file.split("/").pop().replace(/\.mdx?$/, ""))}`);
  }
  if (!hasDesc && h1) kept.push(`description: ${JSON.stringify(`${h1} — MATA documentation.`)}`);

  const next = `---\n${kept.join("\n")}\n---\n${body}`;
  if (next !== text) {
    writeFileSync(file, next);
    changed++;
    console.log("normalized", file.replace(ROOT + "/", ""));
  }
}
console.log(`\n${changed} file(s) normalized`);
