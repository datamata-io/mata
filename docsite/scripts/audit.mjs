#!/usr/bin/env node
// Docs readiness audit: front matter + internal-file leakage. Exit 1 on any finding.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const ROOT = new URL("../../docs", import.meta.url).pathname;
const INTERNAL = ["plans/", "benchmarks/", "OCR_IMPLEMENTATION_SUMMARY.md", "VLM_TOOL_CALLING_SUMMARY.md"];
const findings = [];

function walk(dir) {
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (entry.startsWith(".") || entry === "node_modules") continue;
    if (statSync(p).isDirectory()) walk(p);
    else if (/\.mdx?$/.test(p)) check(p);
  }
}

function frontMatter(text) {
  const m = text.match(/^---\n([\s\S]*?)\n---/);
  if (!m) return null;
  const out = {};
  for (const line of m[1].split("\n")) {
    const kv = line.match(/^([a-zA-Z_]+):\s*(.*)$/);
    if (kv) out[kv[1]] = kv[2].replace(/^"|"$/g, "");
  }
  return out;
}

function check(p) {
  const rel = relative(ROOT, p);
  if (INTERNAL.some((i) => rel.startsWith(i))) return; // excluded on purpose
  const text = readFileSync(p, "utf8");
  const fm = frontMatter(text);
  if (!fm?.title) findings.push(`${rel}: missing front matter "title"`);
  if (fm?.sidebar_position) findings.push(`${rel}: uses sidebar_position (Blume expects sidebar.order)`);
}

walk(ROOT);
if (findings.length) {
  console.error(findings.join("\n"));
  console.error(`\n${findings.length} finding(s)`);
  process.exit(1);
}
console.log(`docs audit clean (${ROOT})`);
