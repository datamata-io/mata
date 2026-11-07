import { defineConfig } from "blume";

export default defineConfig({
  title: "MATA Documentation",
  description:
    "MATA — production-grade computer vision: inference, recognition, tracking, training and validation.",
  content: {
    root: "../docs",
    include: ["**/*.{md,mdx}"],
    exclude: [
      "plans/**",
      "benchmarks/**",
      "OCR_IMPLEMENTATION_SUMMARY.md",
      "VLM_TOOL_CALLING_SUMMARY.md",
    ],
  },
  // Served under /docs on the datamata origin (Phase 5). Astro's base prefixes
  // both routes and asset URLs, which is what a sub-path mount needs.
  deployment: {
    site: "https://datamata.io",
    base: "/docs",
  },
  agents: { llmsTxt: true },
  seo: { sitemap: true, robots: true },
});
