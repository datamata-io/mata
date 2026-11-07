---
title: "MATA documentation"
description: "MATA documentation — MATA documentation."
---
# MATA documentation

These Markdown files are the single source of the MATA docs. The site is built with
[Blume](https://useblume.dev) from `../docsite/` and served at `/docs` on the datamata website.

```bash
cd docsite
npm install
npm run dev       # local preview on http://localhost:4321/docs
npm run build     # static output in docsite/dist
npm run validate  # link/anchor check (CI runs this)
```

Writing rules:

- Front matter needs `title`; `description` is expected on every page. Ordering inside a folder comes from
  `sidebar.order` (per page) and `meta.ts` (per folder).
- Internal links are absolute and start with `/docs/` (for example `/docs/inference/cli`). Never link to a
  file path — the link checker resolves URLs, not files.
- Links to repository files (examples, scripts) must be full GitHub URLs.
- `plans/` and `benchmarks/` are excluded from the site: they are internal research, not documentation.
- Known content gaps (pages whose examples do not match the shipped API) are tracked in
  [`../docsite/CONTENT-GAPS.md`](https://github.com/datamata-io/mata/blob/main/docsite/CONTENT-GAPS.md).
