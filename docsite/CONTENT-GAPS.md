# Content gaps found while building these docs (internal — not published)

These pages were imported from `datamata-io/main_website` (`packages/content/docs/v0`) during the Blume
migration. Several of them describe an API surface that **does not exist in this repository**. They are
structurally fine (they build and link-check) but are **not factual** yet. Do not publish them as-is.

Verified against this repo (2026-11-07, commit `be6006a`):

| Claim in the imported page(s) | Reality in the repo | Pages |
|---|---|---|
| CLI is invoked as `datamata <command>` | `pyproject.toml` → `[project.scripts] mata = "mata.cli:main"` | getting-started/{install,quickstart,overview}, concepts/model-packs, deployment/index, integrations/index, migration/index |
| CLI commands `pull`, `predict`, `bench`, `list`, `info`, `serve` | Only `run`, `recognize`, `track`, `val`, `annotate`, `export` are registered in `src/mata/cli.py` | same as above |
| `KACA.from_pack("kaca-nano")` and `datamata.kaca.KACA` | No `KACA` class and no `from_pack` anywhere in `src/mata` | getting-started/{quickstart,overview}, concepts/model-packs, reference/api |
| `datamata serve <model> --port 8080` | No server entry point (`fastapi`/`uvicorn` absent from `src/mata`) | deployment/index, integrations/index |
| `datamata predict --output ...` (migration notes) | No `predict` command | migration/index |

## Suggested resolutions (pick per page, do not do both)

1. **Rewrite against the current release** — replace the examples with the real API
   (`mata.run("detect", "image.jpg", model=...)`, `mata annotate`, `mata export`) and drop the
   nonexistent commands. Best for anything that ships to users now.
2. **Keep as v1-forward** — if these pages really describe the planned `datamata` 1.0 surface, add a short
   banner at the top of each: *"This page describes the upcoming 1.0 API; the current release ships
   `mata run|recognize|track|val|annotate|export` — see the CLI page."* Then they are honest instead of wrong.

Pages that are accurate and can ship today: `inference/cli.md` (from `CLI_REFERENCE.md`), `guides/*`,
`graph/*`, `training/guide.md`, `validation/index.md`, `reference/{supported-models,quick-reference}.md`,
`concepts/architecture.md`.
