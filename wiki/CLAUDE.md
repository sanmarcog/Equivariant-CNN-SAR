# Wiki Schema — Phase 2: Equivariant SAR Segmentation

This file defines how you (Claude) should read, write, and maintain this wiki.
Follow these instructions in every session that touches the wiki.

---

## Purpose

This wiki is the persistent knowledge base for Phase 2 of the Equivariant CNN for SAR
Avalanche Debris project. It replaces re-deriving context from scratch each session.
You own the wiki layer entirely. The user curates sources and directs analysis.

---

## File layout

```
wiki/
  CLAUDE.md          ← this file (schema + instructions)
  index.md           ← content catalog (one line per page)
  log.md             ← append-only activity record
  goal.md            ← core scientific goal and success criteria
  architecture.md    ← model architecture spec
  datasets.md        ← all datasets, splits, GT details
  evaluation.md      ← metrics, matching protocol, comparison targets
  baselines.md       ← Gattimgatti 2026 and other comparators
  phase1_results.md  ← Phase 1 outcomes carried forward
  size_estimation.md ← D-scale proxy, deposit area, SeNorge fusion
  open_questions.md  ← the 3 key scientific questions + sub-questions
  out_of_scope.md    ← what is explicitly NOT in Phase 2
```

---

## Page conventions

- **One concept per file.** Do not combine architecture and evaluation in one page.
- **Lead with a one-line summary** (the same line that appears in index.md).
- **Use `> ⚠ OPEN` callouts** for unresolved decisions, and `> ✓ DECIDED` for settled ones.
- **Cross-reference with markdown links** (`[datasets](datasets.md)`).
- **Do not duplicate.** If a fact belongs in `datasets.md`, reference it there — do not copy it into `architecture.md`.

---

## Operations

### Ingest (new paper / data source arrives)
1. Read the source.
2. Update every relevant wiki page (can be multiple).
3. Add a one-line entry to `log.md`: `INGEST | YYYY-MM-DD | <source> | <pages updated>`.
4. If the source warrants its own page (key baseline, major dataset), create it and add to `index.md`.

### Query (user asks a scientific question)
1. Read `index.md` to locate relevant pages.
2. Read those pages; synthesize an answer with in-wiki citations.
3. If the answer reveals a gap or new open question, update `open_questions.md`.
4. Log: `QUERY | YYYY-MM-DD | <question summary> | <pages consulted>`.

### Lint (periodic health check — run when asked or when wiki has grown significantly)
1. Check for contradictions across pages.
2. Check for orphaned pages (not linked from index.md).
3. Check for stale `> ⚠ OPEN` items that have since been decided.
4. Check cross-references are valid.
5. Log: `LINT | YYYY-MM-DD | <issues found and fixed>`.

---

## What NOT to store here

- Raw paper text — summarize in a page, link to arXiv.
- Code — lives in the repo; wiki holds decisions and rationale only.
- Git history — use `git log`.
- Ephemeral task lists — use TodoWrite in the active session.
