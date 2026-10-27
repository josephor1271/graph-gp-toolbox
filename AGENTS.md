# Repository working rules

- Search existing source and callers before writing code. Before adding a module,
  give a two-sentence rationale naming the code reused or duplication removed and
  explaining why the existing path cannot handle the authorized task.
- New modules require a written reason and an explicit owner-approved scope
  exception. An experiment does not automatically get new workers, coordinators,
  reports, case modules, plotting machinery or multiple new test files.
- Keep Python readable: no semicolon-compressed statements, one-line functions or
  control-flow bodies, multi-action lambdas, dynamic globals or clever indirection.
  Use small cohesive functions, useful typing and comments explaining why.
  Do not mix unrelated refactors into a task.
- Every PR records baseline and targeted test results, failure accounting and
  exact empirical claims. Numerical changes also need an independent reference.
  Never hide resampling, adjust thresholds to pass, manufacture wins or mark an
  incomplete benchmark successful.
- Performance comparisons separate representation from inference, baseline from
  candidate, accuracy from elapsed time, and measured costs from projections.
  Do not claim a speedup for an unmeasured path.
- Commit compact sufficient evidence: preserve reports, reproducible figure
  inputs, provenance and unique diagnostics. Keep large raw JSON/arrays in ignored
  results/. Two wide tables covering the same rows/fields require a documented
  reason. Do not rewrite prior evidence opportunistically or delete unique data.
- Default PR budget: at most 500 added lines and 15 changed paths unless a
  project-leader-authored work order explicitly overrides it. For maintenance,
  prefer deletions and simpler logic. If scope exceeds the authorized budget,
  stop and ask for a newly scoped work order; do not silently create another PR.
- Do not add a dependency or architecture without explicit justification and
  approval. Preserve public APIs and documented experiment reproduction.
- End with a concise changed/deleted/kept manifest and outstanding issues.

Stage 06 only: its authorized work order allows 650 added lines and 16 changed
paths, one new Markdown file, no new Python modules/dependencies/test files, and
only the two specified redundant CSV deletions. Do not initiate Stage 07 without
new authorization.
