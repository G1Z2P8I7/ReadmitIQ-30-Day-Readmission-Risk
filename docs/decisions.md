# Decisions and Assumptions Log

Running log of architectural, data, modeling, and evaluation decisions made during development.

| # | Date | Context / Question | Decision | Rationale |
|---|---|---|---|---|
| 1 | Kickoff | Python environment & package structure | Use src/ layout with 'readmit' package; CLI as single source of truth | Windows-compatible, portable across platforms, standard Python packaging. |
| 2 | Kickoff | Initial configuration & invariants | Follow CONTEXT.md & AGENTS.md specs strictly | Zero patient leakage across splits; exclude discharge dispositions 11, 13, 14, 19, 20, 21; one encounter per patient for primary cohort. |
| 3 | M4 Models | Imbalance ablation dependency | Added `imbalanced-learn` to pyproject.toml | Needed for SMOTE resampler in imbalance strategy ablation per Section 8 / M4. |
