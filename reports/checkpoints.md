# Project Execution Checkpoints

Running progress log tracking milestone delivery, commands executed, verification results, and key metrics.

## Milestone Status Overview

- [x] **M0: Scaffold**
- [ ] **M1: Data audit and cohort**
- [ ] **M2: Leakage audit and splits**
- [ ] **M3: Features and baselines**
- [ ] **M4: LR and XGBoost, imbalance ablation**
- [ ] **M5: Calibration and thresholds**
- [ ] **M6: Explainability (SHAP & odds ratios)**
- [ ] **M7: Fairness audit and mitigation**
- [ ] **M8: Final test evaluation (locked)**
- [ ] **M9: Engineering layer (API, Dashboard, Docker)**
- [ ] **M10: Write-up and executive report**

---

## Checkpoint 0: M0 Scaffold Complete
- **What was built:**
  - Python package structure under src/readmit with 
eadmit v0.1.0 installed in editable mode.
  - Complete configuration suite: configs/base.yaml, configs/features.yaml, configs/models.yaml.
  - CLI dispatcher python -m readmit.cli supporting subcommands: data, eatures, 	rain, evaluate, explain, airness, inal, pp, pi.
  - MLflow tracking wrapper in src/readmit/tracking.py initialized and tested.
  - Makefile with targets matching all CLI commands.
  - Test suite skeleton 	ests/test_scaffold.py verifying package version and config loading.
  - Global cache policy configured to D:\Installs\Global Cache\pip (C: drive preserved).
- **Commands run & results:**
  - 
uff check src tests: Passed (0 errors).
  - pytest -v: 2 passed in 0.10s.
  - python -m readmit.cli --help: Cleanly displays all target subcommands.
- **Assumptions made:**
  - Python 3.13 virtual environment on F: with pinned dependencies in pyproject.toml.
- **Open issues:** None. Ready for M1 (Data audit and cohort).
