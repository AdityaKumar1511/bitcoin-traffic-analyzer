# Contributing Guide

This document defines how our team splits work, our development workflow, and our sprint timeline for **Bitcoin Traffic Analyzer** (SIH26146).

---

## Table of Contents

- [Team & Task Division](#team--task-division)
- [Sprint Timeline](#sprint-timeline)
- [Interface Contracts](#interface-contracts)
- [Git Workflow](#git-workflow)
- [Coding Conventions](#coding-conventions)
- [Daily Sync](#daily-sync)
- [Definition of Done](#definition-of-done)

---

## Team & Task Division

Work is split into **three tracks**, each owning an end-to-end vertical slice so members can work in parallel without blocking each other. Every track maps directly to folders in `src/`.

### Track 1 — Data & Graph Foundation
**Owner:** _Aditya_

| Module | Deliverable | Location |
|---|---|---|
| Synthetic dataset generator | Normal traffic + injected criminal patterns + hidden ground truth labels | `data/synthetic_generator.py` |
| Ingestion | CSV/JSON/XML parsers → unified schema, validation | `src/ingestion/` |
| GeoIP enrichment | Offline country/ASN lookup via GeoLite2 | `src/ingestion/` |
| Graph construction | Wallet–Transaction–IP graph, common-input-ownership + change-address heuristics | `src/graph/` |
| Evasion detectors | CoinJoin, peel-chain, mixer signature detection | `src/models/evasion_detectors.py` |

> This is the **critical path** — everything downstream depends on it. Start Day 1, prioritize getting a rough version of every module working end-to-end before polishing any one piece.

### Track 2 — AI/ML Models
**Owner:** _Rudra_

| Module | Deliverable | Location |
|---|---|---|
| Feature engineering | Behavioral, temporal, network features per wallet/transaction | `src/features/` |
| Anomaly detection | Isolation Forest + Autoencoder | `src/models/anomaly.py` |
| Clustering | Node2Vec embeddings + HDBSCAN | `src/models/clustering.py` |
| Taint propagation | Multi-hop risk scoring through the graph | `src/models/taint_propagation.py` |
| Network-blockchain correlation | IP/ASN-to-cluster confidence scoring | `src/correlation/` |
| Composite scoring | Combine all signals into final ranked risk score | `src/scoring.py` |
| Explainability | SHAP integration + plain-English reason generator | `src/explainability/` |
| *(Stretch)* GNN classifier | GraphSAGE/XGBoost, if time allows | `src/models/classifier.py` |

> This is the **technical core** judges will scrutinize most. Get a baseline (Isolation Forest + simple clustering) working fast, then layer in taint propagation and correlation scoring.

### Track 3 — Dashboard, UX & Docs
**Owner:** _Sweety_

| Module | Deliverable | Location |
|---|---|---|
| Streamlit dashboard | Overview, alert table, detail view, link-analysis graph | `dashboard/app.py` |
| Case export | PDF evidence report generator | `dashboard/report_export.py` |
| Feedback loop | Confirm/reject UI + recalibration hook | `src/feedback/` |
| Technical write-up | Approach, model choices, explainability method | `docs/technical_writeup.md` |
| README maintenance | Keep roadmap checkboxes and docs current | `README.md` |
| Demo script | "Day in the life of an investigator" narrative for judging | `docs/` |

> This is what judges **see first**. A polished dashboard and confident demo narrative matter as much as backend correctness — don't leave this to the last day.

---

## Sprint Timeline

A 6-day sprint plan. Adjust dates to your actual hackathon window, but keep the sequencing — Track 1 must produce usable output before Track 2 can build on it.

| Day | Track 1 (Data & Graph) | Track 2 (ML Models) | Track 3 (Dashboard & Docs) |
|---|---|---|---|
| **1** | Synthetic generator + ingestion pipeline | Plan feature list, wait on sample data | Repo scaffold, dashboard wireframe |
| **2** | Graph construction (heuristics) | Feature engineering on early sample data | Streamlit skeleton, static mockups |
| **3** | Evasion detectors (CoinJoin, peel-chain) | Anomaly detection + clustering | Wire dashboard to partial real outputs |
| **4** | Support integration, fix data edge cases | Taint propagation + correlation scoring | Link-analysis graph, case export |
| **5** | Integration testing, buffer for bugs | SHAP explainability, composite scoring | Feedback loop, UI polish |
| **6** | Full pipeline dry run | Evaluation metrics, false-positive testing | Write-up, demo script, rehearsal |

**Day 5–6 are integration + polish days for everyone** — no new features, just making the full pipeline run cleanly end-to-end and rehearsing the demo.

---

## Interface Contracts

To let all three tracks work in parallel without blocking each other, agree on these data contracts **on Day 1** and don't change them without a heads-up to the team:

1. **Ingestion → Graph**: a pandas DataFrame with the exact normalized schema (see README `Dataset` section) — Track 1 internal, but Track 2 should know the columns exist even before the graph is built.
2. **Graph → Features**: a `networkx` graph object with node types `wallet`, `transaction`, `ip`, and documented edge attributes (`amount`, `timestamp`, `port`).
3. **Features → Models**: a DataFrame indexed by `wallet_id` with one row per wallet, one column per feature — Track 2 should publish this schema early so Track 3 can build dashboard tables against mock data before real scores exist.
4. **Models → Dashboard**: a final DataFrame/CSV: `entity_id, risk_score, top_reasons[], cluster_id, related_ips[], related_wallets[]` — this is the single handoff file Track 3 needs; agree on its exact column names by Day 2 so dashboard work isn't blocked.

**Tip:** Track 3 can build the entire dashboard against a small hand-written mock CSV matching the final schema, starting Day 1 — no need to wait for real model output.

---

## Git Workflow

- **Branches:** each track works on its own branch (`track1-data`, `track2-ml`, `track3-dashboard`), merging into `main` at the end of each day or milestone.
- **Commits:** small, descriptive commits (`git commit -m "Add common-input-ownership heuristic"`), not one giant end-of-day dump.
- **Pull Requests:** even solo, open a PR before merging into `main` — gives a quick self-review checkpoint and keeps `main` always demo-ready.
- **Merge conflicts:** since tracks map to separate folders, conflicts should be rare. If two people touch `scoring.py` or `app.py`, coordinate directly before merging.
- **`main` must always run.** Never leave `main` in a broken state overnight — if something's half-done, keep it on your branch.

---

## Coding Conventions

- Python 3.11+, follow **PEP 8**.
- Type hints on function signatures where practical (`def score_wallet(wallet_id: str) -> float:`).
- Every module gets a short docstring explaining what it does and its expected input/output.
- No hardcoded file paths — use `pathlib` and config constants (important for the Windows/WSL split across the team).
- Keep notebooks in `notebooks/` for exploration only — production logic belongs in `src/`, not notebooks.

---

## Daily Sync

A **10-minute check-in** (voice call or chat) each day covering:
1. What did I finish yesterday?
2. What am I doing today?
3. Am I blocked on anything from another track?

This is cheap insurance against integration surprises showing up on Day 5.

---

## Definition of Done

A module is "done" when:
- [ ] It runs end-to-end on the synthetic dataset without errors
- [ ] It matches the agreed interface contract (input/output schema)
- [ ] It's committed to the track branch and merged into `main`
- [ ] It has at least a one-line docstring/comment explaining what it does
- [ ] The relevant README/roadmap checkbox is updated

---

*Questions or blockers? Raise them in the daily sync, don't sit on them — with a 6-day timeline, a half-day blocked is a real cost.*
