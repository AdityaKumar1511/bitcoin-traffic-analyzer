# Bitcoin Traffic Analyzer

**AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic**

An offline system that ingests bulk Bitcoin transaction and network metadata, correlates network-layer signals (IP / port / timing) with blockchain-layer data (wallets / TXIDs / amounts), and applies AI/ML to detect anomalies, cluster entities, and generate ranked, explainable investigative leads — presented through an interactive dashboard.

Built for **SIH26146** · National Technical Research Organisation (NTRO) · Theme: **Blockchain & Cybersecurity**

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Explainability](#explainability)
- [Evaluation](#evaluation)
- [Roadmap](#roadmap)
- [Team](#team)
- [License](#license)

---

## Problem Statement

Bitcoin's pseudonymous, peer-to-peer design lets illicit actors move, layer, and cash out funds — ransomware payments, darknet-market proceeds, extortion, and laundering — while evading traditional financial surveillance.

This project builds a complete **offline** system that:
1. Ingests bulk Bitcoin transaction/network metadata (CSV/JSON/XML)
2. Correlates network-layer observations (IP/port/timing) with blockchain-layer data (wallet/TXID/amount)
3. Applies AI/ML to detect anomalies, cluster entities, and generate prioritized, explainable investigative leads

> Full technical breakdown available in [`docs/technical_writeup.md`](docs/technical_writeup.md).

## Solution Overview

Given a dataset of "who broadcast which Bitcoin transaction, from what IP, at what time" plus "which wallets sent how much to which wallets" — the system automatically:

- Flags wallets/transactions that look like criminal money-flow patterns (ransomware collection, laundering, darknet payments)
- Groups wallets that likely belong to the same real-world actor
- Ranks the most suspicious entities with a confidence score
- Explains, in plain language, *why* each entity was flagged
- Displays everything on an interactive, offline dashboard

No live blockchain node, no internet dependency, no cloud APIs — everything runs locally on Linux (via WSL2 on Windows).

## Key Features

- **Multi-format ingestion** — CSV, JSON, and XML parsers normalized into a unified schema
- **Offline GeoIP enrichment** — country/ASN lookup via a local MaxMind GeoLite2 database
- **Entity/transaction graph** — heterogeneous graph linking wallets, transactions, and IPs
- **Forensic heuristics** — common-input-ownership clustering, change-address detection, peel-chain identification, fan-in/fan-out analysis
- **AI/ML detection**
  - Unsupervised anomaly detection (Isolation Forest, Autoencoder)
  - Graph embeddings (Node2Vec) + density-based clustering (HDBSCAN)
  - Optional GNN/XGBoost classifier for supervised risk scoring
- **Network-blockchain correlation scoring** — quantifies how strongly an IP/ASN pattern ties to a wallet cluster
- **Explainable AI** — SHAP-backed, human-readable justification for every alert
- **Composite risk scoring** — ranked, filterable list of investigative leads
- **Interactive dashboard** — Streamlit-based UI with link-analysis graph visualization
- **Fully offline** — no external API calls at runtime; runs standalone on Linux

## Architecture

```
┌──────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│  1. Ingestion      │ →  │ 2. Graph Builder    │ →  │ 3. Feature Engine   │
│  CSV/JSON/XML      │     │ Wallet–TX–IP graph  │     │ Per-node/edge       │
│  parser + cleaner  │     │ (NetworkX/Neo4j)    │     │ features             │
└──────────────────┘     └───────────────────┘     └────────┬───────────┘
                                                              ↓
┌──────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│ 6. Dashboard       │ ←  │ 5. Explainability   │ ←  │ 4. ML Detection     │
│ Streamlit + link-   │     │ SHAP + composite    │     │ Anomaly detection   │
│ analysis graph      │     │ risk scoring         │     │ + clustering + GNN  │
└──────────────────┘     └───────────────────┘     └────────────────────┘
```

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| Data handling | pandas, NumPy |
| Storage | SQLite / Parquet |
| GeoIP | MaxMind GeoLite2 (offline `.mmdb`) + `geoip2` |
| Graph | NetworkX (+ optional Neo4j Community) |
| Graph embeddings | Node2Vec / GraphSAGE (PyTorch Geometric) |
| Anomaly detection | scikit-learn (Isolation Forest), PyTorch (Autoencoder) |
| Clustering | HDBSCAN |
| Supervised model (optional) | XGBoost / GCN |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Graph visualization | Pyvis / Plotly |

## Project Structure

```
bitcoin-traffic-analyzer/
├── data/
│   ├── raw/                    # input CSV/JSON/XML (gitignored)
│   ├── geoip/                  # GeoLite2 .mmdb (gitignored)
│   └── synthetic_generator.py  # synthetic dataset generator
├── src/
│   ├── ingestion/               # parsers + normalization
│   ├── graph/                   # graph builder + forensic heuristics
│   ├── features/                # feature engineering
│   ├── models/                  # anomaly detection, clustering, classifier
│   ├── explainability/          # SHAP wrapper + reason-string generator
│   └── scoring.py               # composite risk score
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── notebooks/                   # exploratory analysis
├── docs/
│   └── technical_writeup.md     # detailed approach & methodology
├── tests/
├── requirements.txt
├── run.sh
└── README.md
```

## Getting Started

### Prerequisites

- Linux or WSL2 (Ubuntu recommended)
- Python 3.11+
- ~2 GB free disk space (for GeoIP database + dependencies)

### Installation

```bash
git clone https://github.com/<your-username>/bitcoin-traffic-analyzer.git
cd bitcoin-traffic-analyzer

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### GeoIP Database Setup

1. Create a free account at [MaxMind](https://www.maxmind.com/en/geolite2/signup)
2. Download `GeoLite2-Country.mmdb` and `GeoLite2-ASN.mmdb`
3. Place both files in `data/geoip/`

### Generate Synthetic Dataset

```bash
python data/synthetic_generator.py --output data/raw/synthetic_transactions.csv
```

## Usage

**Run the full pipeline:**
```bash
bash run.sh
```

**Run the dashboard:**
```bash
streamlit run dashboard/app.py
```

Then open `http://localhost:8501` in your browser.

**Run individual stages:**
```bash
python -m src.ingestion.parser --input data/raw/synthetic_transactions.csv
python -m src.graph.builder
python -m src.features.engineer
python -m src.models.anomaly
python -m src.scoring
```

## Dataset

Since real seized/intercepted Bitcoin data is sensitive and cannot be distributed, this project uses a **synthetic dataset** modeled on real Bitcoin P2P/transaction fields.

**Minimum schema:**

| Field | Description |
|---|---|
| `timestamp` | Transaction broadcast time |
| `src_ip`, `dst_ip`, `src_port`, `dst_port` | Network-layer relay data |
| `txid` | Transaction ID |
| `input_addresses[]`, `output_addresses[]` | Wallets involved |
| `input_amounts[]`, `output_amounts[]` | BTC amounts |
| `fee`, `script_type` | Transaction metadata |
| `geo_country`, `asn` | Derived via offline GeoIP lookup |

The synthetic generator injects realistic criminal patterns (ransomware collection, laundering fan-in/fan-out, peel chains, same-actor multi-wallet clusters) alongside normal traffic, with hidden ground-truth labels used for internal evaluation.

## Methodology

1. **Ingestion & normalization** — unify CSV/JSON/XML into a common schema; enrich with offline GeoIP
2. **Graph construction** — build a wallet–transaction–IP graph; apply common-input-ownership and peel-chain heuristics
3. **Feature engineering** — compute per-wallet and per-transaction behavioral, temporal, and network features
4. **AI/ML detection**
   - Isolation Forest / Autoencoder for anomaly scoring
   - Node2Vec embeddings + HDBSCAN for entity clustering
   - Optional GNN/XGBoost classifier for supervised risk scoring
5. **Network-blockchain correlation** — score IP/ASN/timing correlation against wallet clusters
6. **Composite risk scoring** — combine all signals into a single ranked score
7. **Explainability** — generate SHAP-backed, human-readable justifications per alert
8. **Visualization** — present ranked alerts and link-analysis graph via dashboard

## Explainability

Every flagged wallet/transaction includes:
- A numeric **risk score** (0–1)
- The **top contributing features** (via SHAP)
- An auto-generated **plain-English justification**
- Traceable **evidence links** (related transactions, wallets, and IPs)

Example:
> Flagged due to high fan-in ratio (0.91), shared broadcast IP with 4 other wallets within 60 seconds, and origin ASN flagged as high-risk hosting.

## Evaluation

Model performance is validated against self-injected synthetic ground truth:
- Precision / recall / F1 of flagged entities vs. known-injected criminal patterns
- Cluster purity against deliberately co-spent wallet groups
- False-positive rate on pure "normal traffic" runs
- Manual review of top-N alerts for explanation coherence

## Roadmap

- [ ] Core ingestion pipeline (CSV/JSON/XML)
- [ ] Graph construction + forensic heuristics
- [ ] Feature engineering
- [ ] Isolation Forest anomaly detection
- [ ] Node2Vec + HDBSCAN clustering
- [ ] SHAP explainability layer
- [ ] Composite risk scoring
- [ ] Streamlit dashboard + link-analysis graph
- [ ] Optional GNN classifier
- [ ] Technical write-up
- [ ] Demo video / presentation

## Team

| Name | Role |
|---|---|
| _Add team member_ | _Role_ |
| _Add team member_ | _Role_ |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Submitted for Smart India Hackathon — Problem Statement SIH26146, National Technical Research Organisation (NTRO).*
