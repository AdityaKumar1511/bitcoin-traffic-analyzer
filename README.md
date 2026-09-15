<div align="center">

# 🔗 Bitcoin Traffic Analyzer

### AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

*Correlating network-layer signals with blockchain-layer data to surface explainable, investigator-ready leads — fully offline.*

**Smart India Hackathon · Problem Statement SIH26146**
**Organisation:** National Technical Research Organisation (NTRO) · **Theme:** Blockchain & Cybersecurity

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Offline](https://img.shields.io/badge/Runtime-100%25%20Offline-success)]()
[![Platform](https://img.shields.io/badge/Platform-Linux%20%2F%20WSL2-orange?logo=linux&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow)]()

</div>

---

## 📖 Table of Contents

- [Problem Statement](#-problem-statement)
- [Why This Is Hard](#-why-this-is-hard)
- [Solution Overview](#-solution-overview)
- [What Makes This Different](#-what-makes-this-different)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Methodology](#-methodology-deep-dive)
- [Explainability](#-explainability)
- [Evaluation](#-evaluation)
- [Existing Solutions & Prior Art](#-existing-solutions--prior-art)
- [Roadmap](#-roadmap)
- [Team](#-team)
- [License](#-license)

---

## 🎯 Problem Statement

Bitcoin's pseudonymous, peer-to-peer design lets illicit actors move, layer, and cash out funds — ransomware payments, darknet-market proceeds, extortion, and laundering — while evading traditional financial surveillance.

Every Bitcoin transaction is permanently recorded on a public ledger, but wallet addresses aren't tied to real identities. The one crack in this anonymity: to broadcast a transaction, a wallet must connect to the Bitcoin P2P network via a real IP address — creating a rare, fleeting link between an on-chain identity (wallet) and a real-world signal (network location).

This project builds a complete **offline** system that:

1. **Ingests** bulk Bitcoin transaction/network metadata (CSV/JSON/XML)
2. **Correlates** network-layer observations (IP/port/timing) with blockchain-layer data (wallet/TXID/amount)
3. **Applies AI/ML** to detect anomalies, cluster entities, and generate prioritized, explainable investigative leads
4. **Visualizes** findings via a dashboard/link-analysis interface — fully offline, on Linux

> 📄 Full technical breakdown available in [`docs/technical_writeup.md`](docs/technical_writeup.md).

## 🧩 Why This Is Hard

| Challenge | Why it matters |
|---|---|
| **No ground truth** | We rarely know for certain which wallets are criminal — this is fundamentally an unsupervised/semi-supervised problem, not simple classification |
| **Evasion techniques** | Criminals use mixers/tumblers, CoinJoin, peel chains, and one-time addresses — all of which look statistically "weird" even when innocent, causing false positives |
| **Noisy network data** | An observed IP might be a VPN exit node, Tor relay, or innocent peer just forwarding traffic — not the true sender |
| **Explainability requirement** | Agencies can't act on a black-box score — every flag must be justifiable to a human investigator |
| **Offline constraint** | No live blockchain node, no cloud APIs — everything (including GeoIP) must run locally on Linux |

## 💡 Solution Overview

Given a dataset of *"who broadcast which Bitcoin transaction, from what IP, at what time"* plus *"which wallets sent how much to which wallets"* — the system automatically:

- Flags wallets/transactions matching criminal money-flow patterns (ransomware collection, laundering, darknet payments)
- Groups wallets that likely belong to the same real-world actor
- Ranks the most suspicious entities with a confidence score
- Explains, in plain language, *why* each entity was flagged
- Displays everything on an interactive, offline dashboard

No live blockchain node, no internet dependency, no cloud APIs — everything runs locally on Linux (via WSL2 on Windows).

---

## 🏆 What Makes This Different

The core building blocks here — graph analysis, anomaly detection, clustering, explainability — are the *correct* tools for this problem, and any competent solution converges on them. Our differentiation comes from **depth and execution** in the areas most teams skip under time pressure:

### 1. Rigorous Network-Blockchain Correlation *(the actual NTRO-specific ask)*
Most on-chain forensic tools (Chainalysis, Elliptic, GraphSense) analyze the blockchain layer only. We go further by building a genuine **IP-to-wallet correlation model** that:
- Weighs confidence by number of independent observations tying an IP/ASN to a wallet cluster (not a single coincidence)
- Explicitly discounts confidence for known VPN/Tor/hosting-provider ASNs to avoid false correlation
- Produces a **calibrated correlation score**, not a binary yes/no link

### 2. Multi-Hop Taint Propagation
Inspired by real forensic techniques used by agencies (the "poison"/"haircut" model): if a wallet is known/suspected bad, its risk **propagates through the transaction graph** to wallets it interacts with, decaying with distance — similar to tracing contamination spreading and weakening through a network. Most hackathon-scale projects stop at single-wallet scoring; we model risk as something that flows.

### 3. Adversarial-Awareness (Evasion-Specific Detection)
Rather than generic anomaly detection, we explicitly detect known **laundering evasion signatures**:
- **CoinJoin-style transactions** (multiple parties mixing coins to break traceability)
- **Peeling chains** (large amounts repeatedly split with a small "peel" forwarded onward)
- **Mixer/tumbler behavioral fingerprints**

Each detector maps to a real, documented laundering technique — showing domain understanding, not just applied ML.

### 4. Investigator-Ready Workflow, Not Just a Demo Dashboard
- **Case export** — auto-generated evidence report (PDF) per flagged wallet: summary, graph snapshot, transaction timeline
- **Analyst feedback loop** — mark alerts "confirmed" / "false positive" to recalibrate scoring over time
- **Full audit trail** — every score is logged and traceable back to the exact evidence that produced it

### 5. Honest False-Positive Analysis
We explicitly test and report the false-positive rate on **pure normal traffic** (e.g., legitimate exchanges naturally resemble laundering hubs by raw fan-in/fan-out) and document mitigations like exchange-pattern whitelisting — rather than only demoing cherry-picked "wins."

### 6. Temporal Behavior Modeling
Beyond static snapshots, we track how wallet behavior **evolves over time** — sudden activity spikes, dormant-then-active cash-out patterns — using time-windowed features, catching patterns a single-snapshot model would miss.

---

## ✨ Key Features

- **Multi-format ingestion** — CSV, JSON, and XML parsers normalized into a unified schema
- **Offline GeoIP enrichment** — country/ASN lookup via a local MaxMind GeoLite2 database
- **Entity/transaction graph** — heterogeneous graph linking wallets, transactions, and IPs
- **Forensic heuristics** — common-input-ownership clustering, change-address detection, peel-chain identification, fan-in/fan-out analysis
- **AI/ML detection**
  - Unsupervised anomaly detection (Isolation Forest, Autoencoder)
  - Graph embeddings (Node2Vec) + density-based clustering (HDBSCAN)
  - Optional GNN/XGBoost classifier for supervised risk scoring
- **Network-blockchain correlation scoring** — quantifies how strongly an IP/ASN pattern ties to a wallet cluster, with confidence calibration
- **Multi-hop taint propagation** — suspicion score flows through the transaction graph, decaying with distance
- **Evasion-specific detectors** — CoinJoin, peel-chain, and mixer signature detection
- **Explainable AI** — SHAP-backed, human-readable justification for every alert
- **Composite risk scoring** — ranked, filterable list of investigative leads
- **Interactive dashboard** — Streamlit-based UI with link-analysis graph visualization
- **Case export** — auto-generated PDF evidence reports per flagged entity
- **Analyst feedback loop** — confirm/reject alerts to recalibrate future scoring
- **Fully offline** — no external API calls at runtime; runs standalone on Linux

---

## 🏗️ Architecture

```
┌──────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│  1. Ingestion     │     │ 2. Graph Builder   │     │ 3. Feature Engine  │
│  CSV/JSON/XML     │ →   │ Wallet–TX–IP graph │  →  │ Per-node/edge      │
│  parser + cleaner │     │ (NetworkX/Neo4j)   │     │ + temporal features│
└──────────────────┘     └───────────────────┘     └────────┬───────────┘
                                                              ↓
┌──────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│ 7. Dashboard      │     │ 6. Explainability  │     │ 4. ML Detection    │
│ Streamlit + link- │  ←  │ SHAP + composite   │  ←  │ Anomaly detection  │
│ analysis + export │     │ risk scoring       │     │ + clustering + GNN │
└──────────────────┘     └─────────┬─────────┘     └────────┬───────────┘
                                    ↑                          ↓
                          ┌───────────────────┐     ┌────────────────────┐
                          │ 8. Feedback Loop   │     │ 5. Taint Propagation│
                          │ Analyst confirm/   │  ←  │ + Network-Blockchain│
                          │ reject → recalibrate│     │ Correlation Scoring │
                          └───────────────────┘     └────────────────────┘
```

---

## 🛠️ Tech Stack

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
| Report export | ReportLab / WeasyPrint (PDF generation) |

---

## 📁 Project Structure

```
bitcoin-traffic-analyzer/
├── data/
│   ├── raw/                       # input CSV/JSON/XML (gitignored)
│   ├── geoip/                     # GeoLite2 .mmdb (gitignored)
│   └── synthetic_generator.py     # synthetic dataset generator
├── src/
│   ├── ingestion/                  # parsers + normalization
│   ├── graph/                      # graph builder + forensic heuristics
│   ├── features/                   # feature engineering (incl. temporal)
│   ├── models/
│   │   ├── anomaly.py               # Isolation Forest / Autoencoder
│   │   ├── clustering.py            # Node2Vec + HDBSCAN
│   │   ├── classifier.py            # optional GCN/XGBoost
│   │   ├── evasion_detectors.py     # CoinJoin, peel-chain, mixer signatures
│   │   └── taint_propagation.py     # multi-hop risk propagation
│   ├── correlation/                 # network-blockchain correlation scoring
│   ├── explainability/              # SHAP wrapper + reason-string generator
│   ├── feedback/                    # analyst confirm/reject loop
│   └── scoring.py                   # composite risk score
├── dashboard/
│   ├── app.py                       # Streamlit dashboard
│   └── report_export.py             # PDF case report generator
├── notebooks/                       # exploratory analysis
├── docs/
│   └── technical_writeup.md         # detailed approach & methodology
├── tests/
├── requirements.txt
├── run.sh
└── README.md
```

---

## 🚀 Getting Started

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

---

## ▶️ Usage

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
python -m src.correlation.network_blockchain
python -m src.models.taint_propagation
python -m src.scoring
```

**Export a case report for a flagged wallet:**
```bash
python -m dashboard.report_export --wallet-id <wallet_address> --output reports/
```

---

## 📊 Dataset

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

**Injected patterns** (with hidden ground truth for internal evaluation):
- Ransomware collector pattern (many victim wallets → one collector → peel chain)
- Laundering fan-in/fan-out pattern
- Same-actor multi-wallet clusters sharing a broadcast IP
- CoinJoin-style mixing transactions
- Realistic noise: legitimate shared IPs (NAT/VPN), address reuse, malformed records

---

## 🔬 Methodology Deep Dive

1. **Ingestion & normalization** — unify CSV/JSON/XML into a common schema; enrich with offline GeoIP
2. **Graph construction** — build a wallet–transaction–IP graph; apply common-input-ownership and change-address heuristics
3. **Feature engineering** — behavioral, temporal, and network features per wallet/transaction
4. **AI/ML detection**
   - Isolation Forest / Autoencoder → anomaly score
   - Node2Vec embeddings + HDBSCAN → entity clusters
   - Optional GNN/XGBoost → supervised risk probability
5. **Evasion-specific detection** — CoinJoin, peel-chain, and mixer signature detectors
6. **Network-blockchain correlation** — confidence-calibrated IP/ASN-to-cluster linkage
7. **Taint propagation** — risk score flows through the graph from known/suspected-bad seeds, decaying with hop distance
8. **Composite risk scoring** — combine all signals into a single ranked, weighted score
9. **Explainability** — SHAP-backed, human-readable justification per alert
10. **Visualization & export** — dashboard, link-analysis graph, and PDF case reports
11. **Feedback loop** — analyst confirmation/rejection recalibrates future scoring

---

## 🔍 Explainability

Every flagged wallet/transaction includes:
- A numeric **risk score** (0–1)
- The **top contributing features** (via SHAP)
- An auto-generated **plain-English justification**
- Traceable **evidence links** (related transactions, wallets, and IPs)
- A **taint propagation path**, if risk was inherited from a known-bad wallet upstream

**Example alert:**
> Flagged due to high fan-in ratio (0.91), shared broadcast IP with 4 other wallets within 60 seconds, origin ASN flagged as high-risk hosting, and 2-hop taint inheritance from a known ransomware collector wallet (confidence: 0.78).

---

## 📈 Evaluation

Model performance is validated against self-injected synthetic ground truth:
- Precision / recall / F1 of flagged entities vs. known-injected criminal patterns
- Cluster purity against deliberately co-spent wallet groups
- **False-positive rate on pure "normal traffic" runs** — reported honestly, with mitigations documented
- Manual review of top-N alerts for explanation coherence
- Ablation checks: performance with/without network-layer correlation, to quantify its actual contribution

---

## 🌐 Existing Solutions & Prior Art

| Tool | Approach | Gap |
|---|---|---|
| **Chainalysis Reactor/KYT** | On-chain wallet clustering, entity attribution | Closed-source, cloud-hosted, no network-layer correlation |
| **Elliptic** | Labeled on-chain dataset + risk scoring | On-chain only |
| **GraphSense** (open-source) | Clustering + tagging | Heavy infra (full node + Spark), no network-layer fusion |
| **BlockSci** | Fast on-chain parsing | No ML/anomaly layer, no network correlation |
| **Academic P2P deanonymization research** (Koshy et al., 2014) | IP-to-address linkage via P2P sniffing | No integrated ML, clustering, or dashboard |

**Our gap-fill:** no existing public/open tool combines on-chain clustering, network-layer IP/timing correlation, modern explainable ML, and an investigator-ready dashboard in one **offline** package — which is precisely what this problem statement asks for.

---

## 🗺️ Roadmap

- [ ] Core ingestion pipeline (CSV/JSON/XML)
- [ ] Graph construction + forensic heuristics
- [ ] Feature engineering (behavioral + temporal)
- [ ] Isolation Forest anomaly detection
- [ ] Autoencoder anomaly detection
- [ ] Node2Vec + HDBSCAN clustering
- [ ] Network-blockchain correlation scoring
- [ ] Multi-hop taint propagation
- [ ] Evasion-specific detectors (CoinJoin, peel-chain, mixer signatures)
- [ ] SHAP explainability layer
- [ ] Composite risk scoring
- [ ] Streamlit dashboard + link-analysis graph
- [ ] PDF case report export
- [ ] Analyst feedback loop
- [ ] Optional GNN classifier
- [ ] False-positive stress testing
- [ ] Technical write-up
- [ ] Demo video / presentation

---

## 👥 Team

| Name | Role |
|---|---|
| Aditya | _Role_ |
| Rudra | _Role_ |
| Sweety | _Role_ |

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*Submitted for Smart India Hackathon — Problem Statement SIH26146, National Technical Research Organisation (NTRO).*

</div>
