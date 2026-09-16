# AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
### SIH26146 | NTRO (National Technical Research Organisation) | Blockchain & Cybersecurity

---

## 1. Problem Statement — Plain-English Breakdown

### 1.1 What is actually being asked

NTRO wants a system that:

1. Takes in **bulk Bitcoin transaction + network metadata** (CSV/JSON/XML files) — not live blockchain data, a static dataset.
2. **Joins two different worlds of data** that are normally analyzed separately:
   - **Blockchain layer** — wallet addresses, TXIDs, amounts, fees, script types (all public, permanent, on-chain).
   - **Network layer** — the IP address, port, and timestamp of the machine that first broadcast a transaction onto the Bitcoin P2P network (this data is NOT on the blockchain; it can only be captured by running/monitoring P2P nodes, i.e., surveillance infrastructure).
3. Uses **actual AI/ML models** (not just if-else rule engines) to:
   - Detect anomalies (suspicious transactions/wallets)
   - Cluster wallets that likely belong to the same real person/group
   - Produce a **ranked, explainable** list of investigative leads with confidence scores
4. Shows results on a **dashboard / link-analysis graph**.
5. Runs **fully offline** on Linux — no internet, no live blockchain node, no cloud APIs.

### 1.2 Why does this problem exist? (The real-world "why")

Bitcoin is often mistakenly called "anonymous." It's actually **pseudonymous**:

- Every transaction, wallet address, and amount is permanently recorded on a public ledger (the blockchain). Anyone can see that "address A sent 0.5 BTC to address B" — but not *who* A and B are in real life.
- Criminals exploit this gap between "publicly traceable money flow" and "untraceable real identity" for:
  - **Ransomware payments** (victim pays extortionist's wallet)
  - **Darknet market transactions** (drugs, weapons, stolen data)
  - **Extortion / blackmail payments**
  - **Money laundering** — moving dirty money through many wallets, mixers, and exchanges to obscure its origin before cashing out

The one weak point criminals can't fully hide: to actually **send** a Bitcoin transaction, their wallet software must broadcast it to the Bitcoin peer-to-peer network — and that broadcast comes from a real IP address. Intelligence agencies that operate "supernodes" (nodes connected to a large fraction of the P2P network) can often observe **which IP address first relayed a given transaction**, before it gets mixed in with everyone else's traffic. This is a well-documented deanonymization technique used in real blockchain forensics (similar to what Chainalysis, the FBI, and academic researchers like Dan Kaminsky and the "Bitcoin transaction graph analysis" research community have used).

**So the core intelligence problem is:** *Can we correlate "who broadcast this transaction, from where, and when" (network layer) with "whose wallets are transacting with whom, how much, how often" (blockchain layer) — and use AI to surface the transactions/wallets most likely tied to criminal activity, with a clear explanation of why?*

### 1.3 Why is this hard? (What makes it a real research/engineering problem, not a toy)

| Challenge | Why it's hard |
|---|---|
| **Scale** | Bitcoin processes hundreds of thousands of transactions daily; a real system must handle bulk data efficiently, not just a toy CSV of 100 rows. |
| **No ground truth labels** | You rarely know for certain which wallets are "criminal" — so this is mostly an **unsupervised/semi-supervised** problem, not simple classification. |
| **Address reuse & obfuscation** | Criminals use **mixers/tumblers**, **peel chains** (repeatedly sending slightly smaller amounts through a chain of new addresses), **CoinJoin** transactions (multiple people's coins combined to break traceability), and **new address per transaction** (standard Bitcoin privacy practice, which also happens to look "suspicious" even when innocent) — so naive rules produce huge false positive rates. |
| **Network-layer noise** | An IP might not correspond to the actual sender — it could be a relay node, a VPN exit node, Tor exit node, or another peer just forwarding the transaction. Correlating "first-seen IP" with "true origin" is itself probabilistic, not certain. |
| **Explainability requirement** | Agencies can't act on a black-box "0.87 suspicious" score — every flag must be justifiable to a human investigator/court, which is why explainable AI (XAI) is explicitly demanded. |
| **Offline constraint** | No calling live APIs (blockchain.info, Etherscan-style explorers, GeoIP web lookups) — everything, including the GeoIP database, must be bundled and run locally. |

### 1.4 What NTRO is really testing (hackathon judging lens)

- Can you build a proper **data pipeline** (ingest → clean → structure)?
- Can you model this as a **graph problem** correctly (wallets, transactions, IPs as a heterogeneous graph)?
- Do you understand **real blockchain-forensics heuristics** (common-input-ownership, peel chains, fan-in/fan-out)?
- Can you apply **actual ML** (not just hardcoded thresholds) — unsupervised anomaly detection, graph embeddings, clustering?
- Can you make results **explainable** (SHAP, rule traces, feature attribution)?
- Can you present it usably (**dashboard**) for a non-technical investigator?

---

## 2. Existing Solutions & Prior Art (Landscape Review)

It's important to know what already exists, both to avoid reinventing things and to position your solution's novelty.

### 2.1 Commercial blockchain forensics tools
- **Chainalysis Reactor / KYT** — industry-leading wallet clustering, entity attribution (exchanges, mixers, darknet markets), transaction graph visualization. Used by FBI, Europol, and most national agencies. **Closed-source, subscription-based, cloud-hosted.**
- **CipherTrace (Mastercard)** — similar wallet risk-scoring and AML compliance product.
- **Elliptic** — known for the public **Elliptic Dataset** (a labeled Bitcoin transaction graph dataset used heavily in academic ML research — licit/illicit/unknown labels on ~200k transactions). This is the closest public benchmark to what this hackathon wants.
- **TRM Labs** — similar risk intelligence platform.

**Gap:** All of these are blockchain-layer only (on-chain analysis). None of them (publicly) combine **network-layer IP/timing correlation** because that data requires running actual P2P surveillance infrastructure — which is exactly the NTRO-specific angle this problem statement wants, since NTRO is a signals/technical intelligence body, not just a financial compliance company.

### 2.2 Academic research
- **"Bitcoin transaction graph analysis"** research (e.g., Meiklejohn et al., 2013, *"A Fistful of Bitcoins"*) — established the common-input-ownership heuristic and change-address heuristics still used today.
- **Elliptic++ / Elliptic dataset ML papers** — apply GCN (Graph Convolutional Networks), GraphSAGE, Random Forest, XGBoost for illicit transaction classification on the Elliptic dataset; GNN-based approaches typically outperform flat feature-based classifiers because they exploit the transaction graph topology.
- **"Deanonymizing Bitcoin transactions via network-layer information"** (multiple academic papers, e.g., from University of Luxembourg / Qatar Computing Research Institute, and the well-known 2014 paper by Koshy, Koshy & McDaniel, *"An Analysis of Anonymity in Bitcoin Using P2P Network Traffic"*) — this is the direct academic ancestor of the network-blockchain correlation this problem statement is asking for. Their method: run a listening node, record which IP relays each TXID first, then statistically link addresses to IPs.
- **Autoencoder/Isolation Forest anomaly detection** papers applied to financial fraud and crypto transactions are common baselines in AML literature.

### 2.3 Open-source tools
- **GraphSense** (an open-source cryptocurrency analytics platform, EU-funded) — closest open-source equivalent to Chainalysis; does clustering and tagging but is heavy infrastructure (needs full node + Spark cluster) and again has no network-layer correlation.
- **BlockSci** — fast blockchain analysis framework (C++/Python) for on-chain parsing and heuristic clustering.
- **Bitcoin Core `-debug=net` logs / custom sniffer nodes** — the standard way researchers capture first-relay IP data; no polished product exists that fuses this with on-chain clustering + ML + dashboard in one offline toolkit.

### 2.4 What's genuinely missing (your opportunity/novelty)
No existing public/open tool does **all four** of these together in one offline package:
1. On-chain wallet/transaction clustering (heuristics + graph ML)
2. Network-layer IP/timing correlation
3. Modern ML-based anomaly detection with explainability (SHAP)
4. A ready-to-use investigator dashboard

That combination — specifically the network+blockchain fusion with XAI — **is the actual novelty NTRO is asking you to build**, even at prototype/synthetic-data scale.

---

## 3. What We Are Actually Solving (Restated Simply)

> **Given a synthetic-but-realistic dump of "who broadcast which Bitcoin transaction, from what IP, at what time" plus "which wallets sent how much to which wallets in that transaction" — automatically figure out which wallets/transactions look like they belong to a criminal money-flow pattern (ransomware, laundering, darknet payments), group wallets that are probably the same real actor, rank the most suspicious ones, and explain in plain language why each one was flagged — all in an offline dashboard.**

That's the whole problem in one sentence. Everything else is implementation detail.

---

## 4. Proposed Solution — Full System Design

### 4.1 High-Level Architecture

```
┌──────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│  1. INGESTION      │ →  │ 2. GRAPH BUILDER    │ →  │ 3. FEATURE ENGINE   │
│  CSV/JSON/XML      │     │ Wallet-TX-IP graph  │     │ per-node features   │
│  parser + cleaner  │     │ (NetworkX/Neo4j)    │     │ per-edge features    │
└──────────────────┘     └───────────────────┘     └────────┬───────────┘
                                                              ↓
┌──────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│ 6. DASHBOARD       │ ←  │ 5. EXPLAINABILITY   │ ←  │ 4. ML DETECTION     │
│ Streamlit/Dash +   │     │ SHAP + rule trace   │     │ Anomaly detection   │
│ link-analysis graph│     │ + composite score   │     │ + clustering + GNN  │
└──────────────────┘     └───────────────────┘     └────────────────────┘
```

Everything runs as a single offline pipeline: `raw files → SQLite/Parquet → graph → features → models → scored alerts → dashboard`.

---

### 4.2 Module 1 — Data Ingestion & Normalization

**Input formats supported:** CSV, JSON, XML (as specified).

**Minimum schema (as given in problem statement), normalized into one internal table:**

| Field | Type | Description |
|---|---|---|
| `timestamp` | datetime | When the transaction was observed/broadcast |
| `src_ip`, `dst_ip` | string | Network peer IPs involved in relay |
| `src_port`, `dst_port` | int | Ports used |
| `txid` | string | Unique transaction ID (blockchain layer) |
| `input_addresses[]` | list[string] | Wallets funding the transaction |
| `output_addresses[]` | list[string] | Wallets receiving funds |
| `input_amounts[]` | list[float] | BTC amount per input |
| `output_amounts[]` | list[float] | BTC amount per output |
| `fee` | float | Miner fee paid |
| `script_type` | string | e.g., P2PKH, P2SH, P2WPKH, multisig |
| `geo_country`, `asn` | string/int | Derived via offline GeoIP lookup on `src_ip` |

**Steps:**
1. Format-agnostic parser (`pandas.read_csv`, `json.load`, `xml.etree.ElementTree`) → unify into one DataFrame/SQLite table.
2. Data-quality checks: malformed TXIDs, negative/NaN amounts, duplicate rows, timestamp parsing/timezone normalization.
3. **Offline GeoIP enrichment** using a locally downloaded MaxMind **GeoLite2-Country/ASN** database (`.mmdb` file, queried via the `geoip2` Python library — no internet call at runtime).
4. Flag known high-risk ASN ranges (bulletproof hosting providers, common VPN/Tor exit ASNs) — a useful static reference list you bundle offline.
5. Store cleaned data in **SQLite** (simple, offline, zero server setup) or **Parquet** files (fast, columnar, good for pandas-scale analytics).

---

### 4.3 Module 2 — Entity/Transaction Graph Construction

This is the analytical backbone. Model everything as a **heterogeneous multigraph** with 3 node types:

- **Wallet nodes** (`W`)
- **Transaction nodes** (`T`, keyed by TXID)
- **IP nodes** (`I`)

**Edge types:**

| Edge | Meaning | Weight/Attributes |
|---|---|---|
| `Wallet → Transaction` (input) | Wallet spent from this tx | amount |
| `Transaction → Wallet` (output) | Wallet received from this tx | amount |
| `Transaction → IP` | This IP first relayed this tx | timestamp, port |
| `Wallet ↔ Wallet` (co-spend/derived) | Two wallets used as inputs to the same transaction | co-occurrence count |
| `IP ↔ Wallet cluster` (derived) | Statistical link between IP and a wallet cluster over repeated observations | correlation score |

**Key forensic heuristics applied to build derived edges (classic blockchain forensics, not ML — but essential ground truth signal for ML features):**

1. **Common-Input-Ownership Heuristic** — if two+ addresses appear as inputs in the *same* transaction, they were very likely signed by the same wallet software/owner (since a wallet must control the private keys for all inputs it spends). This is the #1 real-world clustering heuristic used by Chainalysis et al.
2. **Change-Address Heuristic** — in a typical transaction, one output is "change" returning to the sender; detectable by characteristics like: address never seen before, output script type matches input type, "round number" output going to the other party while change is a leftover odd amount.
3. **Peel-Chain Detection** — a sequence of transactions where a large amount is repeatedly split with a small amount peeled off and the remainder forwarded to a new address — a classic laundering pattern.
4. **Fan-in / Fan-out ratio** — many-inputs-to-one-output (fan-in, typical of consolidation/mixing) or one-input-to-many-outputs (fan-out, typical of distribution to money mules).

Implementation: `networkx` for prototype scale (thousands–low millions of nodes); `Neo4j` (Community Edition, run offline in Docker) if you want a queryable graph DB and Cypher-based pattern queries for the dashboard.

---

### 4.4 Module 3 — Feature Engineering

For every **wallet** and every **transaction**, compute a feature vector:

**Wallet-level features:**
- Total in-degree / out-degree (# of counterparty wallets)
- Total BTC volume in / out
- Number of transactions
- Average / variance of transaction amounts
- Time-of-day / day-of-week transaction pattern (entropy of activity times — bots/automated laundering often show flat/unnatural time distributions)
- Number of distinct IPs associated with this wallet's transactions
- Number of distinct countries/ASNs its transactions were broadcast from
- Fan-in and fan-out ratio
- Peel-chain participation flag/depth
- Address age (first-seen to last-seen span) and reuse frequency
- Cluster size (from common-input-ownership clustering)

**Transaction-level features:**
- Number of inputs / outputs
- Total value, fee, fee-per-byte
- Script type
- Time between broadcast and first block confirmation (if available)
- Whether broadcasting IP is on a flagged high-risk ASN/geo list
- Number of hops from a "known seed" suspicious wallet (if any labeled seeds exist)

**Network-layer features:**
- IP reuse across multiple wallets (single IP broadcasting for many different wallet clusters = possible mixing service or exchange hot wallet)
- Timing tightness: multiple transactions from different wallets broadcast within a very short window from the same IP/subnet
- Port anomalies (non-standard ports, indicative of non-standard node software — sometimes used by automated tools)

---

### 4.5 Module 4 — AI/ML Detection (the core "working model" requirement)

Use **three complementary techniques**, since the problem explicitly lists multiple AI/ML focus areas and wants more than one trick:

#### A. Unsupervised Anomaly Detection
- **Isolation Forest** — fast, interpretable-ish, works well on tabular wallet/transaction features; isolates anomalies as points that are easy to separate in few splits.
- **Autoencoder (neural net)** — trained on the bulk of "normal-looking" transactions; wallets/transactions with high **reconstruction error** are anomalous. Good for capturing non-linear combinations of features.
- Output: an **anomaly score** per wallet/transaction (0–1 normalized).

#### B. Entity Clustering (grouping wallets = same real actor)
- Start from the **deterministic common-input-ownership clusters** (ground truth-ish signal).
- Enrich with **graph embeddings**: run **Node2Vec** or **GraphSAGE** on the wallet-transaction-IP graph to learn dense vector representations that capture structural similarity (wallets that behave similarly in the graph end up close in embedding space, even without directly co-spending).
- Cluster embeddings with **HDBSCAN** (preferred over k-means/DBSCAN since it doesn't require specifying cluster count and naturally labels sparse/noise points, which matters a lot here since most wallets are legitimate one-off addresses).
- Output: a `cluster_id` per wallet + a "same-actor confidence" score based on embedding distance.

#### C. Supervised / Semi-Supervised Classification (optional, if you self-label synthetic "known-bad" seeds)
- If the synthetic dataset includes any ground-truth labels (or you inject a few known "ransomware wallet" seeds into your generator), train a **Random Forest** or **XGBoost classifier**, or better, a **Graph Neural Network (GraphSAGE/GCN)** directly on the graph — this mirrors the approach used on the public **Elliptic dataset**, where GNN-based models outperform flat classifiers by a good margin because they exploit multi-hop graph context (a wallet two hops from a known-bad wallet is riskier than a random wallet).
- Output: a probability of "illicit" per wallet/transaction.

#### D. Network-Blockchain Correlation Score
- A custom scoring function: `correlation_score = f(IP-wallet co-occurrence frequency, timing tightness, ASN risk, geo risk)`.
- This directly operationalizes the "correlate network-layer with blockchain-layer" requirement from the problem statement and is the most NTRO-specific/novel part of the solution — most public tools skip this entirely.

#### Composite Risk Score
Combine all four signals into one final score per wallet/transaction, e.g.:

```
risk_score = w1 * anomaly_score
           + w2 * cluster_risk (cluster containing any known-bad seed, or cluster density anomaly)
           + w3 * classifier_probability (if labels available)
           + w4 * network_correlation_score
```

Weights (`w1..w4`) can be tuned manually for the demo, or learned via a simple logistic regression meta-model if you have any labels at all.

---

### 4.6 Module 5 — Explainability (XAI)

Non-negotiable per the problem statement ("explainable investigative leads").

- **SHAP (SHapley Additive exPlanations)** applied to the Isolation Forest / Random Forest / XGBoost models → for every flagged wallet/transaction, output the **top contributing features** (e.g., "fan-in ratio = 0.91 [+0.32 to risk], shared IP with 4 other wallets within 60s [+0.28], ASN flagged as bulletproof hosting [+0.15]").
- Convert SHAP output into a **plain-English justification string** auto-generated per alert — this is what an investigator actually reads, not raw SHAP plots.
- For clustering, explain cluster membership via the **specific heuristic** that grouped the wallets (e.g., "grouped via common-input-ownership in TXID `abc123`" or "grouped via graph-embedding similarity, cosine distance 0.04").
- Keep an **audit trail**: every alert should be traceable back to the exact rows/evidence that produced it (critical for anything that might feed an actual investigation).

---

### 4.7 Module 6 — Ranked Alert Generation

Output a table, sortable/filterable, containing:

| wallet/tx_id | risk_score | top_reasons[] | cluster_id | related_ips[] | related_wallets[] | first_seen | last_seen |
|---|---|---|---|---|---|---|---|

Export as CSV/JSON so it can be handed off to other analyst tooling too.

---

### 4.8 Module 7 — Dashboard / Link-Analysis Visualization

Build with **Streamlit** or **Dash** (pure Python, runs fully offline, no external hosting needed):

**Screens:**
1. **Overview** — summary stats (# wallets, # transactions, # flagged, score distribution histogram).
2. **Ranked Alerts Table** — sortable, filterable by risk score/date/country, click-through to detail view.
3. **Alert Detail View** — SHAP explanation, transaction history timeline for that wallet, list of related wallets/IPs.
4. **Link-Analysis Graph** — interactive network graph (via `pyvis` or `plotly.graph_objects` network layout, or `streamlit-agraph`) showing wallets/transactions/IPs as nodes, with flagged entities highlighted in red, sized by risk score, and edges labeled with amount/timestamp. Investigators can click a node to pivot and explore neighbors.
5. **Geo/ASN view** — map or table showing which countries/ASNs are contributing the most flagged traffic (useful since GeoIP enrichment is explicitly required).

---

## 5. Full Tech Stack Summary

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| Data ingestion | pandas, `json`, `xml.etree.ElementTree` |
| Storage | SQLite / Parquet |
| GeoIP | MaxMind GeoLite2 (`.mmdb`, offline) + `geoip2` library |
| Graph | NetworkX (prototype) or Neo4j Community (offline Docker, optional) |
| Graph embeddings | Node2Vec / GraphSAGE (PyTorch Geometric or DGL) |
| Anomaly detection | scikit-learn `IsolationForest`, custom PyTorch Autoencoder |
| Clustering | HDBSCAN, scikit-learn |
| Supervised model (optional) | XGBoost / Random Forest / GCN (PyTorch Geometric) |
| Explainability | SHAP |
| Dashboard | Streamlit (fastest to build) or Dash |
| Graph visualization | `pyvis`, `plotly`, or `streamlit-agraph` |
| Packaging | `requirements.txt` + a single `run.sh` for fully offline Linux setup |

---

## 6. Synthetic Dataset Generation (if you build your own instead of/alongside the provided one)

Since real data won't be given, build a **generator script** that simulates realistic patterns:

1. **Normal traffic**: random wallets transacting with random counterparties, realistic amount distributions (log-normal), normal geographic/IP spread, one broadcasting IP per wallet-owner (simulating an individual's node/wallet software).
2. **Injected criminal patterns** (label these secretly for your own evaluation, even if not scored publicly):
   - **Ransomware pattern**: many distinct "victim" wallets all sending to one central collector wallet within a tight time window, collector then fans out through a peel chain.
   - **Mixing/laundering pattern**: many-in/many-out transactions with obfuscated amounts, high fan-in then high fan-out through intermediate "layering" wallets.
   - **Same-actor multi-wallet pattern**: several wallets, all broadcasting exclusively from the same 1–2 IPs, with common-input co-spends between some of them.
   - **Darknet-market-like pattern**: many small, similar-value payments converging to a small set of vendor wallets, from geographically diverse IPs (buyers) at irregular but frequent intervals.
3. Add realistic noise: shared IPs due to legitimate NAT/VPN use (false-positive stress test), address reuse for legitimate reasons, occasional missing/malformed fields to test ingestion robustness.

This makes your model evaluation meaningful (you can compute precision/recall against your own injected ground truth even without official labels) — and is a strong thing to show judges since it proves your detection actually works rather than just running without crashing.

---

## 7. Evaluation & Validation Strategy

Since there's likely no official labeled test set:
- Use **self-injected synthetic ground truth** (as above) to compute precision/recall/F1 of your flagged list against known-injected criminal wallets.
- Sanity-check clustering against your own common-input-ownership ground truth (wallets you deliberately made co-spend should end up in the same cluster).
- Manually inspect top-20 highest-risk alerts and confirm the SHAP explanations make logical sense (no nonsensical justifications).
- Stress-test with pure "normal" traffic only to check the **false positive rate** stays low — this matters a lot in real deployment, since flooding investigators with false leads is as bad as missing real ones.

---

## 8. Deliverables Checklist (mapped to problem statement's "Expected Solution")

- [x] Fully offline, Linux-runnable prototype
- [x] Ingestion supporting CSV/JSON/XML
- [x] Entity/transaction graph (wallets–transactions–IPs)
- [x] At least one **working ML model** (anomaly detection minimum; clustering + optional GNN classifier for depth)
- [x] Ranked, explainable alert list with confidence scores (SHAP-backed)
- [x] Dashboard with link-analysis visualization
- [x] Short technical write-up (architecture, model choice justification, explainability method)
- [x] Code repository, cleanly structured, with `README.md` and setup instructions

---

## 9. Suggested Repository Structure

```
bitcoin-traffic-analyzer/
├── data/
│   ├── raw/                  # input CSV/JSON/XML dumps
│   ├── geoip/                # offline GeoLite2 .mmdb file
│   └── synthetic_generator.py
├── src/
│   ├── ingestion/             # parsers + normalization
│   ├── graph/                 # graph builder + heuristics (common-input, peel-chain)
│   ├── features/               # feature engineering
│   ├── models/
│   │   ├── anomaly.py          # Isolation Forest / Autoencoder
│   │   ├── clustering.py       # Node2Vec/GraphSAGE + HDBSCAN
│   │   └── classifier.py       # optional GCN/XGBoost
│   ├── explainability/         # SHAP wrapper + reason-string generator
│   └── scoring.py              # composite risk score
├── dashboard/
│   └── app.py                  # Streamlit app
├── notebooks/                  # exploratory analysis
├── docs/
│   └── technical_writeup.md
├── requirements.txt
├── run.sh
└── README.md
```

---

## 10. Key Talking Points for Your Presentation/Demo (Why Judges Should Care)

1. **You fuse two data layers no public tool combines** (network + blockchain) — this is the specific NTRO angle, not generic crypto-AML.
2. **You use real forensic heuristics** (common-input-ownership, peel-chain) as ground-truth signal feeding the ML — not naive black-box ML on raw fields.
3. **Multiple complementary AI techniques**, not one model doing everything — anomaly detection + graph embeddings + clustering (+ optional GNN classifier).
4. **Explainability is built-in**, not bolted on — every single alert has a SHAP-backed, human-readable justification, which is what makes this usable by actual investigators, not just a research demo.
5. **Fully offline** — no dependency on live APIs, meeting the sensitive/air-gapped operational reality of an intelligence agency.
6. **Realistic synthetic dataset with injected ground truth** lets you actually *prove* precision/recall numbers in your demo instead of hand-waving.

---

*Document prepared as a detailed technical reference for SIH26146 (NTRO) — AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic.*
