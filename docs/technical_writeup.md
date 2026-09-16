# Technical Write-Up
## Bitcoin Traffic Analyzer — AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

**Problem Statement ID:** SIH26146
**Organisation:** National Technical Research Organisation (NTRO)
**Theme:** Blockchain & Cybersecurity

---

## 1. Problem Understanding

### 1.1 Statement

Bitcoin's pseudonymous, peer-to-peer design allows illicit actors — ransomware operators, darknet-market vendors, extortionists, and money launderers — to move and cash out funds while evading traditional financial surveillance. Wallet addresses are not directly tied to real-world identities, and transactions propagate through a decentralized P2P network rather than a single monitorable choke point.

The objective is to design and build a complete **offline** system that:
1. Ingests bulk Bitcoin transaction/network metadata (CSV/JSON/XML)
2. Correlates network-layer observations (IP/port/timing) with blockchain-layer data (wallet/TXID/amount)
3. Applies AI/ML to detect anomalies, cluster entities, and generate prioritized, explainable investigative leads
4. Presents findings via a dashboard/link-analysis visualization

### 1.2 The Core Insight

Bitcoin is pseudonymous, not anonymous. Two data layers exist independently:

| Layer | What it reveals | Limitation |
|---|---|---|
| **Blockchain layer** | Wallet addresses, TXIDs, amounts, fees, script types — permanent and public | No identity information; addresses are opaque codes |
| **Network layer** | The IP address that first relays/broadcasts a transaction to the P2P network | Only observable by nodes actively listening on the P2P network (i.e., surveillance infrastructure); noisy — the observed IP may be a relay, VPN exit, or Tor node rather than the true origin |

Neither layer alone is sufficient for attribution. The system's central technical contribution is **fusing these two layers** to produce stronger, more actionable, and better-explained leads than either layer could produce independently.

### 1.3 Why This Is Non-Trivial

- **No reliable ground truth.** Real investigations rarely have confirmed "criminal" labels at scale, making this primarily an **unsupervised/semi-supervised** problem rather than standard classification.
- **Deliberate evasion.** Techniques like CoinJoin (coin mixing across multiple parties), peel chains (repeated small-amount splitting), and one-time addresses (a *normal* Bitcoin privacy practice) all produce statistically unusual patterns — meaning naive anomaly detection produces high false-positive rates unless evasion patterns are explicitly modeled.
- **Noisy network signal.** An IP relaying a transaction is not proof of ownership; it could be an intermediate peer, VPN, or Tor exit node. Correlation must be treated probabilistically, not deterministically.
- **Explainability is mandatory.** A risk score with no justification is not actionable for an investigator or admissible as an investigative lead — every flag must trace back to specific, human-readable evidence.
- **Offline-only constraint.** No live blockchain nodes, no cloud APIs, no internet dependency at runtime — consistent with the operational reality of a sensitive intelligence workflow.

---

## 2. Related Work / Existing Solutions

| System | Approach | Gap Relative to This Problem |
|---|---|---|
| **Chainalysis Reactor / KYT** | On-chain wallet clustering, entity attribution (exchanges, mixers, darknet markets), transaction graph visualization | Closed-source, cloud-hosted, subscription-based; no network-layer (IP/timing) correlation |
| **Elliptic** | Labeled on-chain transaction dataset (~200k transactions, licit/illicit/unknown) widely used in academic ML research | On-chain only; no network-layer fusion; primarily a benchmark dataset, not a deployable tool |
| **CipherTrace / TRM Labs** | Wallet risk-scoring, AML compliance | Commercial, closed, on-chain focused |
| **GraphSense** (open-source, EU-funded) | Clustering + tagging on full blockchain data | Requires full node + Spark cluster; heavy infrastructure; no network-layer correlation |
| **BlockSci** | Fast C++/Python blockchain parsing and heuristic clustering | Parsing/analysis framework only; no ML anomaly layer, no dashboard, no network correlation |
| **Academic P2P deanonymization research** (Koshy, Koshy & McDaniel, 2014, *"An Analysis of Anonymity in Bitcoin Using P2P Network Traffic"*; Meiklejohn et al., 2013, *"A Fistful of Bitcoins"*) | Establishes the common-input-ownership heuristic and IP-to-address linkage via P2P sniffing | Academic methodology only; no integrated ML pipeline, clustering, explainability, or dashboard |

**Identified gap:** No existing public or open-source tool combines (a) on-chain wallet/transaction clustering, (b) network-layer IP/timing correlation, (c) modern explainable ML (anomaly detection + graph embeddings + XAI), and (d) an investigator-ready dashboard, in a single **offline** package. This combination is the specific contribution of this project.

---

## 3. Proposed Approach

### 3.1 System Architecture

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

The pipeline is a single, sequential offline process: raw files → normalized store → graph → features → models → scored, explained alerts → dashboard.

### 3.2 Data Ingestion & Normalization

**Minimum schema** (as specified in the problem statement):

```
timestamp, src_ip, dst_ip, src_port, dst_port, txid,
input_addresses[], output_addresses[], input_amounts[], output_amounts[],
fee, script_type, geo_country, asn
```

- Format-agnostic parsers (CSV via `pandas`, JSON via `json`, XML via `xml.etree.ElementTree`) normalize all inputs into one internal schema, stored in SQLite/Parquet.
- Data-quality checks: malformed TXIDs, negative/NaN amounts, duplicate rows, timezone-normalized timestamps.
- **Offline GeoIP enrichment**: `src_ip` is resolved to country/ASN via a locally downloaded MaxMind GeoLite2 `.mmdb` database — no runtime network calls.
- A static, bundled reference list flags known high-risk ASN ranges (bulletproof hosting, common VPN/Tor exit ranges) for use as a feature later.

### 3.3 Graph Construction

The core analytical structure is a **heterogeneous multigraph** with three node types — `wallet`, `transaction`, `ip` — and the following edges:

| Edge | Semantics | Attributes |
|---|---|---|
| `wallet → transaction` | Wallet spent from this transaction | amount |
| `transaction → wallet` | Wallet received from this transaction | amount |
| `transaction → ip` | This IP first relayed this transaction | timestamp, port |
| `wallet ↔ wallet` (derived) | Co-spend: both addresses appeared as inputs to the same transaction | co-occurrence count |

**Deterministic forensic heuristics** (established techniques, not ML, but critical ground-truth-like signal feeding the ML layer):

1. **Common-Input-Ownership Heuristic** — if multiple addresses are inputs to the same transaction, the spender must have controlled all their private keys, implying common ownership. This is the primary real-world clustering heuristic (used by Chainalysis and established in Meiklejohn et al., 2013).
2. **Change-Address Heuristic** — identifies the "change" output returning to the sender (e.g., a previously unseen address, matching script type, non-round amount).
3. **Peel-Chain Detection** — flags sequences where a large balance is repeatedly split, with a small amount peeled off at each step and the remainder forwarded onward — a classic layering/laundering pattern.
4. **Fan-in / Fan-out Ratio** — many-inputs-to-one-output (consolidation, often mixing-related) or one-input-to-many-outputs (distribution, often mule-related).

Implementation: `networkx` for prototype-scale graphs; optionally `Neo4j` (Community Edition, run offline) for larger graphs and Cypher-based pattern queries powering the dashboard.

### 3.4 Feature Engineering

**Per-wallet features:**
- In-degree / out-degree (distinct counterparties)
- Total volume in/out, transaction count, amount mean/variance
- Time-of-day / day-of-week activity entropy (automated laundering often shows unnaturally flat time distributions)
- Number of distinct IPs and ASNs/countries associated with the wallet's transactions
- Fan-in / fan-out ratio
- Peel-chain participation flag and depth
- Address age (first-seen to last-seen span), reuse frequency
- Common-input-ownership cluster size

**Per-transaction features:**
- Input/output counts, total value, fee, fee-per-byte
- Script type
- Broadcasting IP's ASN/geo risk flag

**Network-layer features:**
- IP reuse across multiple distinct wallet clusters (a single IP serving many clusters suggests a mixing service or exchange hot wallet)
- Timing tightness: multiple distinct wallets' transactions broadcast from the same IP/subnet within a short window
- Port anomalies (non-standard ports, indicative of non-standard node software)

**Temporal features (behavior over time, not just a static snapshot):**
- Activity-burst detection (sudden spikes after dormancy — typical of a cash-out event)
- Rolling-window transaction rate changes

### 3.5 AI/ML Detection

Multiple complementary techniques are used, since no single model captures every relevant signal:

#### A. Unsupervised Anomaly Detection
- **Isolation Forest** — isolates anomalies via random recursive partitioning; anomalous points require fewer splits to isolate than points embedded in dense "normal" regions. Fast, works well on tabular wallet/transaction features, and provides a natural anomaly score without needing labels.
- **Autoencoder** (neural network) — trained to reconstruct "normal" wallet/transaction feature vectors; wallets with high **reconstruction error** are flagged as anomalous, capturing non-linear feature interactions that Isolation Forest may miss.
- Both produce independent anomaly scores (0–1); agreement between the two is treated as a stronger signal than either alone.

#### B. Entity Clustering
- Starts from deterministic **common-input-ownership** clusters (highest-confidence signal).
- Enriched via **Node2Vec** graph embeddings, which learn dense vector representations capturing structural similarity in the wallet-transaction-IP graph — wallets with similar graph roles/behavior end up close in embedding space even without direct co-spending.
- Embeddings are clustered with **HDBSCAN**, chosen over k-means/DBSCAN because it doesn't require a predefined cluster count and naturally separates sparse "noise" points — important since most wallets in real data are one-off, legitimate addresses.

#### C. Evasion-Specific Detectors
Rather than relying solely on generic anomaly detection, dedicated detectors target known, documented laundering techniques:
- **CoinJoin-style transaction detection** — identifies transactions with many equal-valued outputs from many distinct inputs (the structural fingerprint of coin-mixing protocols).
- **Peel-chain detection** — traces sequences of transactions matching the repeated-split pattern described in Section 3.3.
- **Mixer/tumbler behavioral fingerprinting** — flags wallets whose input/output timing and amount distributions match known mixing-service patterns.

#### D. Multi-Hop Taint Propagation
Modeled on the "poison"/"haircut" propagation technique used in real forensic tools: if a wallet is a known or high-confidence-flagged bad actor, a decaying risk score propagates outward through the transaction graph to wallets it transacts with, at a rate that decreases with hop distance. This allows the system to surface *associates* of known-bad wallets, not just directly anomalous wallets in isolation.

#### E. Network-Blockchain Correlation Scoring
A confidence-calibrated score quantifying how strongly an IP/ASN pattern ties to a specific wallet cluster:
- Weighted by the number of **independent observations** linking an IP to a cluster (a single coincidence is weak evidence; repeated co-occurrence across many transactions over time is strong evidence).
- **Discounted** for IPs on known VPN/Tor/hosting-provider ASN ranges, since these produce spurious apparent correlations.

#### F. Optional Supervised Classifier
If labeled seeds are available (either from the provided dataset or self-injected synthetic labels), a **Random Forest**, **XGBoost**, or **Graph Neural Network** (GraphSAGE/GCN) can be trained. GNN-based approaches are informed by published results on the Elliptic dataset, where graph-aware models outperform flat feature-based classifiers by exploiting multi-hop context (a wallet near a known-bad wallet is riskier than an isolated random wallet).

### 3.6 Composite Risk Scoring

All signals are combined into one final ranked score per wallet/transaction:

```
risk_score = w1 · anomaly_score
           + w2 · cluster_risk (density anomaly / known-bad seed membership)
           + w3 · evasion_detector_flags
           + w4 · taint_propagation_score
           + w5 · network_correlation_score
           + w6 · classifier_probability (if labels available)
```

Weights are manually tuned for the prototype and can be replaced with a learned meta-model (e.g., logistic regression) if any labeled data becomes available.

---

## 4. Explainability Method

Explainability is a first-class requirement, not an add-on, per the problem statement's demand for "explainable investigative leads."

- **SHAP (SHapley Additive exPlanations)** is applied to the tree-based models (Isolation Forest, optional Random Forest/XGBoost) to compute per-feature contribution to each wallet's risk score.
- SHAP output is converted into an **auto-generated, plain-English justification string** for every alert — e.g.:
  > *"Flagged due to high fan-in ratio (0.91), shared broadcast IP with 4 other wallets within 60 seconds, origin ASN flagged as high-risk hosting, and 2-hop taint inheritance from a known ransomware collector wallet (confidence: 0.78)."*
- Cluster membership is explained via the **specific heuristic** responsible (e.g., "grouped via common-input-ownership in TXID `abc123`" or "grouped via graph-embedding similarity, cosine distance 0.04").
- Every alert maintains a **full audit trail** back to the exact source rows/evidence that produced it, supporting downstream investigative or evidentiary use.

This design choice deliberately favors **interpretable, traceable evidence over marginally higher black-box accuracy** — an investigator needs to know *why*, not just *how confident*.

---

## 5. Dataset

Real seized/intercepted Bitcoin surveillance data is sensitive and cannot be distributed for a hackathon; a **synthetic dataset** modeled on realistic Bitcoin P2P/transaction fields is used instead (either the organizer-provided dataset or a custom generator).

**Synthetic generation strategy:**
- **Normal traffic**: random wallets and counterparties, log-normal amount distributions, realistic geographic/IP spread, one broadcasting IP per wallet-owner.
- **Injected criminal patterns** (with hidden ground-truth labels retained for internal evaluation only):
  - Ransomware collector pattern: many distinct victim wallets converge on one collector wallet within a tight time window, followed by a peel chain.
  - Laundering fan-in/fan-out pattern through intermediate layering wallets.
  - Same-actor multi-wallet clusters broadcasting exclusively from one or two shared IPs, with common-input co-spends between some wallets.
  - Darknet-market-like pattern: many small, similar-value payments converging on a small set of vendor wallets from geographically diverse buyer IPs.
  - CoinJoin-style mixing transactions.
- **Realistic noise**: legitimate shared IPs (NAT/VPN), legitimate address reuse, occasional malformed/missing fields to stress-test ingestion robustness.

---

## 6. Evaluation Strategy

In the absence of an official labeled test set, evaluation relies on **self-injected synthetic ground truth**:

- **Precision / recall / F1** of the ranked alert list against known-injected criminal wallets.
- **Cluster purity** — wallets deliberately made to co-spend in the generator should end up in the same detected cluster.
- **False-positive rate on pure normal traffic** — measured explicitly and reported honestly, including known confounds (e.g., legitimate exchanges naturally resembling laundering hubs by raw fan-in/fan-out) and the mitigations applied (e.g., exchange-pattern whitelisting).
- **Ablation testing** — pipeline performance with and without the network-layer correlation module, to quantify its actual marginal contribution rather than assuming it helps.
- **Manual review** of the top-N highest-risk alerts to confirm SHAP-generated justifications are logically coherent and not spurious.

---

## 7. Model Choice Justification (Summary)

| Choice | Rationale |
|---|---|
| Isolation Forest over deep anomaly models for baseline | Fast, no labels needed, interpretable enough for tabular wallet features, strong baseline before adding complexity |
| Autoencoder as a second anomaly signal | Captures non-linear feature interactions Isolation Forest may miss; agreement between the two models is a stronger signal than either alone |
| Node2Vec + HDBSCAN over k-means | No need to predefine cluster count; naturally handles noise/outlier wallets, which dominate real-world data |
| SHAP over LIME or other XAI methods | Theoretically grounded (Shapley values), consistent attribution, well-supported for tree-based models used here |
| NetworkX (with optional Neo4j) | Sufficient for prototype scale; Neo4j path available if scale/query needs grow |
| Streamlit over Flask/Django custom UI | Fastest path to an interactive, offline-capable dashboard within hackathon time constraints |

---

## 8. Limitations & Honest Caveats

- Network-layer IP-to-wallet correlation is inherently **probabilistic**, not proof — VPNs, Tor, and NAT introduce genuine ambiguity that no amount of modeling fully resolves; the system reports calibrated confidence, not certainty.
- Without real labeled data, precision/recall figures are measured against **self-injected** synthetic ground truth, which may not perfectly represent real-world criminal behavior diversity.
- Legitimate high-volume entities (exchanges, payment processors) can structurally resemble laundering hubs; false-positive mitigation (e.g., whitelisting) is necessary and explicitly tested, but not perfect.
- The system is a **decision-support tool for human investigators**, not an automated accusation engine — every output is designed to be reviewed, not acted on unilaterally.

---

## 9. Deliverables Summary

- Offline, Linux-runnable prototype (ingestion → graph → ML → explainability → dashboard)
- Multi-format ingestion (CSV/JSON/XML)
- Wallet–transaction–IP entity graph with forensic heuristics
- Working AI/ML models: anomaly detection, clustering, taint propagation, evasion-specific detectors, network-blockchain correlation, optional supervised classifier
- Ranked, SHAP-explained alert list with confidence scores
- Interactive Streamlit dashboard with link-analysis graph visualization
- PDF case export and analyst feedback loop
- This technical write-up

---

*Prepared for Smart India Hackathon — Problem Statement SIH26146, National Technical Research Organisation (NTRO).*