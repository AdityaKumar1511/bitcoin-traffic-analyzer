# 🎬 Demo Script: "A Day in the Life of a Cyber Forensics Investigator"
### National Technical Research Organisation (NTRO) · Smart India Hackathon (SIH26146)
**Track 3 — Dashboard, UX & Investigative Walkthrough**

---

## 🎯 Objective for Judges
Demonstrate how an intelligence officer or cybercrime investigator uses **Bitcoin Traffic Analyzer (BitForge)** to ingest raw, unlabelled Bitcoin transaction traffic, fuse network broadcast metadata with on-chain ledgers, uncover an active ransomware money laundering syndicate, trace multi-hop fund leakage, and export a court-admissible forensic case dossier — **100% offline with zero cloud dependency**.

---

## ⏱️ Scene-by-Scene Walkthrough (5-Minute Pitch)

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│ Scene 1: Intake │ ──> │ Scene 2: Triage  │ ──> │ Scene 3: Unmask   │
│ Raw Ingestion   │     │ Critical Alerts  │     │ Peel Chain/Mixing │
└─────────────────┘     └──────────────────┘     └─────────┬─────────┘
                                                           │
┌─────────────────┐     ┌──────────────────┐               │
│ Scene 5: Export │ <── │ Scene 4: Trace   │ <─────────────┘
│ Court Dossier   │     │ Multi-Hop Taint  │
└─────────────────┘     └──────────────────┘
```

---

### **Scene 1: The Briefing & Ingestion (0:00 – 1:00)**
> **Investigator Persona:** *"We just received intercepted P2P broadcast logs containing 2,000 transactions and over 7,000 wallet interactions related to an active ransomware outbreak. Due to national security constraints, we cannot query public block explorers or external APIs."*

1. **Open the BitForge Dashboard** at `http://localhost:8501`.
2. **Highlight the Top KPI Bar**:
   - **Ingested Transactions:** `2,000`
   - **Analyzed Wallets:** `7,072` (automatically resolved into `5,814` real-world entity clusters)
   - **Critical Alerts:** `2` flagged for immediate tactical intervention
   - **Evasion Signatures:** `34` CoinJoin mixers and active peel chains unmasked.
3. **Key Talking Point:** Show that ingestion normalizes CSV, JSON, and XML schemas and enriches IP addresses with offline MaxMind GeoLite2 databases.

---

### **Scene 2: High-Priority Triage & Explainable AI (1:00 – 2:00)**
> **Investigator Persona:** *"In a real command center, investigators are overwhelmed with thousands of addresses. Let's look at how our composite scoring engine surfaces the highest-conviction leads."*

1. Navigate to **Tab 1: 🚨 Investigative Alerts**.
2. Filter the table to **CRITICAL** risk levels (`Score >= 0.75`).
3. Select the top flagged wallet: `1VD8tWrX2ZiyUtSrXETGnzEbAizy` (Risk Score: `0.9608`).
4. **Inspect the Forensic Justification Card**:
   - Point out the **plain-English explanation**:
     > *"Flagged due to active carrier in ransomware peel-chain forwarding, high fan-in aggregation ratio (0.88), and persistent broadcast from relay IP `192.247.206.45`."*
   - Show the 4 component metrics: **Anomaly Score (0.82)**, **Evasion Score (0.90)**, **Network Correlation (0.75)**, **Taint Score (1.00)**.
5. **Key Talking Point:** Point out that this is not a black-box number — SHAP feature attributions explain exactly *why* the model flagged the entity.

---

### **Scene 3: Unmasking Evasion & The Laundering Hub (2:00 – 3:00)**
> **Investigator Persona:** *"Criminals don't send money directly to an exchange; they peel transactions or use CoinJoin mixers. Let's inspect the evasion signatures."*

1. Click on **Tab 3: 🔍 Evasion & Laundering Hub**.
2. **Examine CoinJoin Mixes**:
   - Show the 34 detected mixing transactions where multiple input signers receive equal standardized denominations (e.g. `0.1000 BTC`).
3. **Examine Ransomware Peel Chains**:
   - Highlight the identified peel chain with starting amount `2.32 BTC`, length `5 hops`, showing how change outputs are repeatedly peeled while keeping funds moving.
4. **Network Infrastructure Distribution**:
   - Review the ASN category bar chart showing bulletproof hosting vs. standard ISPs.
5. **Key Talking Point:** Unlike generic graph metrics, our detectors specifically model real-world AML evasion tradecraft.

---

### **Scene 4: Interactive Neural Link-Analysis Graph (3:00 – 4:00)**
> **Investigator Persona:** *"Now let's trace where the stolen funds flowed and who is co-spending with our target."*

1. Switch to **Tab 2: 🕸️ Link Analysis Graph**.
2. Select target wallet `1VD8tWrX2ZiyUtSrXETGnzEbAizy`.
3. Adjust the **Exploration Radius** slider to `2 hops`.
4. **Interactive Demonstration on Canvas**:
   - **Target Hub (Glowing Emerald Hexagon):** Represents the primary target address.
   - **Blue Nodes (Wallets) & Yellow Diamonds (Transactions):** Demonstrates flow of BTC.
   - **Purple Hexagons (Broadcasting IPs):** Shows network co-location linking separate transactions to the same physical relay.
   - **Pink Dashed Links:** Demonstrates common-input ownership co-spending entities.
5. **Key Talking Point:** Show how multi-hop taint decay traced contaminated funds from seed wallet `1VD8tWrX...` down to downstream recipients `bc1s6s5ygm...` and `1wk5XJ...`.

---

### **Scene 5: Feedback Triage & Court Dossier Export (4:00 – 5:00)**
> **Investigator Persona:** *"To conclude our investigation, we certify our findings and generate an evidentiary dossier ready for legal submission."*

1. Return to the alert detail panel and demonstrate the **Analyst Feedback Loop**:
   - Mark status as `CONFIRMED`.
   - Add notes: *"Confirmed ransomware collector wallet linked to PE Ivanov Vitaliy bulletproof infrastructure."*
   - Click **Save & Recalibrate** — show how the system updates risk calibration in real-time.
2. Navigate to **Tab 4: 📑 Case Dossier Export**.
3. Select the subject wallet and click **Generate Dossier**.
4. **Showcase the Outputs**:
   - **Download Official Case Report (.pdf):** Professional ReportLab PDF with executive tables, taint paths, and signature attestation block.
   - **Download Markdown (.md):** Portable, structured text record for agency records.
5. **Closing Statement:**
   > *"Bitcoin Traffic Analyzer empowers agencies to move from noisy network sniffing to explainable, court-ready forensic attribution — fast, explainable, and 100% offline."*

---

## 🏆 Key Evaluation Cheat-Sheet for Jury Questions

| Jury Question | Winning Answer / Demonstration |
|---|---|
| **"How do you handle VPNs and Tor exit nodes?"** | Our network correlation engine explicitly discounts confidence for known VPN/Tor ASNs (e.g. 0.5x–0.65x multiplier), while up-weighting persistent bulletproof ASNs and temporal bursts. |
| **"Why not just use Chainalysis?"** | Chainalysis is cloud-hosted, proprietary, expensive, and blockchain-only. Our system is 100% offline, open-source, and correlates the P2P broadcast network layer with on-chain flows. |
| **"How do you prevent false positives on normal exchange wallets?"** | We combine multi-signal composite scoring, time-windowed pass-through heuristics, and an analyst feedback loop that suppresses confirmed false positives. |
| **"Is this scalable to millions of transactions?"** | All core feature extraction and graph algorithms operate on localized k-hop ego-nets and sparse NetworkX adjacency structures with sub-second per-query lookups. |
