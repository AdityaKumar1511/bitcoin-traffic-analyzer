"""
Synthetic Bitcoin Transaction Dataset Generator for Fraud Detection.

This module generates a realistic synthetic Bitcoin transaction dataset combining
normal traffic with injected criminal activity patterns for forensic graph analysis,
clustering, and fraud-detection machine learning models.

Supported Criminal Patterns:
1. Pattern A - Ransomware Collector + Peel Chain:
   Multiple victim wallets pay ransom into a single collector wallet within a tight
   timeframe (e.g., <6 hours). The collector then peels off funds across a sequential
   chain of intermediate wallets, retaining the same source IP.
2. Pattern B - Same-Actor Multi-Wallet Cluster:
   A group of 5-8 wallets sharing 1-2 source IPs that frequently co-spend inputs
   in the same transaction (common-input-ownership heuristic).
3. Pattern C - CoinJoin-Style Mixing:
   High-fan-in and high-fan-out transactions (8-15 inputs and 8-15 outputs) where
   outputs share identical/standardized BTC amounts (e.g., 0.1 BTC) to obscure lineage.

Output Files:
- synthetic_transactions.csv: Main transaction dataset adhering to a strict schema.
- ground_truth.csv: Hidden entity labels (wallet/txid) for model benchmarking.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple
import uuid

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants & Helper Functions
# ---------------------------------------------------------------------------

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BECH32_ALPHABET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
STANDARD_SCRIPT_TYPES = ["P2PKH", "P2SH", "P2WPKH", "P2WSH"]
UNUSUAL_SCRIPT_TYPES = ["MULTISIG", "NONSTANDARD", "P2TR", "P2PKH+P2SH"]


def generate_wallet_address(prefix: str | None = None) -> str:
    """Generate a realistic Bitcoin address with '1' or 'bc1' prefix."""
    if prefix is None:
        prefix = random.choice(["1", "bc1"])

    if prefix == "1":
        length = random.randint(26, 34)
        return "1" + "".join(random.choices(BASE58_ALPHABET, k=length - 1))
    elif prefix == "bc1":
        length = random.randint(26, 35)
        return "bc1" + "".join(random.choices(BECH32_ALPHABET, k=length - 3))
    else:
        length = random.randint(26, 34)
        return prefix + "".join(random.choices(BASE58_ALPHABET, k=max(1, length - len(prefix))))


def generate_txid() -> str:
    """Generate a 64-character hex transaction ID."""
    token = f"{uuid.uuid4().hex}-{random.random()}-{uuid.uuid4().hex}".encode("utf-8")
    txid = hashlib.sha256(token).hexdigest()
    assert len(txid) == 64, f"TXID must be exactly 64 hex characters, got {len(txid)}"
    return txid


def generate_ipv4() -> str:
    """Generate a realistic, public routable IPv4 address string."""
    while True:
        first = random.randint(1, 223)
        if first in (10, 127):
            continue
        second = random.randint(0, 255)
        if first == 172 and 16 <= second <= 31:
            continue
        if first == 192 and second == 168:
            continue
        if first == 169 and second == 254:
            continue
        third = random.randint(0, 255)
        fourth = random.randint(1, 254)
        return f"{first}.{second}.{third}.{fourth}"


def generate_port(is_dst: bool = False) -> int:
    """
    Generate port number.

    Bitcoin standard P2P port is 8333. Most destination ports are 8333,
    while source ports frequently use high ephemeral ports (1024-65535).
    """
    if is_dst:
        # Standard daemon listening port
        return 8333 if random.random() < 0.85 else random.randint(1024, 65535)
    else:
        # Source/client connections
        return 8333 if random.random() < 0.55 else random.randint(1024, 65535)


def format_timestamp(dt: datetime) -> str:
    """Format datetime into ISO 8601 string in UTC."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def split_amount(total: float, parts: int) -> List[float]:
    """Split a total amount into `parts` positive floats that sum exactly to total."""
    if parts == 1:
        return [round(total, 8)]

    # Generate random Dirichlet weights
    weights = np.random.dirichlet(np.ones(parts))
    amounts = [round(w * total, 8) for w in weights]

    # Adjust rounding discrepancy to match total exactly
    diff = round(total - sum(amounts), 8)
    amounts[0] = round(amounts[0] + diff, 8)
    # Ensure all amounts are positive
    for i in range(len(amounts)):
        if amounts[i] <= 0:
            amounts[i] = 0.00001
    return amounts


# ---------------------------------------------------------------------------
# Normal Transaction Generator
# ---------------------------------------------------------------------------

def generate_normal_transactions(
    count: int,
    start_dt: datetime,
    end_dt: datetime,
    shared_vpn_ips: List[str],
) -> List[Dict[str, Any]]:
    """
    Generate standard normal Bitcoin transactions.

    Characteristics:
    - 1-3 inputs and 1-3 outputs
    - Amounts drawn from log-normal distribution (mean 0.1-2 BTC)
    - Realistic noise: shared VPN/NAT IPs, rare missing fees, unusual script combos
    """
    transactions = []
    total_seconds = int((end_dt - start_dt).total_seconds())

    for _ in range(count):
        # Timestamp
        offset = random.randint(0, max(1, total_seconds))
        ts = start_dt + timedelta(seconds=offset)

        # Source IP: occasionally pick from shared NAT/VPN pool to simulate false positives
        if random.random() < 0.08 and shared_vpn_ips:
            src_ip = random.choice(shared_vpn_ips)
        else:
            src_ip = generate_ipv4()

        dst_ip = generate_ipv4()
        src_port = generate_port(is_dst=False)
        dst_port = generate_port(is_dst=True)
        txid = generate_txid()

        # Input & output addresses
        num_inputs = random.choices([1, 2, 3], weights=[0.65, 0.25, 0.10])[0]
        num_outputs = random.choices([1, 2, 3], weights=[0.30, 0.60, 0.10])[0]

        in_addrs = [generate_wallet_address() for _ in range(num_inputs)]
        out_addrs = [generate_wallet_address() for _ in range(num_outputs)]

        # Amount & fee
        base_amt = float(np.random.lognormal(mean=-0.5, sigma=0.85))
        base_amt = max(0.001, min(base_amt, 45.0))

        fee_val = round(random.uniform(0.00001, 0.0005), 8)

        # Realistic noise: missing/empty fee in ~1% of rows
        has_missing_fee = random.random() < 0.012

        total_in = round(base_amt + fee_val, 8)
        total_out = round(base_amt, 8)

        in_amts = split_amount(total_in, num_inputs)
        out_amts = split_amount(total_out, num_outputs)

        # Script type with ~1.5% unusual script combos
        if random.random() < 0.015:
            script_type = random.choice(UNUSUAL_SCRIPT_TYPES)
        else:
            script_type = random.choice(STANDARD_SCRIPT_TYPES)

        transactions.append({
            "timestamp": format_timestamp(ts),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "txid": txid,
            "input_addresses": ";".join(in_addrs),
            "output_addresses": ";".join(out_addrs),
            "input_amounts": ";".join(f"{a:.8f}" for a in in_amts),
            "output_amounts": ";".join(f"{a:.8f}" for a in out_amts),
            "fee": "" if has_missing_fee else f"{fee_val:.8f}",
            "script_type": script_type,
            "_pattern": "normal",
        })

    return transactions


# ---------------------------------------------------------------------------
# Criminal Pattern Generators
# ---------------------------------------------------------------------------

def generate_pattern_a_ransomware(
    target_count: int,
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate Pattern A: Ransomware Collector + Peel Chain.

    Characteristics:
    - 8-15 victim wallets pay ransom (similar amount) to one collector wallet
      within a tight timestamp window (< 6 hours).
    - Peel chain: Collector transfers remaining balance through 3-5 peel hops,
      peeling off a smaller amount each time.
    - Collector and peel chain transactions share the same source IP.
    """
    transactions: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []
    total_seconds = int((end_dt - start_dt).total_seconds())

    while len(transactions) < target_count:
        campaign_start_offset = random.randint(0, max(1, total_seconds - 86400 * 3))
        campaign_start = start_dt + timedelta(seconds=campaign_start_offset)

        actor_ip = generate_ipv4()
        collector_wallet = generate_wallet_address()
        ground_truth.append({
            "entity_id": collector_wallet,
            "entity_type": "wallet",
            "is_criminal": True,
            "pattern_type": "ransomware_peelchain",
        })

        # 8-15 victim wallets paying ransom
        num_victims = random.randint(8, 15)
        ransom_base = round(random.uniform(0.75, 2.5), 4)

        collected_balance = 0.0

        for i in range(num_victims):
            if len(transactions) >= target_count:
                break

            victim_wallet = generate_wallet_address()
            # Spread victim transactions within a 6-hour window
            tx_time = campaign_start + timedelta(seconds=random.uniform(0, 6 * 3600))
            victim_amt = round(ransom_base + random.uniform(-0.05, 0.05), 8)
            fee_val = round(random.uniform(0.00002, 0.00015), 8)
            total_in = round(victim_amt + fee_val, 8)

            txid = generate_txid()
            collected_balance += victim_amt

            transactions.append({
                "timestamp": format_timestamp(tx_time),
                "src_ip": generate_ipv4(),  # Victims have separate source IPs
                "dst_ip": actor_ip,
                "src_port": generate_port(is_dst=False),
                "dst_port": 8333,
                "txid": txid,
                "input_addresses": victim_wallet,
                "output_addresses": collector_wallet,
                "input_amounts": f"{total_in:.8f}",
                "output_amounts": f"{victim_amt:.8f}",
                "fee": f"{fee_val:.8f}",
                "script_type": random.choice(STANDARD_SCRIPT_TYPES),
                "_pattern": "ransomware_peelchain",
            })

            ground_truth.append({
                "entity_id": victim_wallet,
                "entity_type": "wallet",
                "is_criminal": True,
                "pattern_type": "ransomware_peelchain",
            })
            ground_truth.append({
                "entity_id": txid,
                "entity_type": "txid",
                "is_criminal": True,
                "pattern_type": "ransomware_peelchain",
            })

        # Peel chain: sequential hops after collection window
        peel_time = campaign_start + timedelta(hours=random.uniform(6.5, 9.0))
        current_wallet = collector_wallet
        current_balance = round(collected_balance, 8)

        num_peels = random.randint(3, 5)
        for _ in range(num_peels):
            if len(transactions) >= target_count or current_balance <= 0.2:
                break

            next_peel_wallet = generate_wallet_address()
            peeled_cashout_wallet = generate_wallet_address()

            fee_val = round(random.uniform(0.00003, 0.0002), 8)
            peel_amt = round(random.uniform(0.05, 0.25), 8)
            pass_forward_amt = round(current_balance - peel_amt - fee_val, 8)

            if pass_forward_amt <= 0.05:
                break

            txid = generate_txid()
            peel_time += timedelta(minutes=random.randint(15, 60))

            transactions.append({
                "timestamp": format_timestamp(peel_time),
                "src_ip": actor_ip,  # Consistent actor IP across peel steps
                "dst_ip": generate_ipv4(),
                "src_port": generate_port(is_dst=False),
                "dst_port": 8333,
                "txid": txid,
                "input_addresses": current_wallet,
                "output_addresses": f"{next_peel_wallet};{peeled_cashout_wallet}",
                "input_amounts": f"{current_balance:.8f}",
                "output_amounts": f"{pass_forward_amt:.8f};{peel_amt:.8f}",
                "fee": f"{fee_val:.8f}",
                "script_type": random.choice(STANDARD_SCRIPT_TYPES),
                "_pattern": "ransomware_peelchain",
            })

            ground_truth.append({
                "entity_id": next_peel_wallet,
                "entity_type": "wallet",
                "is_criminal": True,
                "pattern_type": "ransomware_peelchain",
            })
            ground_truth.append({
                "entity_id": peeled_cashout_wallet,
                "entity_type": "wallet",
                "is_criminal": True,
                "pattern_type": "ransomware_peelchain",
            })
            ground_truth.append({
                "entity_id": txid,
                "entity_type": "txid",
                "is_criminal": True,
                "pattern_type": "ransomware_peelchain",
            })

            current_wallet = next_peel_wallet
            current_balance = pass_forward_amt

    return transactions[:target_count], ground_truth


def generate_pattern_b_cluster(
    target_count: int,
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate Pattern B: Same-Actor Multi-Wallet Cluster.

    Characteristics:
    - 5-8 wallets all broadcasting from the SAME 1-2 source IPs.
    - Multiple transactions where 2+ cluster wallets appear as joint inputs
      (common-input-ownership heuristic).
    """
    transactions = []
    ground_truth = []
    total_seconds = int((end_dt - start_dt).total_seconds())

    # Create actor entity cluster
    actor_ips = [generate_ipv4(), generate_ipv4()]
    cluster_size = random.randint(5, 8)
    cluster_wallets = [generate_wallet_address() for _ in range(cluster_size)]

    for wallet in cluster_wallets:
        ground_truth.append({
            "entity_id": wallet,
            "entity_type": "wallet",
            "is_criminal": True,
            "pattern_type": "same_actor_cluster",
        })

    for i in range(target_count):
        offset = random.randint(0, max(1, total_seconds))
        tx_time = start_dt + timedelta(seconds=offset)
        tx_src_ip = random.choice(actor_ips)
        tx_dst_ip = generate_ipv4()
        txid = generate_txid()

        # Co-spending transactions (2+ cluster wallets co-signing inputs)
        # Ensure frequent co-spending for strong common-input heuristic
        is_co_spend = (i % 3 == 0) or (random.random() < 0.45)

        if is_co_spend:
            k_inputs = random.randint(2, min(4, len(cluster_wallets)))
            in_addrs = random.sample(cluster_wallets, k=k_inputs)
        else:
            in_addrs = [random.choice(cluster_wallets)]

        out_count = random.choice([1, 2])
        # Sometimes outputs transfer back to cluster wallet or external counterparty
        out_addrs = []
        for _ in range(out_count):
            if random.random() < 0.3:
                out_addrs.append(random.choice(cluster_wallets))
            else:
                out_addrs.append(generate_wallet_address())

        # Amounts
        fee_val = round(random.uniform(0.00002, 0.00025), 8)
        base_amt = round(random.uniform(0.2, 4.0), 8)
        total_in = round(base_amt + fee_val, 8)
        total_out = round(base_amt, 8)

        in_amts = split_amount(total_in, len(in_addrs))
        out_amts = split_amount(total_out, len(out_addrs))

        transactions.append({
            "timestamp": format_timestamp(tx_time),
            "src_ip": tx_src_ip,
            "dst_ip": tx_dst_ip,
            "src_port": generate_port(is_dst=False),
            "dst_port": 8333,
            "txid": txid,
            "input_addresses": ";".join(in_addrs),
            "output_addresses": ";".join(out_addrs),
            "input_amounts": ";".join(f"{a:.8f}" for a in in_amts),
            "output_amounts": ";".join(f"{a:.8f}" for a in out_amts),
            "fee": f"{fee_val:.8f}",
            "script_type": random.choice(STANDARD_SCRIPT_TYPES),
            "_pattern": "same_actor_cluster",
        })

        ground_truth.append({
            "entity_id": txid,
            "entity_type": "txid",
            "is_criminal": True,
            "pattern_type": "same_actor_cluster",
        })

    return transactions, ground_truth


def generate_pattern_c_coinjoin(
    target_count: int,
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate Pattern C: CoinJoin-Style Mixing Transactions.

    Characteristics:
    - Transactions with 8-15 distinct input addresses and 8-15 distinct output addresses.
    - Most output amounts are equal/standardized values (e.g. 0.1 BTC, 0.05 BTC),
      providing the classic CoinJoin structural fingerprint.
    """
    transactions = []
    ground_truth = []
    total_seconds = int((end_dt - start_dt).total_seconds())

    # Standard CoinJoin denominations
    denominations = [0.1, 0.05, 0.25, 0.5]

    for _ in range(target_count):
        offset = random.randint(0, max(1, total_seconds))
        tx_time = start_dt + timedelta(seconds=offset)
        denom = random.choice(denominations)

        n_participants = random.randint(8, 15)
        in_addrs = [generate_wallet_address() for _ in range(n_participants)]
        out_addrs = [generate_wallet_address() for _ in range(n_participants)]

        txid = generate_txid()
        fee_per_user = round(random.uniform(0.00002, 0.00008), 8)
        total_fee = round(fee_per_user * n_participants, 8)

        # In a CoinJoin, each participant provides slightly more than denom to cover fee
        in_amts = [round(denom + fee_per_user, 8) for _ in range(n_participants)]
        # Suspiciously identical outputs equal to denom
        out_amts = [round(denom, 8) for _ in range(n_participants)]

        src_ip = generate_ipv4()
        dst_ip = generate_ipv4()

        transactions.append({
            "timestamp": format_timestamp(tx_time),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": generate_port(is_dst=False),
            "dst_port": 8333,
            "txid": txid,
            "input_addresses": ";".join(in_addrs),
            "output_addresses": ";".join(out_addrs),
            "input_amounts": ";".join(f"{a:.8f}" for a in in_amts),
            "output_amounts": ";".join(f"{a:.8f}" for a in out_amts),
            "fee": f"{total_fee:.8f}",
            "script_type": random.choice(["P2WPKH", "P2SH", "P2WSH"]),
            "_pattern": "coinjoin_mixing",
        })

        # Register ground truth
        ground_truth.append({
            "entity_id": txid,
            "entity_type": "txid",
            "is_criminal": True,
            "pattern_type": "coinjoin_mixing",
        })
        for addr in in_addrs + out_addrs:
            ground_truth.append({
                "entity_id": addr,
                "entity_type": "wallet",
                "is_criminal": True,
                "pattern_type": "coinjoin_mixing",
            })

    return transactions, ground_truth


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------------------------

def generate_dataset(
    num_transactions: int = 2000,
    output_path: Path = Path("data/raw/synthetic_transactions.csv"),
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate dataset generation, file export, and ground truth creation.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Reference time window: past 30 days up to a fixed anchor for reproducibility
    end_dt = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=30)

    # Target counts: ~5% criminal total, roughly split evenly across 3 patterns
    target_criminal = max(3, int(round(num_transactions * 0.05)))
    count_a = target_criminal // 3
    count_b = target_criminal // 3
    count_c = target_criminal - (count_a + count_b)
    count_normal = num_transactions - (count_a + count_b + count_c)

    # Pre-generate 3 shared VPN/NAT IPs for realistic noise
    shared_vpn_ips = [generate_ipv4() for _ in range(3)]

    # Generate patterns
    normal_txs = generate_normal_transactions(
        count=count_normal,
        start_dt=start_dt,
        end_dt=end_dt,
        shared_vpn_ips=shared_vpn_ips,
    )

    pattern_a_txs, gt_a = generate_pattern_a_ransomware(
        target_count=count_a,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    pattern_b_txs, gt_b = generate_pattern_b_cluster(
        target_count=count_b,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    pattern_c_txs, gt_c = generate_pattern_c_coinjoin(
        target_count=count_c,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    # Combine all transactions and shuffle to interleave naturally
    all_transactions = normal_txs + pattern_a_txs + pattern_b_txs + pattern_c_txs
    random.shuffle(all_transactions)

    # Enforce exact main dataset schema
    main_columns = [
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "txid",
        "input_addresses",
        "output_addresses",
        "input_amounts",
        "output_amounts",
        "fee",
        "script_type",
    ]

    df_main = pd.DataFrame(all_transactions)[main_columns]
    assert (df_main["txid"].str.len() == 64).all(), "All main dataset txids must be exactly 64 hex characters"

    # Combine ground truth entities and remove duplicates
    all_gt = gt_a + gt_b + gt_c
    df_gt = pd.DataFrame(all_gt).drop_duplicates(subset=["entity_id"])

    # Ensure output directories exist
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path = output_path.parent / "ground_truth.csv"

    # Export CSVs
    df_main.to_csv(output_path, index=False)
    df_gt.to_csv(gt_path, index=False)

    return df_main, df_gt


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Bitcoin transaction dataset with realistic criminal patterns."
    )
    parser.add_argument(
        "--num-transactions",
        type=int,
        default=2000,
        help="Total number of transactions to generate (default: 2000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/synthetic_transactions.csv"),
        help="Output CSV filepath for transactions (default: data/raw/synthetic_transactions.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[*] Starting synthetic Bitcoin transaction generation...")
    print(f"    - Target transactions: {args.num_transactions}")
    print(f"    - Random seed:         {args.seed}")
    print(f"    - Output path:         {args.output}")

    df_main, df_gt = generate_dataset(
        num_transactions=args.num_transactions,
        output_path=args.output,
        seed=args.seed,
    )

    output_path = args.output.resolve()
    gt_path = output_path.parent / "ground_truth.csv"

    # Pattern counts
    num_criminal = len(df_gt[df_gt["entity_type"] == "txid"])
    criminal_by_pattern = df_gt[df_gt["entity_type"] == "txid"]["pattern_type"].value_counts().to_dict()
    normal_count = len(df_main) - num_criminal

    print("\n" + "=" * 55)
    print("           DATASET GENERATION SUMMARY")
    print("=" * 55)
    print(f"Total Transactions Generated: {len(df_main):,}")
    print(f"  - Normal Transactions:      {normal_count:,}")
    print(f"  - Injected Criminal TXs:    {num_criminal:,}")
    for pattern, count in criminal_by_pattern.items():
        print(f"    * {pattern}: {count}")
    print("-" * 55)
    print(f"Ground Truth Unique Entities: {len(df_gt):,}")
    entity_counts = df_gt["entity_type"].value_counts().to_dict()
    for etype, count in entity_counts.items():
        print(f"  - {etype}s: {count}")
    print("-" * 55)
    print("Files Saved:")
    print(f"  1. Main Dataset: {output_path}")
    print(f"  2. Ground Truth: {gt_path}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
