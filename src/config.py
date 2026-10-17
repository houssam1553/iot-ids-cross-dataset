"""Central configuration: dataset files, label columns and the features we exclude.

Everything that a reviewer might want to question (which columns are dropped and
why) lives in this one file.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# short name -> tokens that must all appear (lower-case) in the CSV file name.
# Works with the official UQ downloads and with the common Kaggle mirrors.
DATASETS = {
    "ton_iot": ["ton", "iot", "v3"],
    "bot_iot": ["bot", "iot", "v3"],
    "unsw_nb15": ["unsw", "v3"],
}
# Not IoT, but same format; enable with --datasets ... cicids2018
OPTIONAL_DATASETS = {
    "cicids2018": ["cic", "2018", "v3"],
}

# Per-class counts published on https://staff.itee.uq.edu.au/marius/NIDS_datasets/
# (NetFlow v3). Used only to WARN if a mirror differs from the official release.
# Note: for NF-BoT-IoT-v3 the page's stated total (16,993,808) does not equal the
# sum of its own class table (16,933,844); we compare per class.
EXPECTED_COUNTS = {
    "ton_iot": {
        "Benign": 16_792_214, "Backdoor": 203_384, "DoS": 203_456, "DDoS": 4_141_256,
        "Injection": 381_777, "MITM": 6_013, "Password": 1_594_777,
        "Ransomware": 3_971, "Scanning": 1_358_977, "XSS": 2_834_435,
    },
    "bot_iot": {
        "Benign": 51_989, "Reconnaissance": 1_695_132, "DDoS": 7_150_882,
        "DoS": 8_034_190, "Theft": 1_651,
    },
    "unsw_nb15": {
        "Benign": 2_237_731, "Fuzzers": 33_816, "Analysis": 2_381, "Backdoor": 1_226,
        "DoS": 5_980, "Exploits": 42_748, "Generic": 19_651, "Reconnaissance": 17_074,
        "Shellcode": 4_659, "Worms": 158,
    },
}

LABEL_COL = "Label"      # 0 = benign, 1 = attack
ATTACK_COL = "Attack"    # attack family name ("Benign" for normal traffic)
TIME_COL = "FLOW_START_MILLISECONDS"

# ---------------------------------------------------------------------------
# Columns that are NEVER model inputs
# ---------------------------------------------------------------------------
# Identify hosts/services of one particular testbed: a model that uses them
# learns "which machine was the attacker", not "what an attack looks like".
IDENTIFIER_COLS = [
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
    "L4_SRC_PORT", "L4_DST_PORT",
    "DNS_QUERY_ID",            # random transaction id, carries no behaviour
]
# Absolute time: attacks were launched in scheduled windows, so the timestamp
# alone predicts the label inside one dataset and is meaningless across datasets.
TIME_COLS = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]
LABEL_COLS = [LABEL_COL, ATTACK_COL, "Dataset"]

# Optional ablation (--drop-ttl): TTL values often encode the testbed's OS mix.
TTL_COLS = ["MIN_TTL", "MAX_TTL"]

# Expected NetFlow v3 schema (53 features), from Luay et al. 2025 and
# Sarhan et al. 2022. Only used for a warn-only schema report.
V3_FEATURES = [
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT",
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]


def excluded_columns(drop_ttl: bool = False) -> set:
    cols = set(IDENTIFIER_COLS) | set(TIME_COLS) | set(LABEL_COLS)
    if drop_ttl:
        cols |= set(TTL_COLS)
    return cols
