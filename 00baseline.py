import re
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Energest Constants for Sky Motes
TICK_DURATION = 1.0 / 32768  # seconds per tick
VOLTAGE = 3.0  # volts

CURRENT_DRAW = {
    "cpu": 1.8,   # mA
    "lpm": 0.0545,
    "tx": 17.4,
    "rx": 18.8
}

# Colours
COLORS = {
    "baseline": "#27A6F5"  # blue
}
# Soft tints for stacked bars
COMPONENT_COLORS = {
    "cpu": "#27A6F5",  # baseline blue
    "lpm": "#5BBBF7",
    "tx":  "#89CEF8",
    "rx":  "#B5E1FA",
}

# Node labels & exclusions
NODE_LABELS = {
    1: "Sink",
    2: "Camera 1",
    3: "Thermostat",
    4: "Camera 2",
    5: "Motion Sensor 1",
    6: "Motion Sensor 2",
    7: "Lightbulb 1",
    8: "Lightbulb 2",
    9: "Smart Plug",
    10: "Attacker"
}
EXCLUDED_NODE_IDS = [10]  # exclude attacker

# Optional: save figures in ./figs
SAVE_FIGS = False
OUTPUT_DIR = "./figs"

#  Parsing & conversion
def parse_delta_energest_from_file(filename):
    pattern = re.compile(
        r'(?P<time>\d+:\d+\.\d+)\s+ID:(?P<node_id>\d+).*?Delta Energest - '
        r'CPU: (?P<cpu>\d+) LPM: (?P<lpm>\d+) TX: (?P<tx>\d+) RX: (?P<rx>\d+)',
        re.MULTILINE
    )
    with open(filename, "r") as f:
        log_text = f.read()

    records = []
    for match in pattern.finditer(log_text):
        data = match.groupdict()
        node_id = int(data["node_id"])
        if node_id in EXCLUDED_NODE_IDS:
            continue
        minutes, seconds = data["time"].split(":")
        total_seconds = int(minutes) * 60 + float(seconds)
        records.append({
            "time": total_seconds,
            "node_id": node_id,
            "cpu": int(data["cpu"]),
            "lpm": int(data["lpm"]),
            "tx": int(data["tx"]),
            "rx": int(data["rx"]),
        })
    return pd.DataFrame(records)

def convert_ticks_to_energy(df):
    if df.empty:
        return df.copy()
    energy_data = []
    for _, row in df.iterrows():
        node_energy = {"time": row["time"], "node_id": row["node_id"]}
        for mode in ["cpu", "lpm", "tx", "rx"]:
            time_sec = row[mode] * TICK_DURATION
            energy_mj = CURRENT_DRAW[mode] * VOLTAGE * time_sec  # mA*V*s = mJ
            node_energy[mode] = energy_mj
        energy_data.append(node_energy)
    return pd.DataFrame(energy_data)

#  Helpers
def discovered_node_ids(df):
    ids = sorted(df["node_id"].unique())
    return [nid for nid in ids if nid not in EXCLUDED_NODE_IDS]

def totals_by_node(df):
    """Return per-node totals for each component (cpu/lpm/tx/rx)."""
    return (df.groupby("node_id")[["cpu","lpm","tx","rx"]]
              .sum()
              .reset_index())

def device_names(node_ids):
    return [NODE_LABELS.get(nid, f"Node {nid}") for nid in node_ids]

def maybe_save(fig, filename):
    if not SAVE_FIGS:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")

# PLOTS (BASELINE ONLY)

def plot_baseline_components_grid(df_baseline):
    """
    2x2 grid, one subplot per component (CPU/LPM/TX/RX),
    bars = totals per node (baseline).
    """
    comp_list = ["cpu", "lpm", "tx", "rx"]
    node_ids = discovered_node_ids(df_baseline)
    names = device_names(node_ids)
    t = totals_by_node(df_baseline).set_index("node_id")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for i, comp in enumerate(comp_list):
        vals = [t.loc[nid, comp] if nid in t.index else 0.0 for nid in node_ids]
        axes[i].bar(np.arange(len(node_ids)), vals, color=COLORS["baseline"])
        axes[i].set_title(f"{comp.upper()} Energy per Node (mJ)")
        axes[i].set_xticks(np.arange(len(node_ids)))
        axes[i].set_xticklabels(names, rotation=45, ha='right')
        axes[i].set_ylabel("Energy (mJ)")
        axes[i].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)

    plt.suptitle("Baseline – Component Totals per Node (2×2)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    maybe_save(fig, "baseline_components_2x2.png")
    plt.show()

def plot_baseline_component_single(df_baseline, component):
    """
    Single bar chart: one component’s totals across nodes (baseline).
    component in {'cpu','lpm','tx','rx'}
    """
    node_ids = discovered_node_ids(df_baseline)
    names = device_names(node_ids)
    t = totals_by_node(df_baseline).set_index("node_id")
    vals = [t.loc[nid, component] if nid in t.index else 0.0 for nid in node_ids]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(node_ids)), vals, color=COLORS["baseline"])
    ax.set_title(f"Baseline {component.upper()} Energy per Node (mJ)")
    ax.set_xticks(np.arange(len(node_ids)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("Energy (mJ)")
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    maybe_save(fig, f"baseline_{component}_per_node.png")
    plt.show()

def plot_baseline_stacked_per_node(df_baseline):
    """
    Stacked bar per node (baseline): CPU+LPM+TX+RX
    """
    node_ids = discovered_node_ids(df_baseline)
    names = device_names(node_ids)
    t = totals_by_node(df_baseline).set_index("node_id")

    cpu = np.array([t.loc[nid, "cpu"] if nid in t.index else 0.0 for nid in node_ids])
    lpm = np.array([t.loc[nid, "lpm"] if nid in t.index else 0.0 for nid in node_ids])
    tx  = np.array([t.loc[nid, "tx"]  if nid in t.index else 0.0 for nid in node_ids])
    rx  = np.array([t.loc[nid, "rx"]  if nid in t.index else 0.0 for nid in node_ids])

    x = np.arange(len(node_ids))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, cpu, color=COMPONENT_COLORS["cpu"], label="CPU")
    ax.bar(x, lpm, bottom=cpu, color=COMPONENT_COLORS["lpm"], label="LPM")
    ax.bar(x, tx,  bottom=cpu+lpm, color=COMPONENT_COLORS["tx"], label="TX")
    ax.bar(x, rx,  bottom=cpu+lpm+tx, color=COMPONENT_COLORS["rx"], label="RX")

    ax.set_title("Baseline Stacked Energy per Node (mJ)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("Energy (mJ)")
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(ncol=4, frameon=False)
    plt.tight_layout()
    maybe_save(fig, "baseline_stacked_per_node.png")
    plt.show()

#  MAIN 
if __name__ == "__main__":
    # Adjust path as needed
    baseline_file = "C:/Logs/baseline_log.txt"

    raw_df_baseline = parse_delta_energest_from_file(baseline_file)
    df_baseline = convert_ticks_to_energy(raw_df_baseline)

    # 1) Baseline components 2x2
    plot_baseline_components_grid(df_baseline)

    # 2) Baseline per-component single charts
    plot_baseline_component_single(df_baseline, "cpu")
    plot_baseline_component_single(df_baseline, "lpm")
    plot_baseline_component_single(df_baseline, "rx")
    plot_baseline_component_single(df_baseline, "tx")

    # 3) Baseline stacked per node
    plot_baseline_stacked_per_node(df_baseline)
