import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Energest Constants for Sky Motes
TICK_DURATION = 1.0 / 32768  # seconds per tick
VOLTAGE = 3.0  # volts

CURRENT_DRAW = {
    "cpu": 1.8,  # mA
    "lpm": 0.0545,
    "tx": 17.4,
    "rx": 18.8
}

# Plot colours
COLORS = {
    "baseline": "#27A6F5",  # blue
    "attack": "#CC2121"     # red
}

# Node ID to Device Name Mapping
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


# Parse raw delta ticks from log
def parse_delta_energest_from_file(filename):
    pattern = re.compile(
        r'(?P<time>\d+:\d+\.\d+)\s+ID:(?P<node_id>\d+).*?Delta Energest - '
        r'CPU: (?P<cpu>\d+) LPM: (?P<lpm>\d+) TX: (?P<tx>\d+) RX: (?P<rx>\d+)', re.MULTILINE
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


# Convert ticks to energy (mJ)
def convert_ticks_to_energy(df):
    energy_data = []
    for _, row in df.iterrows():
        node_energy = {
            "time": row["time"],
            "node_id": row["node_id"]
        }
        for mode in ["cpu", "lpm", "tx", "rx"]:
            time_sec = row[mode] * TICK_DURATION
            energy_mj = (CURRENT_DRAW[mode] * VOLTAGE * time_sec)
            node_energy[mode] = energy_mj
        energy_data.append(node_energy)
    return pd.DataFrame(energy_data)


# Plot line graphs over time
def plot_comparison(df1, label1, df2, label2, node_id):
    if node_id in EXCLUDED_NODE_IDS:
        return
    metrics = ['cpu', 'lpm', 'tx', 'rx']
    df1_node = df1[df1["node_id"] == node_id]
    df2_node = df2[df2["node_id"] == node_id]
    device_label = NODE_LABELS.get(node_id, f"Node {node_id}")

    for metric in metrics:
        plt.figure(figsize=(8, 4))
        plt.plot(df1_node["time"], df1_node[metric],
                 label=label1, marker='.', markersize=4, linewidth=1.2, color=COLORS["baseline"])
        plt.plot(df2_node["time"], df2_node[metric],
                 label=label2, marker=',', markersize=4, linewidth=1.2, color=COLORS["attack"])

        plt.title(f"{metric.upper()} Energy over Time – {device_label}")
        plt.xlabel("Time (seconds)")
        plt.ylabel(f"{metric.upper()} Energy (mJ)")
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Plot total energy bar chart for a single node
def plot_total_energy_bar(df1, label1, df2, label2, node_id):
    if node_id in EXCLUDED_NODE_IDS:
        return
    metrics = ['cpu', 'lpm', 'tx', 'rx']
    totals1 = df1[df1["node_id"] == node_id][metrics].sum()
    totals2 = df2[df2["node_id"] == node_id][metrics].sum()
    device_label = NODE_LABELS.get(node_id, f"Node {node_id}")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(metrics))
    ax.bar([i - 0.18 for i in x], totals1.values, width=0.36, label=label1, color=COLORS["baseline"])
    ax.bar([i + 0.18 for i in x], totals2.values, width=0.36, label=label2, color=COLORS["attack"])
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_title(f"Total Energy Consumption – {device_label}")
    ax.set_ylabel("Energy (mJ)")
    ax.set_xlabel("Component")
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()


# Plot total energy comparison for all nodes
def plot_total_energy_comparison(df1, label1, df2, label2):
    metrics = ['cpu', 'lpm', 'tx', 'rx']
    all_node_ids = sorted(set(df1["node_id"].unique()) | set(df2["node_id"].unique()))
    all_node_ids = [nid for nid in all_node_ids if nid not in EXCLUDED_NODE_IDS]

    device_names = []
    total_energy_baseline = []
    total_energy_attack = []

    for node_id in all_node_ids:
        device_label = NODE_LABELS.get(node_id, f"Node {node_id}")
        device_names.append(device_label)

        energy1 = df1[df1["node_id"] == node_id][metrics].sum().sum()
        energy2 = df2[df2["node_id"] == node_id][metrics].sum().sum()

        total_energy_baseline.append(energy1)
        total_energy_attack.append(energy2)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(device_names))
    ax.bar([i - 0.18 for i in x], total_energy_baseline, width=0.36, label=label1, color=COLORS["baseline"])
    ax.bar([i + 0.18 for i in x], total_energy_attack, width=0.36, label=label2, color=COLORS["attack"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(device_names, rotation=45, ha='right')
    ax.set_title("Total Energy Consumption per Device")
    ax.set_ylabel("Energy (mJ)")
    ax.set_xlabel("Device")
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()


# Plot percentage increase bar chart
def plot_percentage_increase_chart(comparison_df):
    metrics = ['CPU', 'LPM', 'TX', 'RX']
    devices = comparison_df["Device"]
    data = {f"% Increase {metric}": comparison_df[f"% Increase {metric}"] for metric in metrics}
    df_pct = pd.DataFrame(data)
    df_pct.index = devices

    df_pct.plot(kind="bar", figsize=(10, 5))
    plt.title("Percentage Increase in Energy Usage (Attack vs Baseline)")
    plt.ylabel("% Increase")
    plt.xlabel("Device")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()


# Generate energy comparison table
def generate_energy_comparison_table(df1, label1, df2, label2):
    metrics = ['cpu', 'lpm', 'tx', 'rx']
    all_node_ids = sorted(set(df1["node_id"].unique()) | set(df2["node_id"].unique()))
    all_node_ids = [nid for nid in all_node_ids if nid not in EXCLUDED_NODE_IDS]

    rows = []
    for node_id in all_node_ids:
        totals1 = df1[df1["node_id"] == node_id][metrics].sum()
        totals2 = df2[df2["node_id"] == node_id][metrics].sum()

        row = {
            "Node ID": node_id,
            "Device": NODE_LABELS.get(node_id, "Unknown")
        }

        for metric in metrics:
            base_val = totals1[metric]
            attack_val = totals2[metric]
            pct_change = ((attack_val - base_val) / base_val * 100) if base_val > 0 else 0

            row[f"{label1} {metric.upper()}"] = round(base_val, 2)
            row[f"{label2} {metric.upper()}"] = round(attack_val, 2)
            row[f"% Increase {metric.upper()}"] = round(pct_change, 2)

        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    return comparison_df

# Plot side-by-side bar charts for each energy component
def plot_component_bar_charts(df1, label1, df2, label2):
    components = ['cpu', 'lpm', 'tx', 'rx']
    all_node_ids = sorted(set(df1["node_id"].unique()) | set(df2["node_id"].unique()))
    all_node_ids = [nid for nid in all_node_ids if nid not in EXCLUDED_NODE_IDS]
    device_names = [NODE_LABELS.get(nid, f"Node {nid}") for nid in all_node_ids]
    x = np.arange(len(all_node_ids))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, comp in enumerate(components):
        comp_baseline = [df1[df1["node_id"] == nid][comp].sum() for nid in all_node_ids]
        comp_attack = [df2[df2["node_id"] == nid][comp].sum() for nid in all_node_ids]

        axes[i].bar(x - width/2, comp_baseline, width, label=label1, color=COLORS["baseline"])
        axes[i].bar(x + width/2, comp_attack, width, label=label2, color=COLORS["attack"])
        axes[i].set_title(f"{comp.upper()} Energy Consumption per Node (mJ)")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(device_names, rotation=45, ha='right')
        axes[i].set_ylabel("Energy (mJ)")
        axes[i].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
        axes[i].legend()

    plt.suptitle(f"Baseline vs DoS Attack - Component Energy Breakdown", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# MAIN
if __name__ == "__main__":
    baseline_file = "C:/Logs/baseline_log.txt"
    attack_file = "C:/Logs/attacks/DoS.txt"
    node_to_compare = 3  # e.g., Thermostat

    raw_df_baseline = parse_delta_energest_from_file(baseline_file)
    raw_df_attack = parse_delta_energest_from_file(attack_file)

    df_baseline = convert_ticks_to_energy(raw_df_baseline)
    df_attack = convert_ticks_to_energy(raw_df_attack)

    plot_comparison(df_baseline, "Baseline", df_attack, "DoS Attack", node_to_compare)
    plot_total_energy_bar(df_baseline, "Baseline", df_attack, "DoS Attack", node_to_compare)
    plot_total_energy_comparison(df_baseline, "Baseline", df_attack, "DoS Attack")
    plot_component_bar_charts(df_baseline, "Baseline", df_attack, "DoS Attack")

    comparison_table = generate_energy_comparison_table(df_baseline, "Baseline", df_attack, "DoS Attack")
    print("\n=== Energy Comparison Table (in mJ) ===\n")
    print(comparison_table)

    plot_percentage_increase_chart(comparison_table)
    comparison_table.to_csv("G:/_Project2025/energy_comparison_DoS.csv", index=False)
