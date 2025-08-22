#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="output_poly_multi_table/results_table.csv",
                    help="Path to results_table.csv")
    ap.add_argument("--output-dir", type=str, default="output_poly_multi_table",
                    help="Where to save plots")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the multiheader CSV
    df = pd.read_csv(args.input, header=[0,1,2,3])

    # Drop last row (Overall Accuracy %) to avoid messing per-graph values
    df_graphs = df.iloc[:-1].copy()

    # Collect accuracies
    records = []
    for col in df.columns:
        if col[3] == "A%":   # accuracy columns
            model, ordering, setting, metric = col
            values = df_graphs[col].dropna().astype(float).tolist()
            avg = sum(values)/len(values) if values else 0
            records.append({"Model":model, "Ordering":ordering, "Setting":setting, "Accuracy":avg})

    plot_df = pd.DataFrame(records)

    # Plot bar chart
    for model in plot_df["Model"].unique():
        sub = plot_df[plot_df["Model"]==model]
        fig, ax = plt.subplots(figsize=(7,5))
        for ordering in sub["Ordering"].unique():
            subset = sub[sub["Ordering"]==ordering]
            ax.bar([f"{ordering}-{s}" for s in subset["Setting"]],
                   subset["Accuracy"], label=ordering)
        ax.set_ylim(0,100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Average Accuracy per Setting ({model})")
        ax.legend()
        plt.tight_layout()
        outpath = os.path.join(args.output_dir, f"{model}_accuracy_plot.png")
        plt.savefig(outpath)
        plt.close()
        print(f"[*] Saved {outpath}")

    # Also save combined plot
    pivot = plot_df.pivot(index=["Ordering","Setting"], columns="Model", values="Accuracy")
    pivot.plot(kind="bar", figsize=(10,6))
    plt.ylabel("Accuracy (%)")
    plt.title("Average Accuracy by Model × Ordering × Setting")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    combined_path = os.path.join(args.output_dir, "accuracy_comparison.png")
    plt.savefig(combined_path)
    plt.close()
    print(f"[*] Saved {combined_path}")

if __name__=="__main__":
    main()

