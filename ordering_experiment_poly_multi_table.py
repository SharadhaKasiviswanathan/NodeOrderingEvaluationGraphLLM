#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Try Ollama client; fall back to REST
try:
    import ollama
    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False
import requests

RNG_SEED = 42
random.seed(RNG_SEED); np.random.seed(RNG_SEED)

# ----------------- Graph utils -----------------
def make_graph(n=50, p=0.1, seed=None):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return nx.relabel_nodes(G, {i: f"v{i}" for i in G.nodes()})

def pick_pair(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if not comps: return None, None
    comp = comps[0]
    comp_list = sorted(list(comp))                # FIX: random.sample needs a sequence
    if len(comp_list) < 2: return None, None
    return random.sample(comp_list, 2)

def compute_truth(G):
    u, v = pick_pair(G)
    sd = None
    if u and v and nx.has_path(G, u, v):
        sd = nx.shortest_path_length(G, u, v)
    return {"sd_pair": (u, v), "shortest_distance": sd}

def ordering_degree(G):
    return sorted(G.nodes(), key=lambda n: G.degree[n], reverse=True)

def ordering_bfs_all(G):
    """BFS order that covers ALL nodes by iterating components (stable & deterministic)."""
    order = []
    comps = sorted([sorted(list(c)) for c in nx.connected_components(G)], key=len, reverse=True)
    for comp in comps:
        root = comp[0]
        order.extend(list(nx.bfs_tree(G.subgraph(comp), root).nodes()))
    return order

def apply_ordering(G, algo):
    if algo == "degree":
        return ordering_degree(G)
    elif algo == "bfs":
        return ordering_bfs_all(G)
    else:
        return list(G.nodes())

# ----------------- Viz -----------------
def save_graph_image(G, gidx, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"graph_{gidx}.png")
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=RNG_SEED)
    nx.draw(G, pos, node_color="skyblue", node_size=80,
            with_labels=True, font_size=6, edge_color="gray")
    plt.title(f"Graph {gidx}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

# ----------------- Prompt -----------------
def build_prompt(G, truth, ordering="none", explain=False):
    u, v = truth["sd_pair"]
    if ordering == "none":
        nodes = list(G.nodes())
        prefix = ""
    else:
        nodes = apply_ordering(G, ordering)
        prefix = (f"Nodes are ordered using {ordering} ordering.\n" if explain
                  else "Nodes are relabeled in a fixed order.\n")
    return f"""{prefix}
Compute the shortest distance between {u} and {v} in this undirected graph.
NODES: {nodes}
EDGES: {sorted([tuple(sorted(e)) for e in G.edges()])}
Return JSON: {{"task":"shortest_distance","answer":<int>}}"""

# ----------------- LLM I/O -----------------
def call_ollama(model, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "seed": 42},
        "format": "json",
    }
    if _HAS_OLLAMA:
        resp = ollama.chat(model=model,
                           messages=payload["messages"],
                           format="json",
                           options=payload["options"])
        return resp["message"]["content"]
    r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]

def parse_json(txt):
    try:
        return json.loads(txt)
    except Exception:
        # tolerate code fences
        if "```" in txt:
            for chunk in txt.split("```"):
                chunk = chunk.strip()
                if chunk.startswith("{") and chunk.endswith("}"):
                    try: return json.loads(chunk)
                    except Exception: pass
        return {}

# ----------------- Eval -----------------
def eval_sd(parsed, truth):
    try:
        return 100.0 if int(parsed.get("answer")) == truth else 0.0
    except Exception:
        return 0.0

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=25)
    ap.add_argument("--nodes", type=int, default=50)
    ap.add_argument("--p", type=float, default=0.1, help="ER edge prob.")
    ap.add_argument("--models", type=str, default="llama3.2,gemma3:12b")
    ap.add_argument("--output-dir", type=str, default="output_poly_multi_table")
    args = ap.parse_args()

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    # Column groups per your diagram:
    # Model -> Ordering(Degree,BFS) -> Setting(UO,O,OE) -> Metric(E,P,A%)
    orderings = ["degree", "bfs"]
    settings = [("UO", "none", False), ("O", None, False), ("OE", None, True)]
    # For "O" & "OE" we'll inject the actual ordering ("degree"/"bfs") at runtime.

    # Store per-graph cell values and per-group accuracies
    cells = {}   # key: (graph, model, ordering, setting, metric) -> value
    acc_bucket = {}  # key: (model, ordering, setting) -> list of percentages

    # Transcript for auditing
    trans_path = os.path.join(outdir, "transcript.jsonl")
    tf = open(trans_path, "w", encoding="utf-8")

    # Save graph images (one per graph)
    for gidx in range(args.graphs):
        G = make_graph(args.nodes, args.p, seed=RNG_SEED + gidx)
        save_graph_image(G, gidx, outdir)
        truth = compute_truth(G)
        expected = truth["shortest_distance"]

        for model in models:
            for ord_algo in orderings:
                for label, stub_ordering, explain in settings:
                    # Resolve actual ordering setting
                    if label == "UO":
                        use_order = "none"
                    else:
                        use_order = ord_algo  # "degree" or "bfs"

                    prompt = build_prompt(G, truth, ordering=use_order, explain=explain)
                    raw = call_ollama(model, prompt)
                    parsed = parse_json(raw)
                    predicted = parsed.get("answer", None)
                    acc = eval_sd(parsed, expected)

                    # Fill table cells
                    cells[(gidx, model, ord_algo, label, "E")] = expected
                    cells[(gidx, model, ord_algo, label, "P")] = predicted
                    cells[(gidx, model, ord_algo, label, "A%")] = acc

                    # Acc bucket
                    acc_bucket.setdefault((model, ord_algo, label), []).append(acc)

                    # Log
                    tf.write(json.dumps({
                        "graph": gidx,
                        "model": model,
                        "ordering": ord_algo,
                        "setting": label,
                        "prompt": prompt,
                        "truth_shortest_distance": expected,
                        "response_raw": raw,
                        "response_parsed": parsed,
                        "accuracy_percent": acc
                    }, ensure_ascii=False) + "\n")

    tf.close()

    # ---------- Build the MultiIndex table as per your drawing ----------
    graphs_idx = list(range(args.graphs))
    top_cols = []
    for model in models:
        for ord_algo in ["degree", "bfs"]:
            for label in ["UO", "O", "OE"]:
                for metric in ["E", "P", "A%"]:
                    top_cols.append((model, ord_algo.title(), label, metric))
    # Data rows
    data = []
    for gidx in graphs_idx:
        row = {"Graph": gidx}
        for col in top_cols:
            model, ord_name, label, metric = col
            key = (gidx, model, ord_name.lower(), label, metric)
            row[col] = cells.get(key, "")
        data.append(row)

    df = pd.DataFrame(data)
    df_cols = pd.MultiIndex.from_tuples([("","", "", "Graph")] + top_cols,
                                        names=["Model", "Ordering", "Setting", "Metric"])
    # Reorder columns to match MultiIndex
    reordered = {"Graph": df["Graph"]}
    for col in top_cols:
        reordered[col] = df[col]
    final_df = pd.concat(reordered.values(), axis=1)
    final_df.columns = df_cols

    # Append "Overall Accuracy (%)" row
    overall_row = {("","", "", "Graph"): "Overall Accuracy (%)"}
    for col in top_cols:
        model, ord_name, label, metric = col
        if metric == "A%":
            vals = acc_bucket.get((model, ord_name.lower(), label), [])
            overall_row[col] = float(np.mean(vals)) if vals else ""
        else:
            overall_row[col] = ""
    final_df = pd.concat([final_df, pd.DataFrame([overall_row])], ignore_index=True)

    # Write outputs
    final_path = os.path.join(outdir, "results_table.csv")
    final_df.to_csv(final_path, index=False)

    # Also provide a tidy version
    long_rows = []
    for (model, ord_name, label, metric) in top_cols:
        series = final_df.iloc[:-1][(model, ord_name, label, metric)]  # exclude overall row
        for gidx, value in zip(graphs_idx, series.tolist()):
            long_rows.append({
                "Graph": gidx,
                "Model": model,
                "Ordering": ord_name,
                "Setting": label,
                "Metric": metric,
                "Value": value
            })
    pd.DataFrame(long_rows).to_csv(os.path.join(outdir, "results_long.csv"), index=False)

    print(f"[*] Wrote table: {final_path}")
    print(f"[*] Wrote tidy:  {os.path.join(outdir,'results_long.csv')}")
    print(f"[*] Transcript:  {trans_path}")
    print(f"[*] Graph PNGs:  graph_0.png … graph_{args.graphs-1}.png in {outdir}/")
    

if __name__ == "__main__":
    main()

