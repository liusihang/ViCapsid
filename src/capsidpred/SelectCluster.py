#!/usr/bin/env python3
import subprocess
import os
import pandas as pd
import networkx as nx
from Bio import SeqIO
from pathlib import Path

def run_blast_and_cluster(fasta_path, out_dir, threads=8, min_ani=95.0, min_tcov=85.0, min_qcov=85.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / "temp_db"
    blast_out = out_dir / "blast.tsv"
    ani_out = out_dir / "ani.tsv"
    cluster_out = out_dir / "clusters.tsv"
    cluster_dir = out_dir / "cluster_fasta"
    cluster_dir.mkdir(exist_ok=True)

    # Step 1: makeblastdb
    print("[🔄] Building BLAST database...")
    subprocess.run(["makeblastdb", "-in", str(fasta_path), "-dbtype", "nucl", "-out", str(db_path)], check=True)

    # Step 2: blastn all-vs-all
    print("[🔄] Running all-vs-all BLASTN...")
    subprocess.run([
        "blastn", "-query", str(fasta_path), "-db", str(db_path),
        "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen",
        "-max_target_seqs", "10000",
        "-out", str(blast_out), "-num_threads", str(threads)
    ], check=True)

    # Step 3: Calculate ANI table (like anicalc.py)
    print("[📊] Calculating ANI table...")
    ani_df = pd.read_csv(blast_out, sep='\t', header=None,
        names=["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
               "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qlen", "slen"])

    # Filter based on thresholds
    ani_df = ani_df[ani_df['qseqid'] != ani_df['sseqid']]  # remove self-hits
    ani_df['qcov'] = ani_df['length'] / ani_df['qlen'] * 100
    ani_df['tcov'] = ani_df['length'] / ani_df['slen'] * 100

    filtered = ani_df[(ani_df['pident'] >= min_ani) &
                      (ani_df['qcov'] >= min_qcov) &
                      (ani_df['tcov'] >= min_tcov)][['qseqid', 'sseqid']].drop_duplicates()
    filtered.to_csv(ani_out, sep='\t', index=False, header=False)

    # Step 4: Cluster sequences using graph (like aniclust.py)
    print("[🧠] Clustering sequences based on ANI and coverage...")
    G = nx.Graph()
    G.add_edges_from(filtered.values)

    all_seqs = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    all_ids = set(all_seqs.keys())
    linked_ids = set(G.nodes)
    for sid in all_ids - linked_ids:
        G.add_node(sid)

    with open(cluster_out, "w") as cm:
        for i, comp in enumerate(nx.connected_components(G), start=1):
            cluster_id = f"Cluster_{i}"
            for sid in comp:
                cm.write(f"{sid}\t{cluster_id}\n")
            cluster_seqs = [all_seqs[s] for s in comp if s in all_seqs]
            with open(cluster_dir / f"{cluster_id}.fasta", "w") as out_fa:
                SeqIO.write(cluster_seqs, out_fa, "fasta")

    print(f"✅ Done. Clusters written to: {cluster_dir}/ and mapping to: {cluster_out}")

# 示例调用
# run_blast_and_cluster("merged_sequences.fasta", "output_dir", threads=8)
