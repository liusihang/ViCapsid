#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TPM-by-cluster pipeline (Scheme #1: run one method per job)

Steps:
1) Read IDs from CSV by label; expand to whole clusters via MMseqs2 cluster.tsv.
2) Filter the full genes FASTA to build reference (ref.selected.fasta).
3) Run CoverM twice (separately): -m tpm  and  -m count  -> coverm_tpm.tsv / coverm_count.tsv.
4) Parse to Seqs_tpm.csv / Seqs_count.csv (gene-level).
5) Compute gene_length.csv from reference FASTA (bp).
6) Aggregate per cluster (sum within cluster) -> Gene_tpm.csv / Gene_count.csv (cluster-level).

Outputs (in --outdir):
  - ref.selected.fasta
  - id_to_cluster.tsv
  - coverm_tpm.tsv, coverm_count.tsv
  - Seqs_tpm.csv, Seqs_count.csv, gene_length.csv
  - Gene_tpm.csv, Gene_count.csv
  - pipeline.log
"""

import os
import re
import csv
import glob
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set
from collections import defaultdict
from types import SimpleNamespace

import pandas as pd
from Bio import SeqIO


# ---------------------- utils ----------------------

def log(msg: str):
    print(msg, flush=True)


def normalize_id(x: str) -> str:
    """Match Biopython record.id behavior: take token before first space."""
    return x.split()[0].strip()


def run_cmd(cmd: List[str], cwd: Optional[str] = None, log_path: Optional[str] = None):
    prefix = f"[CMD]{' (cwd='+cwd+')' if cwd else ''} "
    log(prefix + " ".join(cmd))
    with subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
        lines = []
        for line in p.stdout:
            print(line, end="")
            lines.append(line)
        ret = p.wait()
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.writelines(lines)
        if ret != 0:
            raise RuntimeError(f"Command failed (exit {ret}): {' '.join(cmd)}")


# ---------------------- step 1: select IDs & build reference ----------------------

def get_ids_from_csv(csv_file: str, label: str, id_col: int = 0, label_col: int = -1) -> Set[str]:
    """Read IDs from CSV where row[label_col] == label."""
    ids: Set[str] = set()
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            lc = label_col if label_col >= 0 else len(row) + label_col
            if lc < 0 or lc >= len(row):
                continue
            if row[lc].strip() == label:
                if 0 <= id_col < len(row):
                    ids.add(normalize_id(row[id_col]))
    return ids


def load_cluster_index(
    cluster_tsv: str,
    cluster_col: int = 0,
    member_col: int = 1,
    delimiter: str = "\t",
    skip_header: bool = False,
):
    """
    Read MMseqs2 cluster TSV (two columns typical): cluster_id/rep \t member_id
    Returns:
      - cluster_to_members: {cluster_id -> {member_id,...}}
      - member_to_clusters: {member_id -> {cluster_id,...}}
    """
    cluster_to_members: Dict[str, Set[str]] = defaultdict(set)
    member_to_clusters: Dict[str, Set[str]] = defaultdict(set)
    with open(cluster_tsv, "r", encoding="utf-8") as f:
        if skip_header:
            next(f, None)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(delimiter)
            if len(parts) <= max(cluster_col, member_col):
                continue
            cid = normalize_id(parts[cluster_col])
            mid = normalize_id(parts[member_col])
            cluster_to_members[cid].add(mid)
            member_to_clusters[mid].add(cid)
    return cluster_to_members, member_to_clusters


def filter_fasta_by_ids(input_fasta: str, output_fasta: str, ids_to_keep: Set[str]) -> int:
    """Filter FASTA by IDs (streaming)."""
    records = (
        rec for rec in SeqIO.parse(input_fasta, "fasta")
        if normalize_id(rec.id) in ids_to_keep
    )
    return SeqIO.write(records, output_fasta, "fasta")


# ---------------------- step 2: samples & coverm ----------------------

def detect_samples(reads_dir: str) -> Dict[str, Tuple[str, str]]:
    """
    Detect paired-end samples with patterns: *_R1.*, *_R2.* (fastq/fq, gz or plain).
    Returns: {sample: (R1, R2)}
    """
    patterns = ["*.fastq.gz", "*.fq.gz", "*.fastq", "*.fq"]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(reads_dir, pat)))
    files = [f for f in files if re.search(r"_R[12]\.", os.path.basename(f))]

    sample_map: Dict[str, Dict[str, str]] = {}
    for f in files:
        base = os.path.basename(f)
        m = re.search(r"(.+)_R([12])\.(fastq|fq)(\.gz)?$", base)
        if not m:
            m = re.search(r"(.+)_R([12])", base)
        if not m:
            log(f"[WARN] Skip unrecognized read file name: {base}")
            continue
        sample = m.group(1)
        rno = m.group(2)
        sample_map.setdefault(sample, {})[rno] = os.path.abspath(f)

    pairs: Dict[str, Tuple[str, str]] = {}
    for s, d in sorted(sample_map.items()):
        if "1" in d and "2" in d:
            pairs[s] = (d["1"], d["2"])
        else:
            log(f"[WARN] Sample {s} lacks both R1/R2. Skipped.")
    if not pairs:
        raise ValueError("No paired-end samples detected. Expect Sample_R1.fastq.gz / Sample_R2.fastq.gz")
    return pairs


def run_coverm_single_method(
    method: str,
    reference_fasta: str,
    samples: Dict[str, Tuple[str, str]],
    out_tsv: Path,
    threads: int,
    mapper: str,
    min_read_aligned_len: int,
    min_read_pct_id: float,
    proper_pairs_only: bool,
    include_secondary: bool,
    exclude_supplementary: bool,
    tmpdir: Optional[str],
    bam_cache_dir: Optional[str],
    discard_unmapped: bool,
    log_path: Optional[str],
):
    cmd = [
        "coverm", "contig",
        "--reference", os.path.abspath(reference_fasta),
        "--output-format", "dense",
        "--contig-end-exclusion", "0",
        "-m", method,
        "-t", str(threads),
        "-p", mapper,
        "-o", str(out_tsv),
        "--min-read-aligned-length", str(min_read_aligned_len),
        "--min-read-percent-identity", str(min_read_pct_id),
    ]
    if proper_pairs_only:
        cmd.append("--proper-pairs-only")
    if include_secondary:
        cmd.append("--include-secondary")
    if exclude_supplementary:
        cmd.append("--exclude-supplementary")
    if tmpdir:
        os.environ["TMPDIR"] = tmpdir
    if bam_cache_dir:
        cmd += ["--bam-file-cache-directory", bam_cache_dir]
    if discard_unmapped:
        cmd.append("--discard-unmapped")

    coupled: List[str] = []
    for s in samples:
        r1, r2 = samples[s]
        coupled += [r1, r2]
    cmd += ["--coupled"] + coupled

    run_cmd(cmd, log_path=log_path)


# ---------------------- step 3: parse & aggregate ----------------------

def parse_coverm_single_method(coverm_out: Path, sample_names: List[str]) -> pd.DataFrame:
    """
    Parse a single-method dense table -> DataFrame indexed by GeneID with columns = sample_names.
    """
    df = pd.read_csv(coverm_out, sep="\t", header=0)
    id_col = df.columns[0]
    df = df.rename(columns={id_col: "GeneID"})
    # Prefer selecting columns exactly matching sample names; fallback to positional
    cols = [c for c in df.columns if c in sample_names]
    if len(cols) != len(sample_names):
        # Fallback: assume first N columns after GeneID are samples
        cols = list(df.columns[1:1 + len(sample_names)])
    d = df[["GeneID"] + cols].copy()
    d = d.set_index("GeneID")
    d.columns = sample_names  # enforce order
    return d


def compute_lengths_from_fasta(ref_fasta: Path) -> pd.DataFrame:
    """Compute sequence lengths (bp) from the reference FASTA."""
    records = [(normalize_id(rec.id), len(rec.seq)) for rec in SeqIO.parse(str(ref_fasta), "fasta")]
    df = pd.DataFrame.from_records(records, columns=["GeneID", "length"]).set_index("GeneID")
    return df


def aggregate_by_cluster(
    gene_df: pd.DataFrame,
    id_to_cluster: Dict[str, str],
    keep_unclustered_as_self: bool = True
) -> pd.DataFrame:
    """
    Sum within cluster. For IDs without cluster mapping:
      - keep_unclustered_as_self=True: keep as singleton cluster (ClusterID=GeneID)
      - else: drop
    """
    idx = gene_df.index.to_list()
    cluster_ids = []
    dropped = 0
    for gid in idx:
        cid = id_to_cluster.get(gid, "")
        if cid:
            cluster_ids.append(cid)
        else:
            if keep_unclustered_as_self:
                cluster_ids.append(gid)
            else:
                cluster_ids.append(None)
                dropped += 1
    if dropped:
        log(f"[WARN] {dropped} GeneIDs dropped due to missing cluster mapping.")

    df2 = gene_df.copy()
    df2.insert(0, "ClusterID", cluster_ids)
    df2 = df2.dropna(subset=["ClusterID"])
    out = df2.groupby("ClusterID").sum(numeric_only=True)
    out.index.name = "ClusterID"
    return out


# ---------------------- runner core ----------------------

def _run(args) -> Dict[str, Path]:
    """
    执行完整流程。返回关键输出文件路径字典，便于程序化调用使用。
    """
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "pipeline.log"

    # Step 1a) CSV IDs
    log(f"[INFO] Read IDs from CSV where label == '{args.label}' ...")
    base_ids = get_ids_from_csv(args.csv, args.label, args.id_col, args.label_col)
    if not base_ids:
        log("[ERROR] No IDs matched in CSV.")
        return {}
    log(f"[INFO] CSV matched IDs: {len(base_ids)}")

    # Step 1b) cluster mapping
    log(f"[INFO] Load cluster TSV: {args.cluster_tsv}")
    clu2mem, mem2clu = load_cluster_index(
        args.cluster_tsv,
        cluster_col=args.cluster_col,
        member_col=args.member_col,
        delimiter=args.cluster_delim,
        skip_header=args.cluster_skip_header,
    )

    hit_clusters: Set[str] = set()
    missing_in_map: Set[str] = set()
    for sid in base_ids:
        if sid in mem2clu:
            hit_clusters.update(mem2clu[sid])
        else:
            missing_in_map.add(sid)
    log(f"[INFO] clusters hit: {len(hit_clusters)}; IDs not found in cluster map: {len(missing_in_map)}")

    expanded_ids: Set[str] = set()
    for cid in hit_clusters:
        expanded_ids.update(clu2mem.get(cid, set()))
    final_ids: Set[str] = expanded_ids.union(base_ids)
    log(f"[INFO] unique IDs after cluster expansion: {len(final_ids)}")

    # Step 1c) build reference FASTA
    ref_fasta = outdir / "ref.selected.fasta"
    log(f"[INFO] Filter FASTA: {args.input_fasta} -> {ref_fasta}")
    n_written = filter_fasta_by_ids(args.input_fasta, str(ref_fasta), final_ids)
    if n_written == 0:
        log("[ERROR] No records written. Check ID consistency (spaces, prefixes).")
        return {}
    log(f"[DONE] wrote {n_written} records to {ref_fasta}")

    # GeneID -> ClusterID map (only for records present in reference)
    id_to_cluster: Dict[str, str] = {}
    for rec in SeqIO.parse(str(ref_fasta), "fasta"):
        gid = normalize_id(rec.id)
        cids = sorted(list(mem2clu.get(gid, [])))
        id_to_cluster[gid] = cids[0] if cids else ""
    map_path = outdir / "id_to_cluster.tsv"
    with open(map_path, "w", encoding="utf-8") as f:
        for gid, cid in id_to_cluster.items():
            f.write(f"{gid}\t{cid if cid else 'UNCLUSTERED'}\n")
    log(f"[DONE] GeneID->ClusterID map: {map_path}")

    # Step 2) run CoverM separately
    samples = detect_samples(args.reads_dir)
    sample_names = list(samples.keys())
    log(f"[INFO] detected {len(sample_names)} samples: {', '.join(sample_names)}")

    tpm_tsv = outdir / "coverm_tpm.tsv"
    cnt_tsv = outdir / "coverm_count.tsv"

    run_coverm_single_method(
        method="tpm",
        reference_fasta=str(ref_fasta),
        samples=samples,
        out_tsv=tpm_tsv,
        threads=args.threads,
        mapper=args.mapper,
        min_read_aligned_len=args.min_read_aligned_len,
        min_read_pct_id=args.min_read_pct_id,
        proper_pairs_only=args.proper_pairs_only,
        include_secondary=args.include_secondary,
        exclude_supplementary=args.exclude_supplementary,
        tmpdir=args.tmpdir,
        bam_cache_dir=args.bam_cache_dir,
        discard_unmapped=args.discard_unmapped,
        log_path=str(log_path),
    )

    run_coverm_single_method(
        method="count",
        reference_fasta=str(ref_fasta),
        samples=samples,
        out_tsv=cnt_tsv,
        threads=args.threads,
        mapper=args.mapper,
        min_read_aligned_len=args.min_read_aligned_len,
        min_read_pct_id=args.min_read_pct_id,
        proper_pairs_only=args.proper_pairs_only,
        include_secondary=args.include_secondary,
        exclude_supplementary=args.exclude_supplementary,
        tmpdir=args.tmpdir,
        bam_cache_dir=args.bam_cache_dir,
        discard_unmapped=args.discard_unmapped,
        log_path=str(log_path),
    )

    # Step 3) parse to gene tables
    gene_tpm = parse_coverm_single_method(tpm_tsv, sample_names)
    gene_cnt = parse_coverm_single_method(cnt_tsv, sample_names)

    # Save gene-level
    gene_tpm_out = outdir / "Seqs_tpm.csv"
    gene_cnt_out = outdir / "Seqs_count.csv"
    gene_tpm.to_csv(gene_tpm_out)
    gene_cnt.to_csv(gene_cnt_out)
    log(f"[DONE] gene TPM  -> {gene_tpm_out}")
    log(f"[DONE] gene COUNT -> {gene_cnt_out}")

    # Compute and save gene lengths from reference FASTA
    gene_len = compute_lengths_from_fasta(ref_fasta)
    gene_len_out = outdir / "gene_length.csv"
    gene_len.to_csv(gene_len_out)
    log(f"[DONE] gene LENGTH -> {gene_len_out}")

    # Aggregate to clusters (sum)
    keep_singletons = not args.drop_unclustered
    cluster_tpm = aggregate_by_cluster(gene_tpm, id_to_cluster, keep_unclustered_as_self=keep_singletons)
    cluster_cnt = aggregate_by_cluster(gene_cnt, id_to_cluster, keep_unclustered_as_self=keep_singletons)

    # Save cluster-level
    cluster_tpm_out = outdir / "Gene_tpm.csv"
    cluster_cnt_out = outdir / "Gene_count.csv"
    cluster_tpm.to_csv(cluster_tpm_out)
    cluster_cnt.to_csv(cluster_cnt_out)
    log(f"[DONE] cluster TPM  -> {cluster_tpm_out}")
    log(f"[DONE] cluster COUNT -> {cluster_cnt_out}")

    # Quick sanity: TPM column sums ~ 1e6 each
    sums = (gene_tpm.sum(axis=0)).round(3)
    log(f"[CHECK] gene TPM column sums (should be near 1e6):\n{sums}")

    log("[DONE] All finished.")

    return {
        "ref_fasta": ref_fasta,
        "id_to_cluster": map_path,
        "coverm_tpm_tsv": tpm_tsv,
        "coverm_count_tsv": cnt_tsv,
        "gene_tpm_csv": gene_tpm_out,
        "gene_count_csv": gene_cnt_out,
        "gene_length_csv": gene_len_out,
        "cluster_tpm_csv": cluster_tpm_out,
        "cluster_count_csv": cluster_cnt_out,
        "log": log_path,
        "outdir": outdir,
    }


# ---------------------- public API & CLI ----------------------

def create_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="CSV+cluster -> reference.fasta; CoverM (TPM & COUNT run separately); cluster-level aggregation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 输入与筛选
    ap.add_argument("--csv", required=True, help="CSV file path")
    ap.add_argument("--label", default="Capsid", help="CSV label value to select (default: Capsid)")
    ap.add_argument("--id_col", type=int, default=0, help="CSV ID column (0-based, default 0)")
    ap.add_argument("--label_col", type=int, default=-1, help="CSV label column (0-based; -1 for last)")

    ap.add_argument("--cluster_tsv", required=True, help="MMseqs2 cluster TSV (two cols: cluster_id/rep, member_id)")
    ap.add_argument("--cluster_col", type=int, default=0, help="cluster column index (default 0)")
    ap.add_argument("--member_col", type=int, default=1, help="member column index (default 1)")
    ap.add_argument("--cluster_delim", default="\t", help="cluster TSV delimiter (default TAB)")
    ap.add_argument("--cluster_skip_header", action="store_true", help="set if cluster TSV has a header row")

    ap.add_argument("--input_fasta", required=True, help="Full genes/CDS FASTA (nucleotide)")
    ap.add_argument("--reads_dir", required=True, help="Directory containing paired reads: *_R1/_R2*.fastq[.gz]")
    ap.add_argument("--outdir", required=True, help="Output directory")

    # CoverM 参数
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mapper", default="bwa-mem2",
                    choices=["bwa-mem2", "bwa-mem", "minimap2-sr", "minimap2-no-preset",
                             "minimap2-ont", "minimap2-pb", "minimap2-hifi"])
    ap.add_argument("--min_read_aligned_len", type=int, default=0)
    ap.add_argument("--min_read_pct_id", type=float, default=0.0)
    ap.add_argument("--proper_pairs_only", action="store_true")
    ap.add_argument("--include_secondary", action="store_true")
    ap.add_argument("--exclude_supplementary", action="store_true")
    ap.add_argument("--tmpdir", default=None)

    # 可选：缓存 BAM
    ap.add_argument("--bam_cache_dir", default=None, help="cache BAMs via coverm --bam-file-cache-directory")
    ap.add_argument("--discard_unmapped", action="store_true", help="discard unmapped when caching BAMs")

    # 聚合行为
    ap.add_argument("--drop_unclustered", action="store_true",
                    help="drop genes without cluster mapping (default: keep as singleton clusters)")
    return ap


def run_gene_abundance(
    csv: str,
    cluster_tsv: str,
    input_fasta: str,
    reads_dir: str,
    outdir: str,
    label: str = "Capsid",
    id_col: int = 0,
    label_col: int = -1,
    cluster_col: int = 0,
    member_col: int = 1,
    cluster_delim: str = "\t",
    cluster_skip_header: bool = False,
    threads: int = 8,
    mapper: str = "bwa-mem2",
    min_read_aligned_len: int = 0,
    min_read_pct_id: float = 0.0,
    proper_pairs_only: bool = False,
    include_secondary: bool = False,
    exclude_supplementary: bool = False,
    tmpdir: Optional[str] = None,
    bam_cache_dir: Optional[str] = None,
    discard_unmapped: bool = False,
    drop_unclustered: bool = False,
) -> Dict[str, Path]:
    """
    程序化 API：在其他脚本中 import 后直接调用。
    返回一个包含关键输出文件路径的字典。
    """
    args = SimpleNamespace(
        csv=csv, label=label, id_col=id_col, label_col=label_col,
        cluster_tsv=cluster_tsv, cluster_col=cluster_col, member_col=member_col,
        cluster_delim=cluster_delim, cluster_skip_header=cluster_skip_header,
        input_fasta=input_fasta, reads_dir=reads_dir, outdir=outdir,
        threads=threads, mapper=mapper,
        min_read_aligned_len=min_read_aligned_len, min_read_pct_id=min_read_pct_id,
        proper_pairs_only=proper_pairs_only, include_secondary=include_secondary,
        exclude_supplementary=exclude_supplementary, tmpdir=tmpdir,
        bam_cache_dir=bam_cache_dir, discard_unmapped=discard_unmapped,
        drop_unclustered=drop_unclustered,
    )
    return _run(args)


def main():
    parser = create_parser()
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
