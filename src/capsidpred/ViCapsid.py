#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified capsid pipeline runner.

Steps:
  1) preprocess  : contigs -> predicted proteins -> NR.faa (CPU heavy)
  2) embed       : ESM embeddings -> HDF5 (GPU preferred)
  3) h5_to_tf    : HDF5 -> sharded TFRecords (CPU/IO)
  4) predict     : TFRecords -> predictions.csv (GPU preferred; can run on CPU with --predict-no-cuda)
  5) tpm         : CSV(label=="Capsid") + cluster.tsv + genes.fna + reads -> CoverM + aggregation (CPU heavy)

You can run any subset via --steps (comma-separated).
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# ------------------------- utils -------------------------

def log(msg: str):
    print(msg, flush=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def steps_from_arg(arg: str) -> List[str]:
    if arg.lower() == "all":
        return ["preprocess", "embed", "h5_to_tf", "predict", "tpm"]
    parts = [s.strip().lower() for s in arg.split(",") if s.strip()]
    valid = {"preprocess", "embed", "h5_to_tf", "predict", "tpm"}
    bad = [s for s in parts if s not in valid]
    if bad:
        raise ValueError(f"Unknown steps: {bad}. Valid: {sorted(valid)}")
    return parts

# ------------------------- imports -------------------------

# 这些模块需要你在同一项目或 PYTHONPATH 中：
try:
    import capsidpred.Preprocess as Preprocess
except ImportError as e:
    log(f"[WARN] Failed to import Preprocess: {e}")
    Preprocess = None

try:
    import capsidpred.emb_from_esm_refactored as emb_from_esm_refactored
except ImportError as e:
    log(f"[WARN] Failed to import emb_from_esm_refactored: {e}")
    emb_from_esm_refactored = None

try:
    from capsidpred.tfrecord_generator import convert_h5_to_tfrecords
except ImportError as e:
    log(f"[WARN] Failed to import convert_h5_to_tfrecords: {e}")
    convert_h5_to_tfrecords = None

try:
    from capsidpred.predict_from_tfrecord_batch import predictor
except ImportError as e:
    log(f"[WARN] Failed to import predictor: {e}")
    predictor = None

try:
    from capsidpred.gene_abundance import run_gene_abundance
except ImportError as e:
    log(f"[WARN] Failed to import tpm_by_cluster: {e}")
    run_gene_abundance = None


# ------------------------- step runners -------------------------

def step_preprocess(args) -> Path:
    """
    contigs -> NR.faa
    """
    if Preprocess is None:
        raise ImportError("Module 'Preprocess' not found. Make sure it's importable and exposes run_preprocess().")
    outdir = args.work_dir / "preprocess"
    ensure_dir(outdir)

    nr_faa = outdir / f"{args.prefix}_NR.faa"
    if nr_faa.exists() and not args.force:
        log(f"[SKIP] preprocess: {nr_faa} exists. Use --force to overwrite.")
        return nr_faa

    log("[RUN] Step 1: preprocess (contigs -> NR.faa)")
    nr_faa_path = Preprocess.run_preprocess(
        input_dir=Path(args.contigs_dir),
        output_dir=outdir,
        prefix=args.prefix,
        min_len=args.min_len,
        max_len=args.max_len,
        min_seq_id=args.min_seq_id,
        cov=args.cov,
        cov_mode=args.cov_mode,
        threads=args.mmseqs_threads,
        tmpdir=args.mmseqs_tmpdir
    )
    log(f"[DONE] NR faa written to: {nr_faa_path}")
    return Path(nr_faa_path)


def step_embed(args, nr_faa: Path) -> Path:
    """
    NR.faa -> HDF5 embeddings
    """
    if emb_from_esm_refactored is None:
        raise ImportError("Module 'emb_from_esm_refactored' not found. Expects run_emb(file=..., output_file=...).")
    outdir = args.work_dir / "embedding"
    ensure_dir(outdir)
    h5_out = args.emb_h5 or (outdir / "NR_emb.h5")
    h5_out = Path(h5_out)

    if h5_out.exists() and not args.force:
        log(f"[SKIP] embed: {h5_out} exists. Use --force to overwrite.")
        return h5_out

    log("[RUN] Step 2: embed (ESM -> HDF5)")
    # 尝试 run_emb / run 两种 API 名称
    if hasattr(emb_from_esm_refactored, "run_emb"):
        emb_from_esm_refactored.run_emb(
            file=str(nr_faa),
            output_file=str(h5_out),
            amp=not args.no_amp,
            repr_layers=args.esm_repr_layers,
            model_name=args.esm_model,
            # 你若在模块里支持 load/save calib table，可在这里透传：load_calib_table=args.esm_calib_table
        )
    else:
        raise AttributeError("emb_from_esm_refactored must provide run_emb(...) or run(...).")
    log(f"[DONE] HDF5 embeddings -> {h5_out}")
    return h5_out


def step_h5_to_tf(args, h5_file: Path) -> Path:
    """
    HDF5 -> sharded TFRecords
    """
    if convert_h5_to_tfrecords is None:
        raise ImportError("Function convert_h5_to_tfrecords not found in h5_to_tfrecords module.")
    outdir = args.work_dir / "tfrecords"
    ensure_dir(outdir)

    # 简单“已有判断”：若已存在至少一个分片，则跳过（除非 --force）
    existing = list(outdir.glob(f"{args.tfrecord_prefix}_*.tfrecords"))
    if existing and not args.force:
        log(f"[SKIP] h5_to_tf: found existing TFRecords in {outdir}. Use --force to regenerate.")
        return outdir

    log("[RUN] Step 3: H5 -> TFRecords")
    convert_h5_to_tfrecords(
        h5_filepath=str(h5_file),
        output_dir=str(outdir),
        filename_prefix=args.tfrecord_prefix,
        layer_index=args.tfrecord_layer_index,
        num_shards=args.tfrecord_shards,
        num_threads=args.tfrecord_threads,
        shuffle=args.tfrecord_shuffle
    )
    log(f"[DONE] TFRecords at: {outdir}")
    return outdir


def step_predict(args, tf_dir: Path) -> Path:
    """
    TFRecords -> predictions.csv ; 同时生成精简 labels.csv 供下一步使用（两列：id,label）
    """
    if predictor is None:
        raise ImportError("Function predictor not found in predict_from_tfrecord_batch module.")
    outdir = args.work_dir / "prediction"
    ensure_dir(outdir)
    pred_csv = args.pred_csv or (outdir / "Prediction_res.csv")
    pred_csv = Path(pred_csv)

    if pred_csv.exists() and not args.force:
        log(f"[SKIP] predict: {pred_csv} exists. Use --force to overwrite.")
    else:
        log("[RUN] Step 4: predict (TFRecords -> CSV)")
        df = predictor(
            model_dir=str(args.predict_model_dir),
            input_dir=str(tf_dir),
            output_file=str(pred_csv),
            batch_size=args.predict_batch_size,
            no_cuda=args.predict_no_cuda,
            return_df=True
        )
        if df is None or df.empty:
            raise RuntimeError("Prediction produced empty results.")

    # 生成简化 labels.csv（id,label），label 为 prediction_label
    labels_csv = outdir / "labels.csv"
    if labels_csv.exists() and not args.force:
        log(f"[SKIP] build labels.csv: {labels_csv} exists. Use --force to overwrite.")
    else:
        log("[POST] Build labels.csv (two columns: id,label)")
        df_all = pd.read_csv(pred_csv)
        if "id" not in df_all.columns:
            raise KeyError("Prediction CSV must contain column 'id'.")
        # 尝试找 'prediction_label'
        if "prediction_label" not in df_all.columns:
            # 兼容：若你的脚本写成 'label' 也可
            if "label" in df_all.columns:
                df_all = df_all.rename(columns={"label": "prediction_label"})
            else:
                raise KeyError("Prediction CSV must contain 'prediction_label' (or 'label').")
        df_slim = df_all[["id", "prediction_label"]].rename(columns={"prediction_label": "label"})
        df_slim.to_csv(labels_csv, index=False)
        log(f"[DONE] labels.csv -> {labels_csv}")

    return pred_csv


def step_tpm(args, labels_csv: Path) -> Dict[str, Path]:
    """
    labels.csv + cluster.tsv + genes.fna + reads -> coverm + aggregation
    """
    if gene_abundance is None:
        raise ImportError("Module 'tpm_by_cluster' not found (expects run_pipeline_api).")
    outdir = args.work_dir / "abundance"
    ensure_dir(outdir)

    # 若已有核心输出且不强制覆盖，则跳过
    cluster_tpm_csv = outdir / "Gene_tpm.csv"
    cluster_cnt_csv = outdir / "Gene_count.csv"
    if cluster_tpm_csv.exists() and cluster_cnt_csv.exists() and not args.force:
        log(f"[SKIP] tpm: Found {cluster_tpm_csv} & {cluster_cnt_csv}. Use --force to rerun.")
        return {
            "Capsid_tpm_csv": cluster_tpm_csv,
            "Capsid_count_csv": cluster_cnt_csv
        }

    log("[RUN] Step 5: TPM aggregation via CoverM")
    outs = run_gene_abundance(
        csv=str(labels_csv),                 # 我们刚刚生成的两列表：id,label
        cluster_tsv=str(args.mmseqs_cluster_tsv),
        input_fasta=str(args.genes_fna),    # 基因/CDS 的核酸序列（非 protein）
        reads_dir=str(args.reads_dir),
        outdir=str(outdir),
        label=args.label_of_interest,
        threads=args.coverm_threads,
        mapper=args.coverm_mapper,
        min_read_aligned_len=args.coverm_min_aln_len,
        min_read_pct_id=args.coverm_min_pct_id,
        proper_pairs_only=args.coverm_proper_pairs_only,
        include_secondary=args.coverm_include_secondary,
        exclude_supplementary=args.coverm_exclude_supplementary,
        tmpdir=args.coverm_tmpdir,
        bam_cache_dir=args.coverm_bam_cache_dir,
        discard_unmapped=args.coverm_discard_unmapped,
        drop_unclustered=args.coverm_drop_unclustered
    )
    log("[DONE] TPM aggregation finished.")
    return outs

# ------------------------- main orchestrator -------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Unified capsid pipeline runner (CPU/GPU split via --steps).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--work-dir", type=Path, required=True, help="Root working directory to hold all step outputs.")
    p.add_argument("--steps", default="all",
                   help="Comma-separated subset to run: preprocess,embed,h5_to_tf,predict,tpm; or 'all'.")

    # Common flags
    p.add_argument("--force", action="store_true", help="Force re-run and overwrite existing outputs.")
    p.add_argument("--prefix", default="gene_catalog", help="Prefix for preprocess outputs.")
    p.add_argument("--label-of-interest", default="Capsid", help="Label to filter in prediction CSV for abundance (e.g., Capsid).")

    # Step 1: preprocess (CPU heavy)
    p.add_argument("--contigs-dir", type=Path, help="Directory with input contigs (FA/FASTA/FNA/FAS).")
    p.add_argument("--min-len", type=int, default=50)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--min-seq-id", type=float, default=0.95)
    p.add_argument("--cov", type=float, default=0.9)
    p.add_argument("--cov-mode", type=int, default=1)
    p.add_argument("--mmseqs-threads", type=int, default=32)
    p.add_argument("--mmseqs-tmpdir", type=str, default=None)

    # Step 2: embed (GPU preferred)
    p.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    p.add_argument("--esm-repr-layers", type=int, nargs="+", default=[-1])
    p.add_argument("--no-amp", action="store_true", help="Disable AMP for embedding.")
    p.add_argument("--emb-h5", type=Path, default=None, help="Explicit path to write embeddings HDF5 (optional).")

    # Step 3: H5 -> TFRecords (CPU/IO)
    p.add_argument("--tfrecord-prefix", default="Data")
    p.add_argument("--tfrecord-layer-index", type=int, default=-1)
    p.add_argument("--tfrecord-shards", type=int, default=30)
    p.add_argument("--tfrecord-threads", type=int, default=8)
    p.add_argument("--tfrecord-shuffle", action="store_true")

    # Step 4: predict (GPU preferred)
    p.add_argument("--predict-model-dir", type=Path, help="Trained model directory (contains training_args.bin).")
    p.add_argument("--predict-batch-size", type=int, default=256)
    p.add_argument("--predict-no-cuda", action="store_true")
    p.add_argument("--pred-csv", type=Path, default=None, help="Explicit output path for predictions CSV (optional).")

    # Step 5: TPM (CPU heavy)
    p.add_argument("--mmseqs-cluster-tsv", type=Path, help="MMseqs2 cluster mapping TSV (cluster_id \\t member_id).", required=False)
    p.add_argument("--genes-fna", type=Path, help="Genes/CDS nucleotide FASTA used as reference for mapping.", required=False)
    p.add_argument("--reads-dir", type=Path, help="Directory containing paired reads *_R1/_R2*.fastq(.gz).", required=False)

    p.add_argument("--coverm-threads", type=int, default=32)
    p.add_argument("--coverm-mapper", default="bwa-mem2",
                   choices=["bwa-mem2", "bwa-mem", "minimap2-sr", "minimap2-no-preset",
                            "minimap2-ont", "minimap2-pb", "minimap2-hifi"])
    p.add_argument("--coverm-min-aln-len", type=int, default=0)
    p.add_argument("--coverm-min-pct-id", type=float, default=0.0)
    p.add_argument("--coverm-proper-pairs-only", action="store_true")
    p.add_argument("--coverm-include-secondary", action="store_true")
    p.add_argument("--coverm-exclude-supplementary", action="store_true")
    p.add_argument("--coverm-tmpdir", default=None)
    p.add_argument("--coverm-bam-cache-dir", default=None)
    p.add_argument("--coverm-discard-unmapped", action="store_true")
    p.add_argument("--coverm-drop-unclustered", action="store_true",
                   help="Drop genes without cluster mapping (default: keep as singleton clusters).")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()

    steps = steps_from_arg(args.steps)
    log(f"[INFO] Running steps: {steps}")
    ensure_dir(args.work_dir)

    # cache for produced paths across steps
    produced: Dict[str, Any] = {}

    # 1) preprocess
    if "preprocess" in steps:
        if args.contigs_dir is None:
            parser.error("--contigs-dir is required for step 'preprocess'.")
        nr_faa = step_preprocess(args)
        produced["nr_faa"] = nr_faa
    else:
        # 如果跳过，但后续需要，用默认位置探测
        candidate = args.work_dir / "preprocess" / f"{args.prefix}_NR.faa"
        if candidate.exists():
            produced["nr_faa"] = candidate

    # 2) embed
    if "embed" in steps:
        nr_faa = produced.get("nr_faa")
        if nr_faa is None or not Path(nr_faa).exists():
            parser.error("Step 'embed' requires NR.faa from 'preprocess' (or provide it at work_dir/preprocess/<prefix>_NR.faa).")
        h5 = step_embed(args, Path(nr_faa))
        produced["emb_h5"] = h5
    else:
        candidate = args.emb_h5 or (args.work_dir / "embedding" / "NR_emb.h5")
        if Path(candidate).exists():
            produced["emb_h5"] = Path(candidate)

    # 3) h5_to_tf
    if "h5_to_tf" in steps:
        h5 = produced.get("emb_h5")
        if h5 is None or not Path(h5).exists():
            parser.error("Step 'h5_to_tf' requires embeddings HDF5 from 'embed' (or set --emb-h5 to an existing file).")
        tf_dir = step_h5_to_tf(args, Path(h5))
        produced["tf_dir"] = tf_dir
    else:
        candidate = args.work_dir / "tfrecords"
        if candidate.exists():
            produced["tf_dir"] = candidate

    # 4) predict
    if "predict" in steps:
        if args.predict_model_dir is None:
            parser.error("--predict-model-dir is required for step 'predict'.")
        tf_dir = produced.get("tf_dir")
        if tf_dir is None or not Path(tf_dir).exists():
            parser.error("Step 'predict' requires TFRecords dir from 'h5_to_tf' (work_dir/tfrecords).")
        pred_csv = step_predict(args, Path(tf_dir))
        produced["pred_csv"] = pred_csv
    else:
        candidate = args.pred_csv or (args.work_dir / "prediction" / "Prediction_res.csv")
        if Path(candidate).exists():
            produced["pred_csv"] = Path(candidate)

    # 5) tpm (abundance)
    if "tpm" in steps:
        # 需要 labels.csv、cluster.tsv、genes.fna、reads_dir
        labels_csv = args.work_dir / "prediction" / "labels.csv"
        if not labels_csv.exists():
            # 若预测步骤没跑，则尝试从 pred_csv 生成 labels.csv（id,prediction_label）
            pred_csv = produced.get("pred_csv")
            if pred_csv is None or not Path(pred_csv).exists():
                parser.error("Step 'tpm' needs labels.csv from 'predict'. Run step 'predict' first or provide prediction CSV at work_dir/prediction/Prediction_res.csv.")
            df_all = pd.read_csv(pred_csv)
            if "id" not in df_all.columns:
                parser.error("Prediction CSV must contain column 'id'.")
            if "prediction_label" not in df_all.columns and "label" not in df_all.columns:
                parser.error("Prediction CSV must contain 'prediction_label' (or 'label').")
            df_all = df_all.rename(columns={"label": "prediction_label"})
            df_slim = df_all[["id", "prediction_label"]].rename(columns={"prediction_label": "label"})
            df_slim.to_csv(labels_csv, index=False)
            log(f"[POST] Built labels.csv -> {labels_csv}")

        if args.mmseqs_cluster_tsv is None or args.genes_fna is None or args.reads_dir is None:
            parser.error("Step 'tpm' requires --mmseqs-cluster-tsv, --genes-fna, --reads-dir.")
        outs = step_tpm(args, labels_csv)
        produced.update(outs)

    log("[ALL DONE] Outputs directory: {}".format(args.work_dir))


if __name__ == "__main__":
    main()
