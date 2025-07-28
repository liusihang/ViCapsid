#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First step，整合了基因预测和去冗余两个步骤。
High CPU requirement
From contigs to single faa/fna
"""
import argparse
from pathlib import Path
import sys

# 导入您编写的两个模块
from . import metagene_predict_merged
from . import mmseqs_nr

def run_preprocess(
    input_dir: Path,
    output_dir: Path,
    prefix: str,
    min_len: int,
    max_len: int,
    min_seq_id: float,
    cov: float,
    cov_mode: int,
    threads: int,
    tmpdir: str = None
):
    """
    运行从Contigs到非冗余基因集的完整流程。
    这是一个可被其他脚本调用的核心函数。

    Args:
        input_dir (Path): 包含contig FASTA文件的目录。
        output_dir (Path): 所有结果的主输出目录。
        prefix (str): 用于合并和最终文件的文件名前缀。
        min_len (int): 预测时保留的最小蛋白质长度。
        max_len (int): 预测时保留的最大蛋白质长度。
        min_seq_id (float): MMseqs2聚类的最小序列一致性。
        cov (float): MMseqs2聚类的最小覆盖度。
        cov_mode (int): MMseqs2的覆盖度模式。
        threads (int): 用于聚类的线程数。
        tmpdir (str, optional): MMseqs2使用的临时目录。Defaults to None.

    Returns:
        Path: 生成的最终非冗余基因集的文件路径。
    """
    # --- 准备工作 ---
    output_dir.mkdir(parents=True, exist_ok=True)
    final_nr_faa_path = output_dir / f"{prefix}_NR.faa"

    print(f"--- Starting Full Pipeline ---")
    print(f"Input Contigs: {input_dir}")
    print(f"Final Output Directory: {output_dir}")
    print(f"Final NR Catalog: {final_nr_faa_path}")
    print("-" * 30)

    # === 步骤 1: 合并与基因预测 ===
    print("\n[STEP 1/2] Merging contigs and predicting genes...")
    fasta_files = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in {".fa", ".fasta", ".fna", ".fas"}])
    if not fasta_files:
        msg = f"Error: No FASTA files found in '{input_dir}'."
        raise FileNotFoundError(msg)

    merged_contig_path = output_dir / f"{prefix}_merged_contigs.fasta"
    metagene_predict_merged.merge_and_rename_fastas(input_dir, merged_contig_path, fasta_files)
    
    base_name_for_prediction = merged_contig_path.stem 
    metagene_predict_merged.process_one_file(
        fasta_path=merged_contig_path,
        out_dir=output_dir,
        do_filter=True,
        min_len=min_len,
        max_len=max_len,
    )
    predicted_faa_path = output_dir / f"{base_name_for_prediction}.faa"
    print(f"Predicted proteins (redundant) saved to: {predicted_faa_path}")

    # === 步骤 2: 聚类去冗余 ===
    print("\n[STEP 2/2] Clustering proteins to create non-redundant catalog...")
    mmseqs_nr.mmseqs_easy_cluster_to_nr(
        in_faa=str(predicted_faa_path),
        out_faa=str(final_nr_faa_path),
        min_seq_id=min_seq_id,
        cov=cov,
        cov_mode=cov_mode,
        threads=threads,
        tmpdir=tmpdir,
    )

    print("\n--- Pipeline Finished Successfully! ---")
    print(f"Final non-redundant gene catalog is ready at: {final_nr_faa_path}")
    
    return final_nr_faa_path


def main():
    """命令行接口，解析参数并调用核心流程函数。"""
    parser = argparse.ArgumentParser(
        description="Full pipeline: from contigs to a non-redundant protein catalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 参数定义部分与之前完全相同
    parser.add_argument("-i", "--input", required=True, help="Directory with contig FASTA files")
    parser.add_argument("-o", "--output", default="gene_catalog_results", help="Main output directory for all results")
    parser.add_argument("--prefix", default="gene_catalog", help="Prefix for the final merged and NR files")
    pg = parser.add_argument_group("Gene Prediction Options (for pyrodigal)")
    pg.add_argument("--min-len", type=int, default=50, help="Minimum protein length to keep")
    pg.add_argument("--max-len", type=int, default=1024, help="Maximum protein length to keep")
    cg = parser.add_argument_group("Clustering Options (for mmseqs2)")
    cg.add_argument("--min-seq-id", type=float, default=0.95, help="Clustering: minimum sequence identity")
    cg.add_argument("--cov", type=float, default=0.9, help="Clustering: minimum coverage")
    cg.add_argument("--cov-mode", type=int, default=1, help="Clustering: mmseqs coverage mode")
    cg.add_argument("--threads", type=int, default=8, help="Threads for the clustering step")
    cg.add_argument("--tmpdir", type=str, default=None, help="Temporary directory for mmseqs2")

    args = parser.parse_args()

    try:
        # 调用核心函数，并将命令行参数解包传入
        run_preprocess(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            prefix=args.prefix,
            min_len=args.min_len,
            max_len=args.max_len,
            min_seq_id=args.min_seq_id,
            cov=args.cov,
            cov_mode=args.cov_mode,
            threads=args.threads,
            tmpdir=args.tmpdir
        )
    except (FileNotFoundError, EnvironmentError, RuntimeError) as e:
        print(f"\nPipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()