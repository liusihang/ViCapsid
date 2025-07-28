#!/usr/bin/env python3
"""
metagene_predict_merged.py

合并一个目录中的所有宏基因组装配 contigs，重命名序列以确保唯一性，
然后使用 Pyrodigal 进行基因预测，并可选择性地过滤生成的蛋白质序列。

该脚本可以作为命令行工具运行，也可以作为模块导入到其他 Python 代码中。
"""
import argparse
import tempfile
from pathlib import Path

import pyrodigal
from Bio import SeqIO


def merge_and_rename_fastas(
    in_dir: Path, out_fasta_path: Path, fasta_files: list[Path]
) -> int:
    """
    合并目录中的多个FASTA文件为一个文件，并重命名序列以防冲突。

    Args:
        in_dir (Path): 包含FASTA文件的输入目录。
        out_fasta_path (Path): 合并后的输出FASTA文件路径。
        fasta_files (list[Path]): 待处理的FASTA文件路径列表。

    Returns:
        int: 写入到输出文件中的总序列数量。
    """
    total_records = 0
    print(f"[>] Merging and renaming contigs from {len(fasta_files)} files...")

    with out_fasta_path.open("w") as out_handle:
        for fasta_path in fasta_files:
            file_stem = fasta_path.stem
            contig_counter = 1
            for record in SeqIO.parse(fasta_path, "fasta"):
                original_id = record.id
                # 创建新的唯一ID，例如: "filename_1"
                new_id = f"{file_stem}_{contig_counter}"
                
                # 更新 record 对象
                record.id = new_id
                record.description = f"[original_id={original_id}]" # 将原ID保存在描述中

                SeqIO.write(record, out_handle, "fasta")
                total_records += 1
                contig_counter += 1
    
    print(f"    - Merged a total of {total_records} contigs into '{out_fasta_path.name}'.")
    return total_records


def filter_protein_file(
    in_faa_path: Path, out_faa_path: Path, min_len: int, max_len: int
) -> int:
    """
    根据长度过滤蛋白质 FASTA 文件。

    Args:
        in_faa_path (Path): 输入的 FAA 文件路径。
        out_faa_path (Path): 输出的过滤后的 FAA 文件路径。
        min_len (int): 接受的最小蛋白质长度。
        max_len (int): 接受的最大蛋白质长度。

    Returns:
        int: 写入到输出文件中的序列数量。
    """
    records_to_keep = (
        rec
        for rec in SeqIO.parse(in_faa_path, "fasta")
        if min_len <= len(rec.seq) <= max_len
    )
    count = SeqIO.write(records_to_keep, out_faa_path, "fasta")
    return count


def process_one_file(
    fasta_path: Path,
    out_dir: Path,
    meta: bool = True,
    do_filter: bool = True,
    min_len: int = 50,
    max_len: int = 1024,
) -> str:
    """
    对单个（合并后的）FASTA 文件进行基因预测，并选择性地过滤蛋白质序列。

    Args:
        fasta_path (Path): 输入的 FASTA 文件路径。
        out_dir (Path): 输出目录的路径。
        meta (bool): 是否使用宏基因组模式。
        do_filter (bool): 是否执行蛋白质序列过滤。
        min_len (int): 过滤时保留的最小蛋白质长度。
        max_len (int): 过滤时保留的最大蛋白质长度。

    Returns:
        str: 处理后的文件的基本名称，用于日志记录。
    """
    # --- 1. 基因预测 ---
    finder = pyrodigal.GeneFinder(meta=meta)
    base = fasta_path.stem

    gff_path = out_dir / f"{base}.gff"
    cds_path = out_dir / f"{base}.fna"
    prot_path = out_dir / f"{base}.faa"

    print(f"[>] Running gene prediction on: {base}")
    with gff_path.open("w") as gff, cds_path.open("w") as cds, prot_path.open(
        "w"
    ) as prot:
        initial_protein_count = 0
        # 对合并后的大文件进行处理
        for idx, rec in enumerate(SeqIO.parse(str(fasta_path), "fasta")):
            genes = finder.find_genes(str(rec.seq))
            initial_protein_count += len(genes)
            genes.write_gff(gff, sequence_id=rec.id, header=(idx == 0))
            genes.write_genes(cds, sequence_id=rec.id)
            genes.write_translations(prot, sequence_id=rec.id)
    
    print(f"    - Predicted {initial_protein_count} proteins for {base}.")

    # --- 2. 蛋白质过滤 (可选) ---
    if do_filter:
        print(f"[>] Filtering predicted proteins for {base}...")
        # 使用临时文件进行过滤，以原子方式替换原文件，防止中途中断导致文件损坏
        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=out_dir) as tmp:
            temp_path = Path(tmp.name)
        
        kept_count = filter_protein_file(prot_path, temp_path, min_len, max_len)
        
        # 用过滤后的文件替换原始的 .faa 文件
        temp_path.rename(prot_path)
        
        print(f"    - Filtered proteins: kept {kept_count} / {initial_protein_count}.")

    return base


def main() -> None:
    """
    命令行接口的主函数。
    解析参数并执行合并、基因预测和过滤的流程。
    """
    p = argparse.ArgumentParser(
        description="Merge contigs, predict ORFs, and filter proteins for metagenome assemblies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- 输入/输出参数 ---
    p.add_argument(
        "-i", "--input", required=True, help="Directory with contig FASTA files"
    )
    p.add_argument(
        "-o", "--output", default="pyrodigal_out", help="Output directory"
    )
    p.add_argument(
        "--prefix", default="merged_assembly", help="Prefix for the merged output files"
    )
    # --- 过滤参数 ---
    g = p.add_argument_group("Filtering options")
    g.add_argument(
        "--no-filter", action="store_true", help="Disable protein sequence filtering"
    )
    g.add_argument("--min-len", type=int, default=50, help="Minimum protein length to keep")
    g.add_argument("--max-len", type=int, default=1024, help="Maximum protein length to keep")
    
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 检查输入并收集FASTA文件 ---
    if not in_path.is_dir():
        print(f"Error: Input path '{in_path}' is not a directory.")
        return

    fasta_files = sorted(
        [
            p
            for p in in_path.glob("*")
            if p.suffix.lower() in {".fa", ".fasta", ".fna", ".fas"}
        ]
    )

    if not fasta_files:
        print(f"No FASTA files found in '{in_path}'.")
        return

    # --- 2. 合并与重命名 ---
    merged_fasta_path = out_dir / f"{args.prefix}.fasta"
    merge_and_rename_fastas(in_path, merged_fasta_path, fasta_files)

    # --- 3. 基因预测与过滤 ---
    # 现在只对合并后的单个文件进行处理
    processed_name = process_one_file(
        fasta_path=merged_fasta_path,
        out_dir=out_dir,
        meta=True,
        do_filter=not args.no_filter,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    
    print(f"\n[✓] Finished processing for merged assembly: {processed_name}")
    print(f"All tasks complete. Results are in '{out_dir}'.")
    print(f" - Merged contigs: {merged_fasta_path}")
    print(f" - GFF annotations: {out_dir / f'{processed_name}.gff'}")
    print(f" - Predicted CDS (nucleotide): {out_dir / f'{processed_name}.fna'}")
    print(f" - Predicted proteins (amino acid): {out_dir / f'{processed_name}.faa'}")


if __name__ == "__main__":
    main()