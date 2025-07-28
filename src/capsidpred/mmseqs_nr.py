#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mmseqs_nr.py
------------
用 mmseqs2 easy-cluster 对给定 faa 文件进行聚类，并用 Biopython 导出非冗余代表序列 fasta。

可作为模块被其它脚本 import，也可直接命令行运行。

依赖:
  - mmseqs (需在 PATH 中)
  - biopython

示例:
  python mmseqs_nr.py input.faa output_nr.faa \
      --min-seq-id 0.9 --cov 0.8 --cov-mode 1 --threads 16 --tmpdir /dev/shm/mmseqs_tmp

作为模块调用:
  from mmseqs_nr import mmseqs_easy_cluster_to_nr
  reps, out_faa = mmseqs_easy_cluster_to_nr("in.faa", "nr.faa")

作者：你自己 :)
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Set, Tuple

from Bio import SeqIO


def run_cmd(cmd, cwd=None):
    """运行外部命令并在失败时抛出异常，标准错误直接透传到终端。"""
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr


def find_cluster_tsv(outdir: str) -> str:
    """在 mmseqs easy-cluster 输出目录里找到 cluster.tsv（不同版本前缀可能不同，做个兜底）"""
    cand = os.path.join(outdir, "cluster.tsv")
    if os.path.exists(cand):
        return cand
    hits = glob.glob(os.path.join(outdir, "*cluster.tsv"))
    if not hits:
        raise FileNotFoundError(f"Cannot find cluster.tsv in {outdir}")
    return hits[0]


def parse_representatives(cluster_tsv: str) -> Set[str]:
    """解析 cluster.tsv，第一列通常是代表序列 ID；返回代表序列 ID 的集合。"""
    reps = set()
    with open(cluster_tsv) as fh:
        for line in fh:
            if not line.strip():
                continue
            rep, _ = line.rstrip("\n").split("\t", 1)
            reps.add(rep)
    return reps


def write_nr_faa(original_faa: str, out_faa: str, rep_ids: Set[str]) -> int:
    """从原始 fasta 里挑出代表序列写到 out_faa。返回写入的数量。"""
    count = 0
    with open(out_faa, "w") as out_handle:
        for rec in SeqIO.parse(original_faa, "fasta"):
            # mmseqs 默认使用 fasta header 的第一个空格前的部分作为 ID
            seq_id = rec.id
            if seq_id in rep_ids:
                SeqIO.write(rec, out_handle, "fasta")
                count += 1
    return count


def mmseqs_easy_cluster_to_nr(
    in_faa: str,
    out_faa: str,
    min_seq_id: float = 0.9,
    cov: float = 0.8,
    cov_mode: int = 1,
    threads: int = 8,
    tmpdir: str = None,
    keep_tmp: bool = False,
) -> Tuple[Set[str], str]:
    """
    核心函数：对 in_faa 进行 easy-cluster，并把代表序列写入 out_faa。
    返回 (rep_ids, out_faa_path)
    """
    if shutil.which("mmseqs") is None:
        raise EnvironmentError("mmseqs 不在 PATH 中，请先安装或加入 PATH。")

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="mmseqs_tmp_")
        created_tmp = True
    else:
        os.makedirs(tmpdir, exist_ok=True)
        created_tmp = False

    outdir = tempfile.mkdtemp(prefix="mmseqs_out_", dir=tmpdir)

    try:
        # 运行 easy-cluster
        cmd = [
            "mmseqs",
            "easy-cluster",
            in_faa,
            outdir,
            tmpdir,
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(cov),
            "--cov-mode",
            str(cov_mode),
            "--threads",
            str(threads),
        ]
        _, stderr = run_cmd(cmd)
        # print(stderr, file=sys.stderr)  # 需要的话可打开

        # 解析代表序列
        cluster_tsv = find_cluster_tsv(outdir)
        rep_ids = parse_representatives(cluster_tsv)

        # 写非冗余 fasta
        n = write_nr_faa(in_faa, out_faa, rep_ids)
        sys.stderr.write(f"[mmseqs_nr] Wrote {n} representative sequences to {out_faa}\n")

        return rep_ids, out_faa
    finally:
        # 清理
        if not keep_tmp:
            shutil.rmtree(outdir, ignore_errors=True)
            if created_tmp:
                shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Use mmseqs easy-cluster to produce a non-redundant faa.")
    parser.add_argument("input_faa", help="输入的蛋白质 fasta (.faa)")
    parser.add_argument("output_faa", help="输出的非冗余 fasta (.faa)")
    parser.add_argument("--min-seq-id", type=float, default=0.9, help="最小序列一致性阈值 (default: 0.9)")
    parser.add_argument("-c", "--cov", type=float, default=0.8, help="覆盖度阈值 (default: 0.8)")
    parser.add_argument("--cov-mode", type=int, default=1, help="mmseqs 覆盖度模式 (default: 1)")
    parser.add_argument("-t", "--threads", type=int, default=8, help="线程数 (default: 8)")
    parser.add_argument("--tmpdir", type=str, default=None, help="mmseqs 临时目录 (默认自动创建并删除)")
    parser.add_argument("--keep-tmp", action="store_true", help="保留临时输出，便于调试")
    args = parser.parse_args()

    mmseqs_easy_cluster_to_nr(
        in_faa=args.input_faa,
        out_faa=args.output_faa,
        min_seq_id=args.min_seq_id,
        cov=args.cov,
        cov_mode=args.cov_mode,
        threads=args.threads,
        tmpdir=args.tmpdir,
        keep_tmp=args.keep_tmp,
    )


if __name__ == "__main__":
    main()
