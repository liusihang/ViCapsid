#!/usr/bin/env python
import os
import argparse
import collections
import pyhmmer
from pyhmmer.easel import SequenceFile
from pyhmmer.plan7 import HMMFile

def run_pyhmmer(faa_file, hmm_dbs, output_file, threads, evalue, bitscore):
    """
    Runs PyHMMER on given faa file with multiple databases.
    
    :param faa_file: 输入的 .faa 文件路径
    :param hmm_dbs: HMM 数据库字典，键为数据库名称，值为数据库文件路径
    :param output_file: 输出结果文件路径
    :param threads: 使用的线程数
    :param evalue: pyhmmer 的 e-value 阈值
    :param bitscore: pyhmmer 的 bitscore 阈值
    """
    # 定义用于保存结果的命名元组
    Result = collections.namedtuple("Result", ["protein", "db", "phrog", "bitscore", "evalue"])
    results = []

    try:
        # 预加载 .faa 文件中的所有序列（注意：对于很大的文件，建议采用分块读取以降低内存压力）
        with SequenceFile(faa_file, digital=True) as seq_file:
            seqs = seq_file.read_block()
    except Exception as e:
        print(f"读取 {faa_file} 文件时出错: {e}")
        return

    # 针对每个 HMM 数据库文件进行搜索
    for db_name, hmm_db_file in hmm_dbs.items():
        try:
            with HMMFile(hmm_db_file) as hmms:
                for hits in pyhmmer.hmmer.hmmsearch(hmms, seqs, cpus=int(threads), E=float(evalue), T=bitscore):
                    try:
                        protein = hits.query_name.decode("utf-8")
                    except Exception:
                        protein = hits.query_name  # 如果已经是字符串则直接使用
                    for hit in hits:
                        if hit.included:
                            try:
                                phrog = hit.name.decode("utf-8")
                            except Exception:
                                phrog = hit.name
                            results.append(Result(protein, db_name, phrog, hit.score, hit.evalue))
        except Exception as e:
            print(f"处理数据库 {hmm_db_file} (名称: {db_name}) 时出错: {e}")
            continue

    # 写入输出文件，添加标题行并输出全部结果
    try:
        with open(output_file, 'w') as out_f:
            out_f.write("phrog\tdb\tprotein\tbitscore\tevalue\n")
            for result in results:
                out_f.write(f"{result.phrog}\t{result.db}\t{result.protein}\t{result.bitscore}\t{result.evalue}\n")
    except Exception as e:
        print(f"写入输出文件 {output_file} 时出错: {e}")
        return

    print(f"对于 {faa_file} 文件的搜索已完成，结果保存在 {output_file}。")

def build_hmm_dbs(db_dir):
    """
    根据指定的数据库目录构建 HMM 数据库字典。  
    只有扩展名为 .hmm 的文件会被载入。
    
    :param db_dir: 包含 HMM 文件的目录路径
    :return: 字典，键为文件名（去除扩展名），值为文件的完整路径
    """
    hmm_dbs = {}
    try:
        for filename in os.listdir(db_dir):
            if filename.endswith('.hmm'):
                db_path = os.path.join(db_dir, filename)
                db_name = os.path.splitext(filename)[0]
                hmm_dbs[db_name] = db_path
    except Exception as e:
        print(f"访问数据库目录 {db_dir} 时出错: {e}")
    return hmm_dbs

def main(args):
    # 检查输入文件和数据库目录是否存在
    if not os.path.exists(args.faa_file):
        print(f"输入文件 {args.faa_file} 不存在。")
        return
    if not os.path.isdir(args.db_dir):
        print(f"数据库目录 {args.db_dir} 不存在或不是一个目录。")
        return

    # 构建 HMM 数据库字典
    hmm_dbs = build_hmm_dbs(args.db_dir)
    if not hmm_dbs:
        print("在指定的目录中未找到任何 HMM 数据库文件。")
        return

    run_pyhmmer(args.faa_file, hmm_dbs, args.output_file, args.threads, args.evalue, args.bitscore)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用多个 HMM 数据库对给定的 .faa 文件运行 pyhmmer")
    parser.add_argument('--faa_file', required=True, help='输入 .faa 文件路径')
    parser.add_argument('--db_dir', required=True, help='包含 HMM 数据库文件的目录路径')
    parser.add_argument('--output_file', required=True, help='输出结果文件路径')
    parser.add_argument('--threads', type=int, default=1, help='使用的线程数')
    parser.add_argument('--evalue', type=float, default=1e-10, help='pyhmmer 的 e-value 阈值')
    parser.add_argument('--bitscore', type=float, default=30, help='pyhmmer 的 bitscore 阈值')

    args = parser.parse_args()
    main(args)