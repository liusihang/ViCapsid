import os
import sys
import shutil
import argparse
import pyhmmer
from pyhmmer.plan7 import HMMFile, Pipeline

def run_pyhmmer(domain, dbdir, input_file, output_file, hmmsearch_score_min, threads, local_scratch, wkdir, scriptdir):
    # 确定 HMM 数据库路径
    if domain == "Viruses":
        hmmdb = os.path.join(dbdir, "hmm/viral/combined.hmm")
    else:
        domain2 = domain
        if domain2 == "Pfamviruses":
            domain2 = "Viruses"
        hmmdb = os.path.join(dbdir, f"hmm/pfam/Pfam-A-{domain2}.hmm")

    # 获取输入文件的基名
    bname = os.path.basename(input_file)
    to_scratch = False
    tmp = None

    # 尝试在本地 scratch 目录中执行高IO操作
    if os.path.isdir(local_scratch):
        try:
            tmp = os.path.join(local_scratch, f"vs2-{os.urandom(12).hex()}")
            os.makedirs(tmp)
            avail = shutil.disk_usage(local_scratch).free
            fsize = os.path.getsize(input_file) * 5
            if avail > fsize:
                shutil.copy(input_file, os.path.join(tmp, bname))
                to_scratch = True
        except Exception as e:
            to_scratch = False

    # 设置输入文件路径
    input_path = os.path.join(tmp, bname) if to_scratch else input_file

    # 初始化输出日志
    with open(f"{output_file}.log", "w") as log_file:
        log_file.write("# HMMER pyhmmer\n")

    try:
        # 加载 HMM 数据库
        with HMMFile(hmmdb) as hmm_file:
            hmms = list(hmm_file)  # 加载所有 HMM

        # 创建比对管道
        pipeline = Pipeline(alphabet=pyhmmer.easel.Alphabet.amino())

        # 加载序列文件
        with pyhmmer.easel.SequenceFile(input_path) as seq_file:
            sequences = list(seq_file)  # 加载所有序列

        # 执行比对并写入输出文件
        with open(output_file, "w") as out_file:
            out_file.write("# Target name        Access.  Query name           Access.  E-value  score  bias\n")
            for hmm in hmms:
                results = pipeline.search_hmm(hmm, sequences)
                for hit in results.hits:
                    if hit.score >= hmmsearch_score_min:  # 应用得分阈值过滤
                        out_file.write(f"{hit.name}\t-\t{hmm.name}\t-\t{hit.evalue:.2e}\t{hit.score:.2f}\t{hit.bias:.2f}\n")

    except Exception as e:
        with open(f"{output_file}.log", "a") as log_file:
            log_file.write(f"Error: {str(e)}\n")
        print(f"See error details in {wkdir}/{output_file}.log")
        sys.exit(1)

    finally:
        if to_scratch and tmp:
            os.remove(os.path.join(tmp, bname))
            os.rmdir(tmp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pyhmmer")
    parser.add_argument("--domain", required=True, help="Domain")
    parser.add_argument("--dbdir", required=True, help="Database directory")
    parser.add_argument("--input", required=True, help="Input fasta file")
    parser.add_argument("--output", required=True, help="Output file")
    parser.add_argument("--hmmsearch_score_min", type=int, required=True, help="HMMsearch score minimum")
    parser.add_argument("--threads", type=int, required=True, help="Number of threads")
    parser.add_argument("--local_scratch", required=True, help="Local scratch directory")
    parser.add_argument("--wkdir", required=True, help="Working directory")
    parser.add_argument("--scriptdir", required=True, help="Script directory")

    args = parser.parse_args()

    run_pyhmmer(
        domain=args.domain,
        dbdir=args.dbdir,
        input_file=args.input,
        output_file=args.output,
        hmmsearch_score_min=args.hmmsearch_score_min,
        threads=args.threads,
        local_scratch=args.local_scratch,
        wkdir=args.wkdir,
        scriptdir=args.scriptdir
    )