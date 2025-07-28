#!/usr/bin/env python
# encoding: utf-8
'''
@desc: Ultimate high-performance script to generate protein embeddings.
       This version features:
       A) 2D CALIBRATION: Build a length->max batch table via binary search (more accurate than single-length cost model).
       B) SMART BATCHING BY TABLE: Pack batches respecting Bsafe(max_len), greatly improving VRAM utilization.
       C) OOM RECOVERY: Remove the longest seq and retry until success; fatal OOM is logged.
       D) OPTIONAL SLICING & STITCHING: Handle ultra-long sequences by windowing and averaging overlaps.
       E) WRITE CACHING: Buffered HDF5 writes for better throughput.
       F) DETAILED LOGGING: Print performance/memory metrics to stderr.

Notes:
- Calibration honors your --include (per_tok/contacts) and repr_layers to avoid underestimation.
- If slicing is off, calibration won't probe lengths beyond truncation_seq_length.
'''
import os
import csv
import torch
import sys
import argparse
import re
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from types import SimpleNamespace
from Bio import SeqIO
from esm import Alphabet, FastaBatchedDataset, pretrained
import time
import h5py
import numpy as np

# ------------------------ Utils: FASTA I/O & Cleaning ------------------------
def fasta_reader(filename):
    """Reads a FASTA file yielding (header, seq) pairs."""
    with open(filename, "r") as f:
        header, seq = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    yield header, seq
                header, seq = line, ""
            else:
                seq += line
        if header:
            yield header, seq

def write_fasta(filename, records):
    """Writes a list of Bio.SeqRecord to a FASTA file."""
    with open(filename, "w") as f:
        SeqIO.write(records, f, "fasta")

def clean_seq(protein_id, seq):
    """Removes non-letters and uppercases."""
    return "".join(re.findall("[a-zA-Z]+", seq)).upper()

# ------------------------ HDF5 Writing Cache ------------------------
def save_result_to_hdf5(h5_file_handle, result_dict):
    """Write one protein's results to HDF5 efficiently."""
    protein_id = result_dict['protein_id']
    if protein_id in h5_file_handle:
        return
    group = h5_file_handle.create_group(protein_id)
    group.create_dataset('seq', data=np.bytes_(result_dict['seq']))
    group.create_dataset('seq_len', data=result_dict['seq_len'])
    for key, data in result_dict.items():
        if isinstance(data, dict):
            sub_group = group.create_group(key)
            for layer, tensor in data.items():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    tensor = tensor.cpu()
                sub_group.create_dataset(str(layer), data=tensor.to(torch.float16).numpy())
        elif key not in ['protein_id', 'seq', 'seq_len']:
            if isinstance(data, torch.Tensor):
                if data.is_cuda:
                    data = data.cpu()
                group.create_dataset(key, data=data.to(torch.float16).numpy())

def write_cache_to_hdf5(cache, h5_file_handle, force_write=False, cache_size=1000):
    """Flush cached results to HDF5."""
    if not cache or (len(cache) < cache_size and not force_write):
        return
    tqdm.write(f"--- I/O: Writing batch of {len(cache)} results to HDF5 file... ---", file=sys.stderr)
    for result in cache:
        save_result_to_hdf5(h5_file_handle, result)
    h5_file_handle.flush()
    cache.clear()
    tqdm.write(f"--- I/O: Write complete. Cache cleared. ---", file=sys.stderr)

# ------------------------ Argument Parser ------------------------
def create_parser():
    parser = argparse.ArgumentParser(
        description="ESM embeddings with 2D calibration smart batching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # I/O
    parser.add_argument("-i", "--file", type=str, required=True, help="Input FASTA or CSV file")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Output HDF5 path")

    # Model & precision
    parser.add_argument("--model_name", type=str, default="esm2_t33_650M_UR50D", help="ESM model name")
    parser.add_argument("--amp", action="store_true", help="Enable AMP (recommended on NVIDIA)")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["off", "default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--nogpu", action="store_true", help="Force CPU mode")

    # Representations
    parser.add_argument("--repr_layers", type=int, default=[-1], nargs="+",
                        help="ESM layers to extract representations from")
    parser.add_argument("--include", type=str, nargs="+", default=["mean", "per_tok", "bos", "contacts"],
                        choices=["mean", "per_tok", "bos", "contacts"],
                        help="Which representations to include")

    # Slicing
    parser.add_argument("--slicing_long_seqs", action="store_true", help="Enable slicing for long sequences")
    parser.add_argument("--truncation_seq_length", type=int, default=1022, help="ESM max tokens window (excl. BOS/EOS)")
    parser.add_argument("--slicing_stride", type=int, default=512, help="Sliding window stride")

    # HDF5 write cache
    parser.add_argument("--write_cache_size", type=int, default=1000, help="Cache size before HDF5 flush")

    # 2D calibration grid
    parser.add_argument("--calib_min_len", type=int, default=100, help="Calibration min length")
    parser.add_argument("--calib_max_len", type=int, default=1100, help="Calibration max length")
    parser.add_argument("--calib_step", type=int, default=100, help="Calibration step")
    parser.add_argument("--calibration_safety_factor", type=float, default=0.95,
                        help="Safety factor applied to Bmax(L) to get Bsafe(L)")
    parser.add_argument("--save_calib_table", type=str, default="",
                        help="Optional path to save calibration table (tsv)")
    parser.add_argument("--load_calib_table", type=str, default="",
                        help="Optional path to load an existing calibration table (tsv)")

    # DataLoader
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch_factor (GPU)")

    return parser

# ------------------------ 2D Calibration (length × batch) ------------------------
def _normalize_repr_layers(model, args):
    return [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

def _try_forward_len_batch(model, alphabet, L, B, args):
    """
    Try a forward pass with length L and batch size B using current include/repr settings.
    Return True if it succeeds, False if OOM.
    """
    seq = "A" * L
    batch = [("calib", seq) for _ in range(B)]
    batch_converter = alphabet.get_batch_converter(args.truncation_seq_length)

    # Use the same repr/contacts options as main to avoid underestimation
    return_contacts = "contacts" in args.include
    repr_layers = _normalize_repr_layers(model, args)

    try:
        with torch.no_grad():
            _, _, toks = batch_converter(batch)
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to("cuda", non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(torch.cuda.is_available() and not args.nogpu and args.amp)):
                out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
                # Access tensors to force materialization (avoid lazy freeing illusions)
                _ = out["representations"][repr_layers[0]].shape
                if return_contacts:
                    _ = out["contacts"].shape
        del out, toks
        if torch.cuda.is_available() and not args.nogpu:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available() and not args.nogpu:
                torch.cuda.empty_cache()
            return False
        raise  # re-raise non OOM errors

def _max_batch_for_length(model, alphabet, L, args, max_probe=16384):
    """
    Exponential probing to find failing upper bound, then binary search to get Bmax(L).
    Return 0 if even B=1 fails.
    """
    # Guard: if slicing is off and L exceeds window, don't test futile lengths
    if not args.slicing_long_seqs and L > args.truncation_seq_length + 2:
        return 0

    lo, hi = 1, 1
    if not _try_forward_len_batch(model, alphabet, L, 1, args):
        return 0
    # Exponential growth to find an upper failing bound
    while hi <= max_probe and _try_forward_len_batch(model, alphabet, L, hi, args):
        lo = hi
        hi = hi * 2
    # Binary search in (lo, hi)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _try_forward_len_batch(model, alphabet, L, mid, args):
            lo = mid
        else:
            hi = mid
    return lo

def calibrate_len_batch_table(model, alphabet, args):
    """
    Build a dict: length -> Bsafe(length), where Bsafe = floor(Bmax * safety_factor).
    Enforce monotonic non-increasing Bsafe with length.
    """
    if args.load_calib_table and os.path.exists(args.load_calib_table):
        table = {}
        with open(args.load_calib_table, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                Ls, Bs = line.strip().split("\t")
                table[int(Ls)] = int(Bs)
        tqdm.write(f"[CAL] Loaded calibration table from {args.load_calib_table}", file=sys.stderr)
        return table

    # If slicing disabled, cap max length at truncation window
    max_len = args.calib_max_len
    if not args.slicing_long_seqs:
        max_len = min(max_len, args.truncation_seq_length)

    grid = list(range(args.calib_min_len, max_len + 1, args.calib_step))
    table = {}
    tqdm.write(f"[CAL] Start 2D calibration: grid={grid}, include={args.include}, repr_layers={args.repr_layers}", file=sys.stderr)
    for L in grid:
        bmax = _max_batch_for_length(model, alphabet, L, args)
        bsafe = max(0, int(bmax * args.calibration_safety_factor))
        if bmax == 1 and bsafe == 0:
            bsafe = 1  # Keep at least 1 if Bmax=1
        table[L] = bsafe
        tqdm.write(f"[CAL] L={L}: Bmax={bmax} -> Bsafe={bsafe}", file=sys.stderr)

    # Post-process to ensure non-increasing with length
    last = None
    for L in sorted(table.keys()):
        if last is None:
            last = table[L]
        else:
            last = min(last, table[L])
            table[L] = last

    if args.save_calib_table:
        try:
            with open(args.save_calib_table, "w") as f:
                for L in sorted(table.keys()):
                    f.write(f"{L}\t{table[L]}\n")
            tqdm.write(f"[CAL] Saved calibration table to {args.save_calib_table}", file=sys.stderr)
        except Exception as e:
            tqdm.write(f"[CAL][WARN] Failed to save calibration table: {e}", file=sys.stderr)

    return table

def _lookup_bsafe(length, table):
    """Get Bsafe for a given length by stepping up to the nearest grid >= length."""
    keys = sorted(table.keys())
    if not keys:
        return 1
    if length <= keys[0]:
        return table[keys[0]]
    if length >= keys[-1]:
        return table[keys[-1]]
    for k in keys:
        if k >= length:
            return table[k]
    return table[keys[-1]]

def create_smart_batches_by_table(seq_lengths_and_indices, len_batch_table):
    """
    Pack batches so that batch_size ≤ Bsafe(max_len_in_batch).
    Keep ascending-by-length to minimize padding waste.
    """
    tqdm.write(f"--- Creating batches with length-batch table (grid={len(len_batch_table)}) ---", file=sys.stderr)
    batches = []
    current = []
    current_max_len = 0

    for length, index in sorted(seq_lengths_and_indices, key=lambda x: x[0]):
        if not current:
            current = [index]
            current_max_len = length
            continue
        new_max_len = max(current_max_len, length)
        bsafe = _lookup_bsafe(new_max_len, len_batch_table)
        if len(current) + 1 <= bsafe:
            current.append(index)
            current_max_len = new_max_len
        else:
            batches.append(current)
            current = [index]
            current_max_len = length

    if current:
        batches.append(current)

    tqdm.write(f"--- Batching complete. Created {len(batches)} batches. ---", file=sys.stderr)
    return batches

class ListBatchSampler(torch.utils.data.Sampler):
    """A minimal BatchSampler wrapper around a list of index-lists."""
    def __init__(self, batches):
        self.batches = batches
    def __iter__(self):
        for b in self.batches:
            yield b
    def __len__(self):
        return len(self.batches)

# ------------------------ Main Embedding Generator ------------------------
def main_embedding_generator(args, model, alphabet, fasta_file_path, write_cache, len_batch_table, final_ids_set):
    """Generate embeddings with smart batching by calibration table + OOM recovery."""
    use_gpu = torch.cuda.is_available() and not args.nogpu

    # Load and sort records by length
    records = list(SeqIO.parse(fasta_file_path, "fasta"))
    sorted_records_with_indices = sorted(enumerate(records), key=lambda x: len(x[1].seq))
    sorted_labels = [rec.id for _, rec in sorted_records_with_indices]
    sorted_seqs = [str(rec.seq) for _, rec in sorted_records_with_indices]
    dataset = FastaBatchedDataset(sorted_labels, sorted_seqs)

    seq_lengths_for_batching = [(len(seq), i) for i, seq in enumerate(sorted_seqs)]
    batches = create_smart_batches_by_table(seq_lengths_for_batching, len_batch_table)

    batch_sampler = ListBatchSampler(batches)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(args.truncation_seq_length),
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
        prefetch_factor=(args.prefetch_factor if use_gpu else 2)
    )

    return_contacts = "contacts" in args.include
    repr_layers = _normalize_repr_layers(model, args)
    isolated_sequences = []

    if use_gpu:
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for batch_labels, batch_strs, batch_tokens in tqdm(data_loader, desc="Processing Batches", file=sys.stderr):
            current_batch_data = list(zip(batch_labels, batch_strs))
            while current_batch_data:
                labels, strs = zip(*current_batch_data)
                # Rebuild tokens if we've modified the batch due to OOM
                if len(labels) != len(batch_labels):
                    batch_converter = alphabet.get_batch_converter(args.truncation_seq_length)
                    _, _, toks = batch_converter(current_batch_data)
                else:
                    toks = batch_tokens

                num_seqs = len(labels)
                total_tokens = sum(len(s) for s in strs)
                max_len = max(len(s) for s in strs) if strs else 0
                avg_len = (sum(len(s) for s in strs) / len(strs)) if strs else 0.0

                tqdm.write(f"--> Attempting batch: {num_seqs} seqs | MaxLen={max_len} | AvgLen={avg_len:.1f} | Tokens={total_tokens:,}", file=sys.stderr)

                try:
                    if use_gpu:
                        toks = toks.to(device="cuda", non_blocking=True)
                    t0 = time.time()
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(use_gpu and args.amp)):
                        out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
                    if use_gpu:
                        torch.cuda.synchronize()
                    dt = time.time() - t0

                    seqs_per_sec = num_seqs / dt if dt > 0 else float('inf')
                    toks_per_sec = total_tokens / dt if dt > 0 else float('inf')
                    log_message = f"[PERF] Batch OK | {dt:.3f}s | {seqs_per_sec:.1f} seq/s | {toks_per_sec:,.0f} tok/s"
                    if use_gpu:
                        gpu_alloc = torch.cuda.memory_allocated() / 1e6
                        gpu_resv  = torch.cuda.memory_reserved() / 1e6
                        peak_alloc = torch.cuda.max_memory_allocated() / 1e6
                        log_message += f" | GPU (MB): alloc={gpu_alloc:.1f} | reserved={gpu_resv:.1f} | peak={peak_alloc:.1f}"
                    tqdm.write(log_message, file=sys.stderr)

                    representations = {layer: t.to("cpu") for layer, t in out["representations"].items()}
                    if return_contacts:
                        contacts = out["contacts"].to("cpu")

                    for i, protein_id in enumerate(labels):
                        true_len = len(strs[i])
                        result = {"protein_id": protein_id, "seq": strs[i], "seq_len": true_len}
                        if "per_tok" in args.include:
                            result["representations"] = {layer: t[i, 1:true_len + 1].clone() for layer, t in representations.items()}
                        if "mean" in args.include:
                            result["mean_representations"] = {layer: t[i, 1:true_len + 1].mean(0).clone() for layer, t in representations.items()}
                        if "bos" in args.include:
                            result["bos_representations"] = {layer: t[i, 0].clone() for layer, t in representations.items()}
                        if return_contacts:
                            result["contacts"] = contacts[i, 1:true_len + 1, 1:true_len + 1].clone()
                        if "_chunk_" not in protein_id:
                            final_ids_set.add(protein_id)
                        write_cache.append(result)

                    write_cache_to_hdf5(write_cache, args.h5_file, cache_size=args.write_cache_size)
                    break

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Runtime OOM guard: remove the longest sequence and retry
                        if use_gpu:
                            torch.cuda.empty_cache()
                        if len(current_batch_data) == 1:
                            protein_id, seq_str = current_batch_data[0]
                            tqdm.write(f"FATAL OOM: '{protein_id}' (len={len(seq_str)}) even when alone.", file=sys.stderr)
                            with open(os.path.splitext(args.output_file)[0] + "_fatal_oom.txt", "a") as f:
                                f.write(f"{protein_id}\t{len(seq_str)}\n")
                            current_batch_data = []
                            continue
                        longest_idx = max(range(len(current_batch_data)), key=lambda i: len(current_batch_data[i][1]))
                        iso = current_batch_data.pop(longest_idx)
                        tqdm.write(f"OOM: removing longest '{iso[0]}' (len={len(iso[1])}) and retrying.", file=sys.stderr)
                        isolated_sequences.append(iso)
                    else:
                        raise
                finally:
                    if 'toks' in locals():
                        del toks
                    if 'out' in locals():
                        del out

    return isolated_sequences

# ------------------------ Stitching for Sliced Seqs ------------------------
def stitch_protein_chunks(original_id, info, h5_file, args):
    """Reconstruct embeddings of a long protein from its chunks and write back."""
    all_chunks_present = all(chunk_id in h5_file for chunk_id in info['chunks'])
    if not all_chunks_present:
        tqdm.write(f"WARNING: Missing chunks for {original_id}. Skipping stitching.", file=sys.stderr)
        return

    sample_chunk_id = info['chunks'][0]
    if sample_chunk_id not in h5_file or 'representations' not in h5_file[sample_chunk_id] or not list(h5_file[sample_chunk_id]['representations']):
        tqdm.write(f"WARNING: Invalid chunk {sample_chunk_id} for {original_id}. Skipping stitching.", file=sys.stderr)
        return

    sample_data_group = h5_file[sample_chunk_id]['representations']
    embedding_dim = sample_data_group[list(sample_data_group.keys())[0]].shape[1]
    stitched_reprs = {int(layer): torch.zeros(info['original_len'], embedding_dim, dtype=torch.float32)
                      for layer in sample_data_group.keys()}
    counts = torch.zeros(info['original_len'], 1, dtype=torch.float32)
    first_chunk_bos = None
    sorted_chunks = sorted(info['chunks'], key=lambda x: int(re.search(r'_chunk_(\d+)-', x).group(1)))

    for chunk_id in sorted_chunks:
        if chunk_id not in h5_file:
            continue
        chunk_group = h5_file[chunk_id]
        m = re.search(r'_chunk_(\d+)-', chunk_id)
        if not m:
            continue
        start = int(m.group(1))
        chunk_len = chunk_group['seq_len'][()]
        effective_end = start + chunk_len
        if 'representations' not in chunk_group:
            continue
        for layer_str, dataset in chunk_group['representations'].items():
            stitched_reprs[int(layer_str)][start:effective_end] += torch.from_numpy(dataset[:])
        counts[start:effective_end] += 1
        if start == 0 and "bos" in args.include and 'bos_representations' in chunk_group:
            first_chunk_bos = {int(layer): torch.from_numpy(dset[:]) for layer, dset in chunk_group['bos_representations'].items()}

    counts[counts == 0] = 1
    for layer in stitched_reprs:
        stitched_reprs[layer] /= counts

    final_result = {"protein_id": original_id, "seq": info['original_seq'], "seq_len": info['original_len']}
    if "per_tok" in args.include:
        final_result["representations"] = stitched_reprs
    if "mean" in args.include:
        final_result["mean_representations"] = {layer: r.mean(0) for layer, r in stitched_reprs.items()}
    if "bos" in args.include and first_chunk_bos:
        final_result["bos_representations"] = first_chunk_bos

    save_result_to_hdf5(h5_file, final_result)
    return True

def run_emb(**kwargs):
    """
    Programmatic API. All keys mirror CLI args in create_parser().
    Example:
        run(file="a.fa", output_file="out.h5", amp=True, include=["mean","per_tok"], repr_layers=[-1])
    """
    # 1) 构造与 argparse.Namespace 等价的对象
    parser = create_parser()
    defaults = vars(parser.parse_args([]))  # 取默认值
    defaults.update(kwargs)
    args = SimpleNamespace(**defaults)

    # 2) ====== 从这里复用 __main__ 里的主体逻辑 ======
    script_start_time = time.time()
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    parent_dir = os.path.dirname(args.file) if os.path.dirname(args.file) else "."

    print("Loading ESM model...")
    model, alphabet = pretrained.load_model_and_alphabet(args.model_name)
    model.eval()
    use_gpu = torch.cuda.is_available() and not args.nogpu
    if use_gpu:
        model = model.cuda()
    if args.compile_mode != "off" and sys.version_info >= (3, 8):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"Model compiled with torch.compile(mode='{args.compile_mode}').")
        except Exception as e:
            print(f"torch.compile() failed: {e}. Running uncompiled.")

    len_batch_table = calibrate_len_batch_table(model, alphabet, args)
    final_protein_ids_in_memory = set()

    with h5py.File(args.output_file, 'a') as h5_file:
        args.h5_file = h5_file
        done_set = set(h5_file.keys())
        initial_final_ids = {k for k in done_set if "_chunk_" not in k}
        final_protein_ids_in_memory.update(initial_final_ids)
        print(f"Found {len(done_set)} total entries, including {len(final_protein_ids_in_memory)} final proteins already in HDF5.")

        write_cache = []
        initial_sequences, chunks_to_stitch = [], {}

        # 读取输入（保持与原 __main__ 完全一致）
        try:
            is_csv = args.file.lower().endswith(".csv")
            reader = csv.reader(open(args.file, 'r', newline='', encoding='utf-8')) if is_csv else fasta_reader(args.file)
            input_sequences = [(f">{row[0].strip()}", row[1].strip()) if is_csv else (row[0], row[1]) for row in reader if row]
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found at {args.file}")

        print("Preprocessing sequences...")
        for protein_id, seq in tqdm(input_sequences, desc="Preprocessing"):
            protein_id_clean = protein_id.lstrip('>')
            if protein_id_clean in final_protein_ids_in_memory:
                continue
            seq = clean_seq(protein_id_clean, seq)
            if not seq:
                continue
            if args.slicing_long_seqs and len(seq) > args.truncation_seq_length:
                chunks_to_stitch[protein_id_clean] = {'chunks': [], 'original_len': len(seq), 'original_seq': seq}
                for i in range(0, len(seq), args.slicing_stride):
                    chunk_seq = seq[i:i + args.truncation_seq_length]
                    if not chunk_seq:
                        continue
                    chunk_id = f"{protein_id_clean}_chunk_{i}-{i+len(chunk_seq)}"
                    if chunk_id in done_set:
                        continue
                    chunks_to_stitch[protein_id_clean]['chunks'].append(chunk_id)
                    initial_sequences.append(SeqRecord(Seq(chunk_seq), id=chunk_id, description=""))
            else:
                initial_sequences.append(SeqRecord(Seq(seq), id=protein_id_clean, description=""))

        sequences_to_process, reprocessing_round = initial_sequences, 1

        while sequences_to_process:
            print(f"\n--- Processing Round {reprocessing_round} with {len(sequences_to_process)} sequences ---")
            temp_fasta_file = os.path.join(parent_dir, f"temp_round_{reprocessing_round}.fasta")
            write_fasta(temp_fasta_file, sequences_to_process)

            round_start_time = time.time()
            isolated = main_embedding_generator(args, model, alphabet, temp_fasta_file,
                                                write_cache, len_batch_table, final_protein_ids_in_memory)
            print(f"--- [TIMER] Round {reprocessing_round}: {time.time() - round_start_time:.2f}s ---")

            if os.path.exists(temp_fasta_file):
                os.remove(temp_fasta_file)

            if isolated:
                sequences_to_process = [SeqRecord(Seq(seq), id=pid, description="") for pid, seq in isolated]
                reprocessing_round += 1
            else:
                sequences_to_process = []

        print("\nFlushing remaining cache before stitching...")
        write_cache_to_hdf5(write_cache, h5_file, force_write=True)

        if args.slicing_long_seqs and chunks_to_stitch:
            stitch_start_time = time.time()
            print("\n--- Stitching Phase ---")
            for original_id, info in tqdm(chunks_to_stitch.items(), desc="Stitching Proteins"):
                success = stitch_protein_chunks(original_id, info, h5_file, args)
                if success:
                    final_protein_ids_in_memory.add(original_id)
            h5_file.flush()

            print("Cleaning up temporary chunk data from HDF5 file...")
            chunk_ids_to_delete = [chunk for info in chunks_to_stitch.values() for chunk in info['chunks'] if chunk in h5_file]
            for chunk_id in tqdm(chunk_ids_to_delete, desc="Deleting Chunks"):
                del h5_file[chunk_id]
            print(f"--- [TIMER] Stitching & Cleanup: {time.time() - stitch_start_time:.2f}s ---")

        print("\nFlushing any final results from cache...")
        write_cache_to_hdf5(write_cache, h5_file, force_write=True)

        keys_filepath = os.path.splitext(args.output_file)[0] + "_proteinID.index"
        print(f"\n--- Exporting Final Protein ID List ---")
        print(f"Exporting all keys to: {keys_filepath}")
        sorted_keys = sorted(list(final_protein_ids_in_memory))
        with open(keys_filepath, 'w') as f:
            for key in sorted_keys:
                f.write(key + '\n')
        print(f"Successfully exported {len(sorted_keys)} final protein IDs.")

    print(f"\nProcessing complete. Results: {args.output_file}")
    print(f"--- [TIMER] Total runtime: {time.time() - script_start_time:.2f}s ---")
    return args.output_file  # 返回结果路径

# 保留原 CLI 入口，复用 run()
if __name__ == "__main__":
    args = create_parser().parse_args()
    run_emb(**vars(args))