#!/usr/bin/env python
# encoding: utf-8

import os
import random
import numpy as np
import argparse
import multiprocessing
import subprocess
import tensorflow as tf
import sys
import h5py

# --- 特征创建辅助函数 (未改变) ---
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))): value = value.numpy()
    if isinstance(value, str): value = value.encode('utf-8')
    if not isinstance(value, list): value = [value]
    else: value = [v.encode('utf-8') if isinstance(v, str) else v for v in value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    if not isinstance(value, list): value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int_feature(value):
    if not isinstance(value, list): value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class _GenerateTFRecordFromH5(object):
    # --- MODIFIED: __init__不再永久持有文件句柄 ---
    def __init__(self, h5_filepath, save_path, layer_index=36, shuffle=False, num_shards=30, filename_prefix="Data"):
        self.h5_filepath = h5_filepath # 只保存文件路径，字符串可以被序列化
        self.save_path = save_path
        self.layer_index = str(layer_index)
        self.shuffle = shuffle
        self.num_shards = num_shards
        self.filename_prefix = filename_prefix
        
        self.hardcoded_label_str = 'Others'
        self.label_to_id = {self.hardcoded_label_str: 0}
        print(f"Hardcoded label for all entries will be '{self.hardcoded_label_str}' with ID 0.")

        if not os.path.exists(h5_filepath):
            print(f"Error: H5 file not found: {h5_filepath}")
            sys.exit(1)

        # --- MODIFIED: 在主进程中临时打开文件，仅用于获取蛋白质ID列表 ---
        print("Reading protein ID list from H5 file...")
        with h5py.File(self.h5_filepath, 'r') as h5_file:
            self.prot_list = list(h5_file.keys())
        print(f"Found {len(self.prot_list)} proteins to process.")
        # 此处 with 语句块结束，文件句柄被自动关闭。self.h5_file 不再存在。

        if not self.prot_list:
            print("Warning: H5 file contains no proteins. Exiting.")
            sys.exit(1)

        if self.shuffle:
            random.shuffle(self.prot_list)

        shard_size = (len(self.prot_list) + num_shards - 1) // num_shards
        self.indices = [(i * shard_size, min((i + 1) * shard_size, len(self.prot_list))) for i in range(num_shards)]
        if self.indices:
            self.indices[-1] = (self.indices[-1][0], len(self.prot_list))

    def _serialize_example(self, prot_id, data):
        # 此函数无需修改
        sequence = data.get('seq', b'').decode('utf-8')
        if not sequence: return None
        d_feature = {'id': _bytes_feature(prot_id),'seq': _bytes_feature(sequence),'L': _int_feature(len(sequence)),}
        d_feature['label'] = _int_feature(self.label_to_id[data.get('label')])
        if 'representations' in data:
            reps = data['representations']
            d_feature['emb_l'], d_feature['emb_size'] = _int_feature(reps.shape[0]), _int_feature(reps.shape[1])
            d_feature['representations'] = _float_feature(list(reps.flatten()))
        if 'bos_representations' in data:
            d_feature['bos_representations'] = _float_feature(list(data['bos_representations'].flatten()))
        if 'contacts' in data:
            contacts = data['contacts']
            d_feature['pdb_l'] = _int_feature(contacts.shape[0])
            d_feature['contacts'] = _float_feature(list(contacts.flatten()))
        return tf.train.Example(features=tf.train.Features(feature=d_feature)).SerializeToString()

    def _generate_index_file(self, tfrecord_filepath):
        # 此函数无需修改
        index_filepath = os.path.splitext(tfrecord_filepath)[0] + ".index"
        print(f"Attempting to generate index file: {index_filepath}")
        try:
            subprocess.run([sys.executable, "-m", "tfrecord.tools.tfrecord2idx", tfrecord_filepath, index_filepath], check=True, capture_output=True, text=True)
            print(f"Successfully generated index file: {index_filepath}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating index file for {tfrecord_filepath}. Stderr: {e.stderr}")
        except Exception as e:
            print(f"An unexpected error occurred while generating index file for {tfrecord_filepath}: {e}")

    # --- MODIFIED: 子进程任务函数，在这里独立打开H5文件 ---
    def _convert_shard(self, shard_idx):
        # 1. 每个子进程打开自己的文件句柄
        with h5py.File(self.h5_filepath, 'r') as h5_file:
            if not os.path.exists(self.save_path): os.makedirs(self.save_path, exist_ok=True)
            tfrecord_fn = os.path.join(self.save_path, f'{self.filename_prefix}_{shard_idx + 1:02d}.tfrecords')
            print(f"Starting shard {shard_idx + 1}/{self.num_shards}. Writing to {tfrecord_fn} [PID: {os.getpid()}]")
            
            start_index, end_index = self.indices[shard_idx]
            shard_prot_list = self.prot_list[start_index:end_index]

            with tf.io.TFRecordWriter(tfrecord_fn) as writer:
                processed_count = 0
                for i, prot_id in enumerate(shard_prot_list):
                    # 2. 使用子进程自己的文件句柄
                    h5_group = h5_file[prot_id]
                    data = {'label': self.hardcoded_label_str}
                    if 'seq' not in h5_group: continue
                    data['seq'] = h5_group['seq'][()]
                    repr_path = f'representations/{self.layer_index}'
                    if repr_path in h5_group: data['representations'] = h5_group[repr_path][()]
                    bos_repr_path = f'bos_representations/{self.layer_index}'
                    if bos_repr_path in h5_group: data['bos_representations'] = h5_group[bos_repr_path][()]
                    if 'contacts' in h5_group: data['contacts'] = h5_group['contacts'][()]
                    example = self._serialize_example(prot_id, data)
                    if example: writer.write(example); processed_count += 1
            
            print(f"Finished shard {shard_idx + 1}. Wrote {processed_count} examples.")
            if processed_count > 0: self._generate_index_file(tfrecord_fn)
        # 3. with 语句块结束，文件句柄在此子进程中自动关闭

    # --- MODIFIED: run方法不再需要关闭文件 ---
    def run(self, num_threads):
        effective_num_threads = min(num_threads, self.num_shards, os.cpu_count() or 1)
        if effective_num_threads > 1:
            with multiprocessing.Pool(processes=effective_num_threads) as pool:
                pool.map(self._convert_shard, range(self.num_shards))
        else:
            for i in range(self.num_shards):
                self._convert_shard(i)
        
        # self.h5_file.close() # <--- REMOVED: 主进程中没有打开的文件需要关闭了
        print("\nAll processing finished.")

def convert_h5_to_tfrecords(h5_filepath, output_dir, filename_prefix="Data", layer_index=36, num_shards=10, num_threads=4, shuffle=False):
    # 此函数无需修改
    print("--- Starting H5 to TFRecord Conversion ---")
    print(f"  Source H5 File: {h5_filepath}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Shuffle: {shuffle}, Shards: {num_shards}, Threads: {num_threads}")
    os.makedirs(output_dir, exist_ok=True)
    generator = _GenerateTFRecordFromH5(h5_filepath=h5_filepath, save_path=output_dir, filename_prefix=filename_prefix, layer_index=layer_index, shuffle=shuffle, num_shards=num_shards)
    generator.run(num_threads=num_threads)
    print("--- H5 to TFRecord Conversion Finished ---")

if __name__ == "__main__":
    # 此部分无需修改
    parser = argparse.ArgumentParser(description="CLI for converting H5 to TFRecords. This script can also be imported as a module.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5_filepath", required=True, help="Path to the input H5 file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output TFRecord files.")
    parser.add_argument("--filename_prefix", default="Data", help="Prefix for the output TFRecord filenames.")
    parser.add_argument("--layer_index", type=int, default=36, help="The specific layer to extract from representations.")
    parser.add_argument("--num_shards", type=int, default=10, help="Number of output TFRecord shards.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of parallel processes to use.")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle the data before processing (for training set).")
    args = parser.parse_args()
    convert_h5_to_tfrecords(h5_filepath=args.h5_filepath, output_dir=args.output_dir, filename_prefix=args.filename_prefix, layer_index=args.layer_index, num_shards=args.num_shards, num_threads=args.num_threads, shuffle=args.shuffle)