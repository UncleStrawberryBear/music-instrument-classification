#!/usr/bin/env python3
"""
make_metadata.py
--------------------------------------------
生成 train_metadata.csv / test_metadata.csv，
自动剔除无法读取或为“静音”的音频样本。
"""

import os, random, argparse, sys
import pandas as pd

# -- 可选地检查“静音” --
USE_TORCHAUDIO = True
if USE_TORCHAUDIO:
    import torchaudio
    SILENCE_RMS_TH = 1e-4      # 判断静音的阈值
    RMS_SECS = 1.0             # 只取前 1 秒做能量估计


def is_valid_audio(path: str) -> bool:
    "存在、能读、且能量高于阈值才返回 True"
    if not os.path.isfile(path):
        return False
    if not USE_TORCHAUDIO:
        return True

    try:
        wf, sr = torchaudio.load(path, frame_offset=0,
                                 num_frames=int(sr * RMS_SECS) if RMS_SECS else -1)
    except Exception:
        return False
    rms = wf.pow(2).mean().sqrt().item()
    return rms >= SILENCE_RMS_TH


def build_metadata(base_dir: str,
                   train_ratio: float = 0.7,
                   min_keep: int = 2,
                   seed: int = 42):
    random.seed(seed)
    train_rows, test_rows = [], []

    # 自动获取一级子文件夹作为 instrument label
    for inst in sorted(os.listdir(base_dir)):
        folder = os.path.join(base_dir, inst)
        if not os.path.isdir(folder):
            continue

        # 找出有效 mp3
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.endswith('.mp3') and is_valid_audio(os.path.join(folder, f))]

        if len(files) < min_keep:
            print(f"[Skip] {inst}: valid files < {min_keep}")
            continue

        random.shuffle(files)
        split = max(1, int(len(files) * train_ratio))
        train_files, test_files = files[:split], files[split:]

        train_rows.extend([(p.replace('\\', '/'), inst) for p in train_files])
        test_rows.extend([(p.replace('\\', '/'), inst) for p in test_files])

        print(f"{inst:<12s} | kept {len(files):3d} | train {len(train_files):3d} | test {len(test_files):3d}")

    return train_rows, test_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="all-samples",
                    help="根音频目录（一级子目录名 = instrument 标签）")
    ap.add_argument("--train_ratio", type=float, default=0.7,
                    help="训练集比例")
    ap.add_argument("--min_keep", type=int, default=2,
                    help="某类最少保留多少有效样本，否则整类跳过")
    ap.add_argument("--train_csv", default="train_metadata.csv")
    ap.add_argument("--test_csv",  default="test_metadata.csv")
    args = ap.parse_args()

    train_rows, test_rows = build_metadata(
        base_dir=args.base_dir,
        train_ratio=args.train_ratio,
        min_keep=args.min_keep,
    )

    if not train_rows or not test_rows:
        sys.exit("No data collected! 请检查音频路径或过滤条件。")

    pd.DataFrame(train_rows, columns=["audio_path", "instrument"]).to_csv(args.train_csv, index=False)
    pd.DataFrame(test_rows,  columns=["audio_path", "instrument"]).to_csv(args.test_csv,  index=False)

    print(f"\nDone!  train={len(train_rows)}  |  test={len(test_rows)}")
    print(f"CSV 写入 ->  {args.train_csv} , {args.test_csv}")


if __name__ == "__main__":
    main()
