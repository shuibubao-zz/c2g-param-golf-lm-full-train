"""C2G dry-run 数据构建：给官方提交脚本喂一套「小规模但格式完全对齐」的数据。

产出（dryrun_data/）：
  sp1024.model          —— 本地语料上训练的 sentencepiece BPE，vocab_size 恰为 1024
                           （官方用 fineweb_1024_bpe.model，这里只要求格式可加载、vocab 对齐）
  fineweb_train_0000.bin —— 官方分片格式：256×int32 头（magic 20240520, ver 1, n_tokens）+ uint16 token
  fineweb_val_0000.bin

语料来源与 pg_lab 相同：本机 Python 标准库 .py 文件（离线、确定性、可哈希）。
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import sentencepiece as spm

HERE = Path(__file__).resolve().parent
OUT = HERE / "dryrun_data"
MANIFEST = HERE.parent / "data" / "manifest.json"

TRAIN_TOKENS_TARGET = 400_000
VAL_TOKENS_TARGET = 48_000


def collect_text() -> str:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    chunks: list[str] = []
    total = 0
    cap = 1_600_000  # 给 spm 训练用的字符预算
    for f in man["files"]:
        if total >= cap:
            break
        p = Path(f["path"])
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        chunks.append(t)
        total += len(t)
    return "\n".join(chunks)


def write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    header[2] = tokens.size
    with open(path, "wb") as fh:
        fh.write(header.tobytes())
        fh.write(tokens.astype("<u2").tobytes())


def main() -> int:
    import os
    OUT.mkdir(exist_ok=True)
    # sentencepiece 的 C++ 层在 Windows 上无法打开含非 ASCII 字符的路径，
    # 因此 chdir 到 dryrun_data 内，全部用纯 ASCII 相对路径。
    cwd0 = os.getcwd()
    os.chdir(OUT)
    try:
        _main_inner()
    finally:
        os.chdir(cwd0)
    return 0


def _main_inner() -> None:
    corpus_txt = Path("corpus.txt")
    if not corpus_txt.exists():
        corpus_txt.write_text(collect_text(), encoding="utf-8")
    print(f"corpus chars: {corpus_txt.stat().st_size:,}")

    model_prefix = "sp1024"
    if not Path(model_prefix + ".model").exists():
        spm.SentencePieceTrainer.train(
            input=str(corpus_txt),
            model_prefix=model_prefix,
            vocab_size=1024,
            model_type="bpe",
            character_coverage=0.9995,
            normalization_rule_name="identity",  # 默认 nmt_nfkc 在本 wheel 缺预编译 charsmap
            input_sentence_size=2_000_000,
            shuffle_input_sentence=False,
            pad_id=-1,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
        )
    sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
    assert sp.vocab_size() == 1024, sp.vocab_size()
    print(f"spm vocab ok: {sp.vocab_size()}")

    text = corpus_txt.read_text(encoding="utf-8")
    ids = np.asarray(sp.encode(text, out_type=int), dtype=np.int64)
    # 确定性切分：前 TRAIN 份训练，之后 VAL 份验证（与训练分布同源，官方 FineWeb 亦同源）
    train = ids[:TRAIN_TOKENS_TARGET]
    val = ids[TRAIN_TOKENS_TARGET : TRAIN_TOKENS_TARGET + VAL_TOKENS_TARGET]
    assert val.size >= 40_000, val.size
    write_shard(Path("fineweb_train_0000.bin"), train)
    write_shard(Path("fineweb_val_0000.bin"), val)

    meta = {
        "spm_vocab": sp.vocab_size(),
        "train_tokens": int(train.size),
        "val_tokens": int(val.size),
        "token_max": int(max(train.max(), val.max())),
        "format": "256x<int32 header(20240520,1,n,0...)> + n×<uint16>",
    }
    Path("manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
