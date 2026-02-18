import argparse
import logging
import random
from pathlib import Path
from typing import TextIO

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge, shuffle, and shard formatted JSONL files for pretraining.",
    )
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of directories containing formatted JSONL files (e.g. data/novels/formatted data/bilibili/formatted)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save sharded output files (e.g. data/pretrain)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.005,
        help="Ratio of data to use for validation (default: 0.005).",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=100000,
        help="Maximum number of lines per shard (default: 100,000).",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=500000,
        help="Buffer size for shuffling (default: 500,000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def get_jsonl_files(dirs: list[str]) -> list[Path]:
    files = []
    for d in dirs:
        path = Path(d)
        if not path.exists():
            logger.warning("Input directory not found: %s", d)
            continue
        files.extend(list(path.rglob("*.jsonl")))
    return files


class ShardWriter:
    """分片写入器，支持 context manager。"""

    def __init__(self, output_dir: Path, prefix: str, shard_size: int):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.shard_idx = 0
        self.current_lines = 0
        self.total_lines = 0
        self.file_handle: TextIO | None = None

    def __enter__(self):
        self._open_new_shard()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _open_new_shard(self):
        if self.file_handle:
            self.file_handle.close()

        filename = f"{self.prefix}_part_{self.shard_idx:03d}.jsonl"
        path = self.output_dir / filename
        self.file_handle = path.open("w", encoding="utf-8")
        self.current_lines = 0
        self.shard_idx += 1
        logger.info("Started new shard: %s", path)

    def write(self, line: str):
        self.file_handle.write(line + "\n")
        self.current_lines += 1
        self.total_lines += 1
        if self.current_lines >= self.shard_size:
            self._open_new_shard()

    def close(self):
        if self.file_handle:
            # 删除空分片文件
            if self.current_lines == 0 and self.file_handle.name:
                path = Path(self.file_handle.name)
                self.file_handle.close()
                self.file_handle = None
                if path.exists():
                    path.unlink()
                    logger.info("Removed empty shard: %s", path)
            else:
                self.file_handle.close()
                self.file_handle = None


def interleaved_line_reader(files: list[Path], lines_per_file: int = 1000):
    """从多个文件中交错读取行，改善 shuffle 质量。

    每次从每个文件读取 lines_per_file 行，循环直至所有文件耗尽。
    """
    handles = []
    for f in files:
        try:
            handles.append(f.open("r", encoding="utf-8"))
        except Exception as e:
            logger.error("Error opening file %s: %s", f, e)

    try:
        active = list(range(len(handles)))
        while active:
            next_active = []
            for i in active:
                count = 0
                for raw_line in handles[i]:
                    line = raw_line.strip()
                    if line:
                        yield line
                        count += 1
                        if count >= lines_per_file:
                            next_active.append(i)
                            break
                # If the inner for loop ended without break, the file is exhausted
            active = next_active
    finally:
        for h in handles:
            try:
                h.close()
            except Exception:
                pass


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files and shuffle file order
    input_files = get_jsonl_files(args.input_dirs)
    random.shuffle(input_files)
    logger.info("Found %d input files. Processing...", len(input_files))

    for f in input_files:
        logger.info("  - %s", f)

    buffer: list[str] = []
    total_lines = 0
    total_val = 0

    val_path = output_dir / "validation.jsonl"

    with (
        open(val_path, "w", encoding="utf-8") as val_file,
        ShardWriter(output_dir, "pretrain_v1", args.shard_size) as train_writer,
    ):

        def flush_buffer(buf: list[str], fraction: float = 0.5) -> list[str]:
            """Shuffle buffer and flush a fraction of it."""
            nonlocal total_val
            random.shuffle(buf)
            flush_count = int(len(buf) * fraction)
            to_flush = buf[:flush_count]
            remaining = buf[flush_count:]

            for item in to_flush:
                if random.random() < args.val_ratio:
                    val_file.write(item + "\n")
                    total_val += 1
                else:
                    train_writer.write(item)

            return remaining

        # 使用交错读取改善 shuffle 质量
        for line in tqdm(
            interleaved_line_reader(input_files, lines_per_file=1000),
            desc="Processing lines",
        ):
            buffer.append(line)
            total_lines += 1

            if len(buffer) >= args.buffer_size:
                buffer = flush_buffer(buffer, fraction=0.5)

        # Flush remaining buffer entirely
        if buffer:
            random.shuffle(buffer)
            for item in buffer:
                if random.random() < args.val_ratio:
                    val_file.write(item + "\n")
                    total_val += 1
                else:
                    train_writer.write(item)

    logger.info("Done.")
    logger.info("Total lines processed: %d", total_lines)
    logger.info("Total train lines: %d", train_writer.total_lines)
    logger.info("Total validation lines: %d", total_val)
    logger.info("Validation file: %s", val_path)
    logger.info("Train shards saved to: %s", output_dir)


if __name__ == "__main__":
    main()
