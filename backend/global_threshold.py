"""
Global Threshold Filter for Network Edge Files.

Reads any file with 3+ columns using positional access (no predefined column names).
Auto-detects separator. Filters rows based on a numeric column threshold.

Output CSV: source, target, + remaining columns (same order as input)

Flask entry point:
    from global_threshold import run_global_threshold
    run_global_threshold(input_file, output_file,
                         col_index=2, threshold=1.0, direction='gte')

CLI (interactive):
    python global_threshold.py edges.csv
"""

import os
import time
import traceback

import pandas as pd
from tqdm import tqdm

def _detect_separator(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readline()
    if '\t' in line:  return '\t'
    if ',' in line:   return ','
    if ';' in line:   return ';'
    return r'\s+'

def _count_lines(filepath: str) -> int:
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def _peek(filepath: str, sep: str, n: int = 5) -> pd.DataFrame:
    return pd.read_csv(
        filepath, sep=sep, header=None,
        nrows=n, dtype=str,
        engine='python', skipinitialspace=True
    )

def _ask_user(sample: pd.DataFrame):
    n_cols = sample.shape[1]
    print("\n=== File preview (first rows) ===")
    print("  " + "  ".join(f"[{i}]" for i in range(n_cols)))
    for i in range(len(sample)):
        print("  " + "  ".join(f"{str(sample.iloc[i, j]):<20}" for j in range(n_cols)))
    print(f"\nColumn indices: 0 to {n_cols - 1}")
    print("  [0] = source/gene   [1] = target/disease   [2+] = numeric values\n")

    while True:
        try:
            col = int(input(f"Filter on column index (0-{n_cols-1}): ").strip())
            if 0 <= col < n_cols:
                break
        except ValueError:
            pass
        print("  Invalid. Enter an integer.")

    while True:
        try:
            thr = float(input("Threshold value: ").strip())
            break
        except ValueError:
            print("  Invalid. Enter a number (e.g. 0.5, 100, 1e-3).")

    while True:
        d = input("Keep rows where value is:\n  [1] >= threshold\n  [2] <= threshold\nChoice: ").strip()
        if d == '1':
            direction = 'gte'; break
        elif d == '2':
            direction = 'lte'; break
        print("  Enter 1 or 2.")

    sym = '>=' if direction == 'gte' else '<='
    print(f"\nFilter: col[{col}] {sym} {thr}\n")
    return col, thr, direction

class GlobalThresholdFilter:
    """
    Filters network edge files by a numeric column threshold.

    Uses positional access (iloc), no predefined column names.
    Auto-detects separator (tab, comma, semicolon, whitespace).

    Example:
        f = GlobalThresholdFilter(col_index=2, threshold=3.0, direction='gte')
        stats = f.run('edges.csv', 'filtered.csv')

    Or via Flask helper:
        run_global_threshold(input_file, output_file,
                             col_index=2, threshold=3.0, direction='gte')
    """

    def __init__(self,
                 col_index: int   = 2,
                 threshold: float = 0.0,
                 direction: str   = 'gte',
                 chunk_size: int  = 50_000,
                 logger=None,
                 progress_callback=None):
        """
        Args:
            col_index   : positional index of the numeric column to filter
            threshold   : numeric threshold value
            direction   : 'gte' → keep rows where col >= threshold
                          'lte' → keep rows where col <= threshold
            chunk_size  : rows per pandas chunk
            logger      : optional logger
            progress_callback : optional callable(percent 0-100)
        """
        self.col_index          = col_index
        self.threshold          = threshold
        self.direction          = direction
        self.chunk_size         = chunk_size
        self.logger             = logger
        self.progress_callback  = progress_callback

    def _log(self, message, level='info'):
        if self.logger:
            getattr(self.logger, level, self.logger.info)(message)
        else:
            print(message)

    def _update_progress(self, percent):
        if self.progress_callback:
            self.progress_callback(percent)

    def run(self, input_file: str, output_file: str) -> dict:
        """
        Reads input_file in chunks, applies threshold filter,
        writes result to output_file as CSV.

        Output header: source, target, col_2, col_3, ... (positional names)
        First two columns always renamed to source / target.

        Returns:
            dict with success, rows_in, rows_out, execution_time
        """
        sep          = _detect_separator(input_file)
        total_lines  = _count_lines(input_file)
        total_chunks = max(1, total_lines // self.chunk_size)
        symbol       = '>=' if self.direction == 'gte' else '<='

        self._log(
            f"Separator: {repr(sep)} | lines: {total_lines} | "
            f"filter: col[{self.col_index}] {symbol} {self.threshold}"
        )
        self._update_progress(5)

        start       = time.time()
        rows_in     = 0
        rows_out    = 0
        first_write = True

        chunk_reader = pd.read_csv(
            input_file, sep=sep, header=None,
            chunksize=self.chunk_size, dtype=str,
            engine='python', skipinitialspace=True
        )

        try:
            for idx, chunk in enumerate(tqdm(chunk_reader, total=total_chunks, desc="Processing network")):
                rows_in += len(chunk)

                chunk = chunk[chunk.iloc[:, 0].notna() & chunk.iloc[:, 1].notna()]
                chunk = chunk[~chunk.iloc[:, 0].astype(str).str.strip().str.lower().isin(['nan', 'none', ''])]
                chunk = chunk[~chunk.iloc[:, 1].astype(str).str.strip().str.lower().isin(['nan', 'none', ''])]

                if self.col_index >= chunk.shape[1]:
                    self._log(f"col_index {self.col_index} out of range for this chunk (ncols={chunk.shape[1]})", 'warning')
                    continue

                numeric_col = pd.to_numeric(chunk.iloc[:, self.col_index], errors='coerce')
                valid       = numeric_col.notna()
                chunk       = chunk[valid]
                numeric_col = numeric_col[valid]

                mask = (numeric_col >= self.threshold) if self.direction == 'gte' \
                       else (numeric_col <= self.threshold)
                filtered = chunk[mask.values]
                rows_out += len(filtered)

                if filtered.empty:
                    self._update_progress(5 + int(90 * (idx + 1) / total_chunks))
                    continue

                n = filtered.shape[1]
                col_names    = ['source', 'target'] + [f'col_{i}' for i in range(2, n)]
                filtered.columns = col_names

                if first_write:
                    filtered.to_csv(output_file, index=False, mode='w', sep='\t')
                    first_write = False
                else:
                    filtered.to_csv(output_file, index=False, mode='a', header=False, sep='\t')

                self._update_progress(5 + int(90 * (idx + 1) / total_chunks))

            if first_write:
                n         = _peek(input_file, sep, 1).shape[1]
                col_names = ['source', 'target'] + [f'col_{i}' for i in range(2, n)]
                pd.DataFrame(columns=col_names).to_csv(output_file, index=False, sep='\t')

            self._update_progress(100)
            elapsed = round(time.time() - start, 2)
            self._log(f"Done in {elapsed}s | in: {rows_in} | out: {rows_out} | → {output_file}")

            return {
                'success':        True,
                'rows_in':        rows_in,
                'rows_out':       rows_out,
                'col_index':      self.col_index,
                'threshold':      self.threshold,
                'direction':      self.direction,
                'execution_time': elapsed,
                'output_file':    os.path.basename(output_file),
            }

        except Exception as e:
            traceback.print_exc()
            self._log(f"Error: {e}", 'error')
            return {'success': False, 'error': str(e)}

def run_global_threshold(input_file: str,
                         output_file: str,
                         col_index: int   = 2,
                         threshold: float = 0.0,
                         direction: str   = 'gte',
                         chunk_size: int  = 50_000,
                         logger=None,
                         progress_callback=None) -> dict:
    """
    Flask entry point.

    Args:
        input_file        : path to input file (any separator, 3+ columns)
        output_file       : path to output CSV
        col_index         : positional index of numeric column to filter (default 2 = weight)
        threshold         : numeric threshold
        direction         : 'gte' = keep >= threshold | 'lte' = keep <= threshold
        chunk_size        : rows per chunk
        logger            : optional logger
        progress_callback : optional callable(percent)

    Returns:
        stats dict
    """
    f = GlobalThresholdFilter(
        col_index=col_index,
        threshold=threshold,
        direction=direction,
        chunk_size=chunk_size,
        logger=logger,
        progress_callback=progress_callback,
    )
    return f.run(input_file, output_file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Global Threshold Filter")
    parser.add_argument('input_file',  help="Input edge file (any separator, 3+ columns)")
    parser.add_argument('--output',    default=None)
    parser.add_argument('--chunk',     type=int,   default=50_000)
    parser.add_argument('--col',       type=int,   default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--direction', choices=['gte', 'lte'], default=None)
    args = parser.parse_args()

    if args.output is None:
        base, ext   = os.path.splitext(args.input_file)
        args.output = f"{base}_filtered.tsv"

    sep    = _detect_separator(args.input_file)
    sample = _peek(args.input_file, sep, 6)

    if all(v is not None for v in [args.col, args.threshold, args.direction]):
        col_index, threshold, direction = args.col, args.threshold, args.direction
    else:
        col_index, threshold, direction = _ask_user(sample)

    stats = run_global_threshold(
        args.input_file, args.output,
        col_index=col_index,
        threshold=threshold,
        direction=direction,
        chunk_size=args.chunk,
    )

    if stats['success']:
        sym = '>=' if stats['direction'] == 'gte' else '<='
        print(f"\n=== Done ===")
        print(f"Rows in:   {stats['rows_in']}")
        print(f"Rows out:  {stats['rows_out']}  (col[{stats['col_index']}] {sym} {stats['threshold']})")
        print(f"Time:      {stats['execution_time']} s")
        print(f"Output:    {args.output}")
    else:
        print(f"Error: {stats.get('error')}")