"""
Hypergraph Projection with Automatic Column Detection
Handles any column names and separators automatically
"""

import pandas as pd
import psutil
from collections import defaultdict
from io import StringIO
import os
import re

class Hypergraph:
    def __init__(self, input_file, chunk_size=50000):
        """
        Initialize Hypergraph processor.

        Parameters
        ----------
        input_file : str
            Path to input file
        chunk_size : int, optional (default=50000)
            Number of rows to process at once
        """
        self.input_file = input_file
        self.chunk_size = chunk_size
        self.node1_to_node2 = defaultdict(set)
        self.node2_to_node1 = defaultdict(set)

        self.detected_sep = None
        self.detected_col1 = None
        self.detected_col2 = None
        self.has_header = None
        self.original_col1_name = None
        self.original_col2_name = None

    def _detect_separator(self):
        """Auto-detect the separator in the input file."""
        with open(self.input_file, 'r') as f:
            first_line = f.readline().strip()

            separators = {
                '\t': first_line.count('\t'),
                ',': first_line.count(','),
                ';': first_line.count(';'),
                ' ': first_line.count(' ')
            }

            separators = {k: v for k, v in separators.items() if v > 0}

            if not separators:
                self.detected_sep = '\t'
                print(f"  No separator detected, using default: TAB")
            else:
                self.detected_sep = max(separators, key=lambda k: separators[k])

            separator_names = {'\t': 'TAB', ',': 'COMMA', ';': 'SEMICOLON', ' ': 'SPACE'}
            print(f"  Detected separator: {separator_names.get(self.detected_sep, self.detected_sep)}")

            return self.detected_sep

    def _detect_columns_by_name(self, df_sample):
        """
        Detect columns by checking header names if they exist.
        """
        columns = df_sample.columns.tolist()

        source_patterns = [
            'source', 'from', 'node1', 'gene1', 'gene', 'protein',
            'col1', 'column1', 'left', 'row', 'start', 'id1', 'id_a',
            'disease', 'phenotype', 'trait', 'condition', 'type'
        ]

        target_patterns = [
            'target', 'to', 'node2', 'gene2', 'disease2', 'protein2',
            'col2', 'column2', 'right', 'end', 'id2', 'id_b',
            'gene', 'protein', 'symbol', 'name'
        ]

        source_col = None
        target_col = None
        source_name = None
        target_name = None

        for col in columns:
            col_lower = str(col).lower().strip()
            if col_lower in source_patterns:
                source_col = col
                source_name = col
                print(f"  Found source column by name: '{col}'")
            if col_lower in target_patterns:
                target_col = col
                target_name = col
                print(f"  Found target column by name: '{col}'")

        if source_col is None or target_col is None:
            df_data = pd.read_csv(self.input_file, sep=self.detected_sep,
                                 header=0 if self.has_header else None,
                                 nrows=10)

            for i, col in enumerate(columns if self.has_header else range(len(df_data.columns))):
                col_name = col if self.has_header else f"Column_{i}"
                col_data = df_data.iloc[:, i] if self.has_header else df_data.iloc[:, i]

                is_string = col_data.dtype == 'object'

                if is_string:
                    sample_val = str(col_data.dropna().iloc[0]) if len(col_data.dropna()) > 0 else ""

                    if len(sample_val) < 100 and not sample_val.replace('.', '').replace('-', '').isdigit():
                        if source_col is None:
                            source_col = col_name
                            source_name = col_name
                            print(f"  Guessed source column: '{col_name}' (contains identifiers)")
                        elif target_col is None and col_name != source_col:
                            target_col = col_name
                            target_name = col_name
                            print(f"  Guessed target column: '{col_name}' (contains identifiers)")

        if source_col is None:
            source_col = columns[0] if self.has_header else 0
            source_name = source_col
            print(f"  Using first column as source: '{source_col}'")

        if target_col is None:
            target_col = columns[1] if self.has_header and len(columns) > 1 else 1
            target_name = target_col
            print(f"  Using second column as target: '{target_col}'")

        return source_col, target_col, source_name, target_name

    def _detect_columns_by_data(self, file_path, sep):
        """
        Detect columns by reading the actual data values (for files without header).
        """
        df_sample = pd.read_csv(file_path, sep=sep, nrows=10, header=None)

        source_col = 0
        target_col = 1
        source_name = "Column_0"
        target_name = "Column_1"

        string_cols = []
        for i in range(len(df_sample.columns)):
            col_data = df_sample.iloc[:, i]
            if col_data.dtype == 'object':
                string_cols.append(i)

        if len(string_cols) >= 2:
            source_col = string_cols[0]
            target_col = string_cols[1]
            source_name = f"Column_{source_col}"
            target_name = f"Column_{target_col}"
            print(f"  Detected source column: column {source_col} (contains strings)")
            print(f"  Detected target column: column {target_col} (contains strings)")
        elif len(string_cols) == 1:
            source_col = string_cols[0]
            source_name = f"Column_{source_col}"
            target_col = source_col + 1 if source_col + 1 < len(df_sample.columns) else 0
            target_name = f"Column_{target_col}"
            print(f"  Detected source column: column {source_col} (contains strings)")
            print(f"  Using column {target_col} as target")
        else:
            print(f"  Using columns 0 and 1 as source and target")

        return source_col, target_col, source_name, target_name

    def run(self, col1=None, col2=None, sep=None, header='infer', names=None):
        """
        Read file and build mappings with automatic column detection.

        Parameters
        ----------
        col1 : str or int, optional (default=None)
            Column name or index for first partition (source/left)
            If None, auto-detected
        col2 : str or int, optional (default=None)
            Column name or index for second partition (target/right)
            If None, auto-detected
        sep : str, optional (default=None)
            Separator (auto-detected if None)
        header : int or 'infer', optional (default='infer')
            Row number to use as header (0 for first row, None for no header)
        names : list, optional (default=None)
            Column names if no header

        Returns
        -------
        tuple
            (content1, content2) - StringIO contents for both projections
        """
        total_rows = 0

        if sep is None:
            self._detect_separator()
        else:
            self.detected_sep = sep
            print(f"  Using specified separator: {sep}")

        if header == 'infer':
            try:
                with open(self.input_file, 'r') as f:
                    first_line = f.readline().strip()
                    parts = first_line.split(self.detected_sep)
                    has_words = any(p.isalpha() and len(p) > 0 for p in parts)
                    self.has_header = has_words
                    print(f"  Header detected: {self.has_header}")
            except:
                self.has_header = False
                print(f"  Header detection failed, assuming no header")
        else:
            self.has_header = header is not None
            print(f"  Header setting: {self.has_header}")

        try:
            sample = pd.read_csv(self.input_file, sep=self.detected_sep,
                                header=0 if self.has_header else None,
                                nrows=5, names=names)

            print(f"  Sample columns: {list(sample.columns)}")

            if col1 is None or col2 is None:
                if self.has_header:
                    detected_col1, detected_col2, col1_name, col2_name = self._detect_columns_by_name(sample)
                else:
                    detected_col1, detected_col2, col1_name, col2_name = self._detect_columns_by_data(self.input_file, self.detected_sep)

                if col1 is None:
                    col1 = detected_col1
                    self.original_col1_name = col1_name
                    print(f"  Auto-detected column 1: '{col1}'")

                if col2 is None:
                    col2 = detected_col2
                    self.original_col2_name = col2_name
                    print(f"  Auto-detected column 2: '{col2}'")

            col1_idx = col1
            col2_idx = col2

            if isinstance(col1, str):
                if col1 in sample.columns:
                    col1_idx = sample.columns.get_loc(col1)
                    self.original_col1_name = col1
                    print(f"  Column 1 '{col1}' found at index {col1_idx}")
                else:
                    try:
                        col1_idx = int(col1)
                        self.original_col1_name = f"Column_{col1}"
                        print(f"  Using column 1 index: {col1_idx}")
                    except:
                        raise ValueError(f"Column 1 '{col1}' not found. Available columns: {list(sample.columns)}")

            if isinstance(col2, str):
                if col2 in sample.columns:
                    col2_idx = sample.columns.get_loc(col2)
                    self.original_col2_name = col2
                    print(f"  Column 2 '{col2}' found at index {col2_idx}")
                else:
                    try:
                        col2_idx = int(col2)
                        self.original_col2_name = f"Column_{col2}"
                        print(f"  Using column 2 index: {col2_idx}")
                    except:
                        raise ValueError(f"Column 2 '{col2}' not found. Available columns: {list(sample.columns)}")

        except Exception as e:
            print(f"Warning: Could not read sample for column detection: {e}")
            col1_idx = 0 if col1 is None else (0 if isinstance(col1, str) else col1)
            col2_idx = 1 if col2 is None else (1 if isinstance(col2, str) else col2)
            print(f"  Using fallback: column indices {col1_idx} and {col2_idx}")

        try:
            chunk_iter = pd.read_csv(
                self.input_file,
                sep=self.detected_sep,
                header=0 if self.has_header else None,
                names=names,
                chunksize=self.chunk_size,
                low_memory=False
            )

            for chunk_num, chunk in enumerate(chunk_iter, 1):
                total_rows += len(chunk)

                if isinstance(col1_idx, int):
                    col1_data = chunk.iloc[:, col1_idx]
                else:
                    col1_data = chunk[col1_idx]

                if isinstance(col2_idx, int):
                    col2_data = chunk.iloc[:, col2_idx]
                else:
                    col2_data = chunk[col2_idx]

                for val1, val2 in zip(col1_data, col2_data):
                    if pd.notna(val1) and pd.notna(val2):
                        val1_str = str(val1).strip()
                        val2_str = str(val2).strip()

                        if val1_str and val2_str and val1_str != 'nan' and val2_str != 'nan':
                            self.node1_to_node2[val1_str].add(val2_str)
                            self.node2_to_node1[val2_str].add(val1_str)

                mem_usage = psutil.virtual_memory().percent
                if chunk_num % 10 == 0:
                    print(f"  Processed {total_rows:,} rows... (Memory: {mem_usage}%)")

        except Exception as e:
            print(f"Error while reading file: {e}")
            import traceback
            traceback.print_exc()
            return "", ""

        buffer1 = StringIO()
        for node1, nodes2 in self.node1_to_node2.items():
            buffer1.write(f"{node1}\t{','.join(sorted(nodes2))}\n")
        content1 = buffer1.getvalue()

        buffer2 = StringIO()
        for node2, nodes1 in self.node2_to_node1.items():
            buffer2.write(f"{node2}\t{','.join(sorted(nodes1))}\n")
        content2 = buffer2.getvalue()

        print(f"\nResults:")
        print(f"  Unique nodes in column 1: {len(self.node1_to_node2):,}")
        print(f"  Unique nodes in column 2: {len(self.node2_to_node1):,}")
        print(f"  Total edges processed: {total_rows:,}")
        print(f"  Unique connections: {sum(len(v) for v in self.node1_to_node2.values()):,}")

        return content1, content2

    def get_column_info(self):
        """
        Get detected column information.

        Returns
        -------
        dict
            Information about detected columns
        """
        return {
            'separator': self.detected_sep,
            'has_header': self.has_header,
            'source_column': self.original_col1_name,
            'target_column': self.original_col2_name,
            'source_column_index': self._get_col_index(self.original_col1_name),
            'target_column_index': self._get_col_index(self.original_col2_name),
            'unique_nodes_col1': len(self.node1_to_node2),
            'unique_nodes_col2': len(self.node2_to_node1),
            'total_edges': sum(len(v) for v in self.node1_to_node2.values())
        }

    def _get_col_index(self, col_name):
        """Helper to get column index if possible."""
        try:
            sample = pd.read_csv(self.input_file, sep=self.detected_sep,
                                header=0 if self.has_header else None, nrows=1)
            if col_name in sample.columns:
                return sample.columns.get_loc(col_name)
        except:
            pass
        return None

    def run_old_compatibility(self):
        """Old method for backward compatibility."""
        return self.run(col1=0, col2=1, sep='\t', header='infer')

def create_hypergraph_extractor(input_file, chunk_size=50000):
    """
    Factory function to create a Hypergraph instance.

    Parameters
    ----------
    input_file : str
        Path to input file
    chunk_size : int, optional (default=50000)
        Number of rows to process at once

    Returns
    -------
    Hypergraph
    """
    return Hypergraph(input_file, chunk_size=chunk_size)

if __name__ == "__main__":
    hg = Hypergraph("test_data.tsv")

    content1, content2 = hg.run()

    info = hg.get_column_info()
    print(f"\nColumn Info: {info}")