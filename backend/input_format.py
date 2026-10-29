"""
Shared input handling for PRED-BiPN.

Every uploaded edge list is rewritten once (normalize_input) into a canonical
tab-separated file whose first line is always a header:

    source<TAB>target[<TAB>weight]

so the method modules no longer have to guess whether row 1 is data.
"""
import re

import pandas as pd

HEADER_KEYWORDS = {
    'source', 'target', 'from', 'to', 'node', 'node1', 'node2',
    'gene', 'genes', 'gene1', 'gene2', 'disease', 'diseases',
    'src', 'dst', 'id', 'name', 'weight', 'score', 'value',
    'col1', 'col2', 'edge', 'index',
}


def detect_sep(path):
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()
    for sep in ('\t', ',', ';'):
        if sep in line:
            return sep
    return r'\s+'


def _is_num(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def has_header_row(path, sep=None):
    """Row 1 is a header iff both leading cells are exact header keywords
    (so real names like 'ALZHEIMER_DISEASE' survive), or column 3 flips
    non-numeric -> numeric between rows 1 and 2 (weighted files)."""
    # Keyword heuristic: a headerless file whose first row really is
    # e.g. "gene<TAB>disease" is misread. The UI 'has_header' option overrides it.
    sep = sep or detect_sep(path)
    splitter = None if sep == r'\s+' else sep
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline().strip()
        second = f.readline().strip()
    parts = first.split(splitter)
    if len(parts) < 2:
        return False
    if parts[0].strip().lower() in HEADER_KEYWORDS and parts[1].strip().lower() in HEADER_KEYWORDS:
        return True
    p2 = second.split(splitter) if second else []
    return len(parts) > 2 and len(p2) > 2 and not _is_num(parts[2]) and _is_num(p2[2])


def normalize_input(src, dst, has_header='auto', logger=None):
    """Rewrite src as a canonical TSV with a header row. Returns a stats dict.

    has_header: 'auto' (detect), or a yes/no value from the interface.
    Node names are stripped of surrounding whitespace.
    """
    sep = detect_sep(src)
    flag = str(has_header).strip().lower()
    header = has_header_row(src, sep) if flag == 'auto' else flag in ('true', '1', 'yes')
    df = pd.read_csv(src, sep=sep, header=None, skiprows=1 if header else 0, dtype=str,
                     engine='python', skipinitialspace=True, keep_default_na=False)
    if df.shape[1] < 2:
        raise ValueError('The input file must have at least two columns (source, target[, weight]).')
    # extra columns are kept (Global Threshold can filter on any column)
    df.columns = (['source', 'target', 'weight'] + [f'col_{i}' for i in range(3, df.shape[1])])[:df.shape[1]]
    df['source'] = df['source'].str.strip()
    df['target'] = df['target'].str.strip()
    df.to_csv(dst, sep='\t', index=False)

    # names that differ only by trailing '_' or spaces (e.g. X_OF_BREAST vs X_OF_BREAST_)
    near_dups = 0
    for col in ('source', 'target'):
        names = pd.Series(df[col].unique())
        near_dups += int(names.str.replace(r'[_\s]+$', '', regex=True).duplicated().sum())
    if logger:
        logger.info(f"Input: {len(df)} rows, {df.shape[1]} columns, header row {'found' if header else 'not found'}")
        if near_dups:
            logger.warning(f"{near_dups} node names differ from another name only by trailing '_' or spaces; "
                           f"they are treated as different nodes")
    return {'rows': len(df), 'columns': df.shape[1], 'header': bool(header), 'near_duplicates': near_dups}


def is_projection(path, sep=None):
    """True when columns 1 and 2 hold mostly the same node set (a one-mode
    projection), False for a bipartite edge list."""
    sep = sep or detect_sep(path)
    df = pd.read_csv(path, sep=sep, dtype=str, usecols=[0, 1], engine='python', keep_default_na=False)
    a, b = set(df.iloc[:, 0]), set(df.iloc[:, 1])
    return len(a & b) > 0.5 * min(len(a), len(b))


if __name__ == '__main__':
    import os
    import tempfile
    d = tempfile.mkdtemp()
    p, q = os.path.join(d, 'in.txt'), os.path.join(d, 'out.tsv')
    with open(p, 'w') as f:
        f.write("A2M\tALZHEIMER'S_DISEASE\nA2M\tCANCER\nB\tCANCER_\n")
    assert not has_header_row(p)
    s = normalize_input(p, q)
    assert s['rows'] == 3 and not s['header'] and s['near_duplicates'] == 1
    assert has_header_row(q) and open(q).readline() == 'source\ttarget\n'
    with open(p, 'w') as f:
        f.write("a,b,w\nx,y,1.5\n")
    assert has_header_row(p) and normalize_input(p, q)['rows'] == 1
    assert normalize_input(p, q, has_header='false')['rows'] == 2
    with open(p, 'w') as f:
        f.write("a\tb\t1\t7\n")
    normalize_input(p, q, has_header='false')
    assert open(q).readline() == 'source\ttarget\tweight\tcol_3\n'
    with open(p, 'w') as f:
        f.write("source\ttarget\tweight\nA\tB\t1\nB\tA\t2\n")
    assert is_projection(p)
    print('input_format self-check OK')
