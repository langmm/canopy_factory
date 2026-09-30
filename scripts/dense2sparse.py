# tests/data/generate 14MB before 12MB after (16MB with individual plants saved)
# output/generate 4.7GB before 3.6GB after
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd


def read_header(fname):
    r"""Read the header comment from a geometry CSV file.

    Args:
        fname (str): CSV file to read.

    Returns:
        str, optional: The ``# HEADER_JSON: ...`` comment line, or None if
            the file does not contain one.

    """
    with open(fname, 'r') as fd:
        first_line = fd.readline().rstrip('\n')
    if first_line.startswith('# HEADER_JSON: '):
        return first_line
    return None


def dense2sparse(src, dst=None, suffix='_sparse', overwrite=False,
                 remove_src=False, component_names=True):
    r"""Convert dense geometry CSV data to a sparse representation.

    The dense representation contains one row per mesh face, repeating
    the plant ID, component ID and any component data for each face. The
    sparse representation collapses consecutive faces with identical
    values into a single row that records the inclusive range of face
    IDs (``face_start`` to ``face_stop``) that belong to that component.

    Args:
        src (str): Dense CSV file.
        dst (str, optional): Destination file. Defaults to the source
            file with ``suffix`` inserted before the extension.
        suffix (str, optional): Suffix used to build the default
            destination file name.
        overwrite (bool, optional): If True, overwrite an existing
            destination file.
        inplace (bool, optional): If True, write the updated file to the
            same location.
        remove_src (bool, optional): If True, remove the source file
            after it is converted.
        component_names (bool, optional): If True, add a ``component``
            column containing the name of the component (looked up from
            the ``component_order`` in the file header) for each range.

    Returns:
        str: Path of the file that was written.

    """
    if dst is None:
        base, ext = os.path.splitext(src)
        dst = f'{base}{suffix}{ext or ".csv"}'
    if os.path.isfile(dst) and not overwrite:
        raise FileExistsError(
            f'Destination already exists at "{dst}" (use --overwrite)')
    header = read_header(src)
    df = pd.read_csv(src, comment='#')
    if 'face_start' in df:
        if src == dst:
            print(f'\"{src}\" already converted')
            return
        raise RuntimeError
    # Find runs of consecutive rows that are identical. Missing values
    # are treated as equal so that they do not split a run.
    keys = df.astype(object).where(df.notna(), '\x00NA')
    changed = keys.ne(keys.shift()).any(axis=1)
    if len(df):
        changed.iloc[0] = True
    starts = df.index[changed].to_numpy()
    stops = np.append(starts[1:], len(df)) - 1
    out = df.iloc[starts].reset_index(drop=True)
    out['face_start'] = starts
    out['face_stop'] = stops
    if component_names and header and 'componentids' in out:
        order = json.loads(
            header.split('# HEADER_JSON: ')[-1]).get('component_order')
        if order:
            pos = list(out.columns).index('componentids') + 1
            out.insert(pos, 'component',
                       [order[i] for i in out['componentids']])
    if header:
        with open(dst, 'w') as fd:
            fd.write(header + '\n')
        out.to_csv(dst, index=False, mode='a')
    else:
        out.to_csv(dst, index=False)
    if remove_src:
        os.remove(src)
    return dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Convert the dense per-face data in generated geometry CSV '
            'files to a sparse representation that records the range of '
            'face IDs corresponding to each component instance.'
        ),
    )
    parser.add_argument(
        'src', type=str, nargs='+',
        help='Dense geometry CSV file(s) to convert (globs allowed)',
    )
    parser.add_argument(
        '--dst', type=str,
        help='Path of the sparse CSV file that should be created. Only '
             'valid when a single source file is provided.',
    )
    parser.add_argument(
        '--suffix', type=str, default='_sparse',
        help='Suffix added to the source name to build the destination '
             'file name (default: "_sparse")',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite the destination file if it already exists',
    )
    parser.add_argument(
        '--remove-src', action='store_true',
        help='Remove the source file after it is converted',
    )
    parser.add_argument(
        '--inplace', action='store_true',
        help='Save the dense file to the same location',
    )
    parser.add_argument(
        '--component-names', action='store_true',
        help='Add a column with the name of each component',
    )
    args = parser.parse_args()
    if len(args.src) == 1:
        args.src = glob.glob(args.src[0]) or args.src
    if args.dst and len(args.src) > 1:
        raise RuntimeError(
            f'Cannot convert {len(args.src)} CSV files to a single '
            f'destination file'
        )
    if args.inplace:
        args.overwrite = True
        args.remove_src = False
        args.suffix = ''
    for src in args.src:
        if args.inplace:
            args.dst = src
        dst = dense2sparse(
            src, dst=args.dst, suffix=args.suffix,
            overwrite=args.overwrite, remove_src=args.remove_src,
            component_names=args.component_names,
        )
        print(f'Converted "{src}" to "{dst}"')
