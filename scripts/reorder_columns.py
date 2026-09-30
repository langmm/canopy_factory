# Reorder the columns in sparse geometry CSV files, e.g.
# tests/data/generate/*.csv, so that face_start/face_stop come first.
import os
import glob
import csv
import argparse


DEFAULT_ORDER = ['face_start', 'face_stop', 'plantids', 'componentids', 'Age']


def read_csv_lines(fname):
    r"""Read a geometry CSV file into its header comment and rows.

    Args:
        fname (str): CSV file to read.

    Returns:
        str, optional: The ``# HEADER_JSON: ...`` comment line (without a
            trailing newline), or None if the file does not contain one.
        list: First row (column names).
        list: Remaining rows.

    """
    with open(fname, 'r', newline='') as fd:
        first_line = fd.readline()
        comment = None
        if first_line.startswith('# HEADER_JSON: '):
            comment = first_line.rstrip('\n')
        else:
            fd.seek(0)
        reader = csv.reader(fd)
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f'\"{fname}\" does not contain any columns')
    return comment, rows[0], rows[1:]


def column_order(columns, order=None):
    r"""Determine the new order of columns.

    Any column named in ``order`` is placed first, in the order given.
    Columns that are not named in ``order`` are appended afterwards in
    their original relative order.

    Args:
        columns (list): Existing column names.
        order (list, optional): Desired leading column names. Defaults to
            :data:`DEFAULT_ORDER`.

    Returns:
        list: New column names.

    """
    if order is None:
        order = DEFAULT_ORDER
    leading = [k for k in order if k in columns]
    trailing = [k for k in columns if k not in order]
    return leading + trailing


def reorder_columns(src, order=None, dst=None, suffix=None, overwrite=False,
                    remove_src=False):
    r"""Reorder the columns of a geometry CSV file.

    Args:
        src (str): CSV file to reorder.
        order (list, optional): Desired leading column names. Defaults to
            :data:`DEFAULT_ORDER`.
        dst (str, optional): Destination file. Defaults to ``src``
            (in-place) unless ``suffix`` is provided.
        suffix (str, optional): If provided (and ``dst`` is not), the
            destination is the source name with ``suffix`` inserted before
            the extension.
        overwrite (bool, optional): If True, overwrite an existing
            destination file.
        remove_src (bool, optional): If True, remove the source file
            after the columns are reordered.

    Returns:
        str: Path of the file that was written.

    """
    if dst is None and suffix is not None:
        base, ext = os.path.splitext(src)
        dst = f'{base}{suffix}{ext or ".csv"}'
    if dst is None:
        dst = src
    if dst != src and os.path.isfile(dst) and not overwrite:
        raise FileExistsError(
            f'Destination already exists at \"{dst}\" (use --overwrite)')
    comment, columns, rows = read_csv_lines(src)
    new_columns = column_order(columns, order=order)
    if new_columns == columns:
        print(f'\"{src}\" columns already in order')
        return src
    index = [columns.index(k) for k in new_columns]
    with open(dst, 'w', newline='') as fd:
        if comment:
            fd.write(comment + '\n')
        writer = csv.writer(fd, lineterminator='\n')
        writer.writerow(new_columns)
        for row in rows:
            writer.writerow([row[i] for i in index])
    if remove_src and dst != src:
        os.remove(src)
    return dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Reorder the columns of geometry CSV files (e.g. '
            'tests/data/generate/*.csv) so that face_start and face_stop '
            'come first. The "# HEADER_JSON:" comment line is preserved '
            'verbatim and values are copied unchanged.'
        ),
    )
    parser.add_argument(
        'src', type=str, nargs='+',
        help='Geometry CSV file(s) to reorder (globs allowed)',
    )
    parser.add_argument(
        '--order', type=str, default=','.join(DEFAULT_ORDER),
        help='Comma-separated list of columns that should come first '
             f'(default: "{",".join(DEFAULT_ORDER)}")',
    )
    parser.add_argument(
        '--dst', type=str,
        help='Path of the reordered CSV file that should be created. Only '
             'valid when a single source file is provided.',
    )
    parser.add_argument(
        '--suffix', type=str, default=None,
        help='Suffix added to the source name to build the destination '
             'file name. By default files are reordered in-place.',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite the destination file if it already exists',
    )
    parser.add_argument(
        '--remove-src', action='store_true',
        help='Remove the source file after it is reordered',
    )
    parser.add_argument(
        '--inplace', action='store_true',
        help='Save the dense file to the same location',
    )
    args = parser.parse_args()
    order = [k.strip() for k in args.order.split(',') if k.strip()]
    if len(args.src) == 1:
        args.src = glob.glob(args.src[0]) or args.src
    if args.dst and len(args.src) > 1:
        raise RuntimeError(
            f'Cannot reorder {len(args.src)} CSV files to a single '
            f'destination file'
        )
    if args.inplace:
        args.overwrite = True
        args.remove_src = False
        args.suffix = ''
    for src in args.src:
        if args.inplace:
            args.dst = src
        dst = reorder_columns(
            src, order=order, dst=args.dst, suffix=args.suffix,
            overwrite=args.overwrite, remove_src=args.remove_src,
        )
        print(f'Reordered \"{src}\" -> \"{dst}\"')
