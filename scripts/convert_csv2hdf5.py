import os
import glob
from canopy_factory import utils


def csv2hdf5(src, dst=None, overwrite=False):
    r"""Convert a CSV file to an HDF5 file.

    Args:
        src (str): Source file.
        dst (str, optional): Destination file.
        overwrite (bool, optional): If True overwrite the existing file.

    """
    if dst is None:
        dst = os.path.splitext(src)[0] + ".hdf5"
    if os.path.isfile(dst):
        print(f"Destination already exists at \"{dst}\"")
    data = utils.read_csv(src, verbose=True, include_units=True)
    utils.write_hdf5(data, dst, verbose=True)
    print(f"Converted \"{src}\" to \"{dst}\"")
    # utils.read_hdf5(dst)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'src', type=str, help="CSV file to convert to HDF5",
        nargs="+",
    )
    parser.add_argument(
        '--dst', type=str,
        help="Path to HDF5 file that should be created",
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help="Overwrite the existing hdf5 file",
    )
    parser.add_argument(
        '--remove-src', action='store_true',
        help="Remove the source file after it is converted",
    )
    args = parser.parse_args()
    if len(args.src) == 1:
        args.src = glob.glob(args.src)[0]
    if args.dst and len(args.src) > 1:
        raise RuntimeError(
            f"Cannot convert {len(args.src)} csv files to a single "
            f"hdf5 file"
        )
    for src in args.src:
        csv2hdf5(src, dst=args.dst, overwrite=args.overwrite)
        if args.remove_src:
            os.remove(src)
