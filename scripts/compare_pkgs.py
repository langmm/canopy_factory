import os
import glob
import shutil
import subprocess

root_dir = os.path.dirname(os.path.dirname(__file__))

def process_file(src, dst=None):
    if dst is None:
        dst = '_pruned'.join(os.path.splitext(src))
    lines = []
    with open(src, 'r') as fd:
        for line in fd.readlines():
            line = '\t'.join(line.split()[:2])
            lines.append(line)
    with open(dst, 'w') as fd:
        fd.write('\n'.join(lines))
    return dst


def compare_files(f1, f2, fdiff):
    f1p = process_file(f1)
    f2p = process_file(f2)
    print(f1p, f2p)
    subprocess.check_output(
        f'diff {f1p} {f2p} &> {fdiff}',
        shell=True
    )


# f1 = 'macos_pkgs.txt'
# # f2 = 'ubuntu_pkgs.txt'
# f2 = 'windows_pkgs.txt'
# fdiff = 'pkgs_windows.diff'
# compare_files(f1, f2, fdiff)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'f1', type=str, help="First file for comparison"
    )
    parser.add_argument(
        'f2', type=str, help="Second file for comparison"
    )
    parser.add_argument(
        '--fdiff', type=str, help="Diff output",
    )
    args = parser.parse_args()
    if not args.fdiff:
        name1 = os.path.basename(args.f1).split('_')[0]
        name2 = os.path.basename(args.f2).split('_')[0]
        args.fdiff = f'{name1}_{name2}.diff'
    compare_files(args.f1, args.f2, args.fdiff)
