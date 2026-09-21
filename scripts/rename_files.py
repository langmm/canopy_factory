import os
import glob
import shutil
import subprocess

root_dir = os.path.dirname(os.path.dirname(__file__))
output_dir = os.path.join(root_dir, 'output')
test_dir = os.path.join(root_dir, 'tests', 'data')
package_dir = os.path.join(root_dir, 'canopy_factory')

replacements = [
    # ('B73_WT', 'B73_WT_2024'),
    # ('B73_rdla', 'B73_rdla_2024'),
    # ('maturity', '60days'),
    ('60days', '66days'),
    # ('WT', 'B73_WT'),
    # ('rdla', 'B73_rdla')
    # ('_periodic', '_periodic1_scene'),
    # ('_periodic1', '_periodic1_scene'),
    # ('_periodic2', '_periodic2_scene'),
]
bookends = [
    ('maize*', '*'),
    # ('*', '_per_plant*'),
    # ('*', '_flux*'),
]
directories = {
    'animate': '.gif',
    'generate': ['.csv', '.obj'],
    'geometryids': '.csv',
    'layout': '.png',
    'match_query': '.json',
    'raytrace': '.csv',
    'raytrace_limits': '.json',
    'render': '.png',
    'render_camera': '.json',
    'totals': '.json',
    'totals_plot': '.png',
    'traced_mesh': '.obj',
    'param': '.json',
}
base_directories = [
    output_dir,
    test_dir,
    package_dir,
]


def do_replace(base_directory, directory, ext, use_git=False):
    if not os.path.isdir(os.path.join(base_directory, directory)):
        return
    if isinstance(ext, list):
        for iext in ext:
            do_replace(base_directory, directory, iext, use_git=use_git)
        return
    for old, new in replacements:
        for prefix, suffix in bookends:
            pattern = os.path.join(base_directory, directory,
                                   prefix + old + suffix + ext)
            for f in glob.glob(pattern):
                if new in f:
                    continue
                if use_git:
                    try:
                        subprocess.check_output(
                            ['git', 'mv', f, f.replace(old, new)],
                            stderr=subprocess.STDOUT,
                        )
                    except subprocess.CalledProcessError as e:
                        if b'not under version control' in e.stdout:
                            shutil.move(f, f.replace(old, new))
                        else:
                            raise
                else:
                    shutil.move(f, f.replace(old, new))


for base_directory in base_directories:
    use_git = base_directory in [test_dir, package_dir]
    for directory, ext in directories.items():
        do_replace(base_directory, directory, ext, use_git=use_git)
