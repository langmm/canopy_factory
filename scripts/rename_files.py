import os
import glob
import shutil
import subprocess

root_dir = os.path.dirname(os.path.dirname(__file__))
output_dir = os.path.join(root_dir, 'output')
test_dir = os.path.join(root_dir, 'tests', 'data')
package_dir = os.path.join(root_dir, 'canopy_factory')

replacements = [
    ('B73_WT', 'B73_WT_2024'),
    ('B73_rdla', 'B73_rdla_2024'),
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
    'generate': '.obj',
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

for base_directory in base_directories:
    use_git = base_directory in [test_dir, package_dir]
    for directory, ext in directories.items():
        if not os.path.isdir(os.path.join(base_directory, directory)):
            continue
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

# traces_dir = os.path.join(output_dir, 'traces')

# for f in glob.glob(os.path.join(traces_dir, '*_periodic.csv')):
#     shutil.move(f, f.replace('_periodic.csv', '_periodic1.csv'))
