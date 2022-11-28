# Generate a package which can be used throughout the workspace

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['image_utils'],  # packages=['package1', 'package2, ...]
    package_dir={'': 'src'}
)
setup(**d)