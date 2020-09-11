import sys

try:
    from skbuild import setup
except ImportError:
    print('Please update pip, you need pip 10 or greater,\n'
          ' or you need to install the PEP 518 requirements in pyproject.toml yourself', file=sys.stderr)
    raise
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gd2dist",
    version="0.0.1",
    description="A gaussian deconvolution package",
    author='Gabriel Torregrosa Cortes',
    author_email="g.torregrosa@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=['gd2dist', 'gd2dist/nestedsampler', 'gd2dist/mcmcsampler', 'gd2dist/shared_functions'],
    package_dir={'': 'src'},
    cmake_install_dir='src/gd2dist',
    python_requires = ">=3.5",
    install_requires = ["numpy","scipy","dynesty","pandas"]
)
