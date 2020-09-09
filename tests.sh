#! /bin/bash

pip uninstall gaussiandeconvolution
./docker.sh
pip install ./dist/gaussiandeconvolution-0.0.1-cp38-cp38-manylinux2010_x86_64.whl
#python ./tests/test_nested.py
python ./tests/test_mcmc.py