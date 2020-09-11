#! /bin/bash

pip uninstall gaussiandeconvolution
./docker.sh
pip install ./dist/*
python ./tests/test_nested.py
#python ./tests/test_mcmc.py