#!/bin/bash

pip uninstall scBayesDec
python -m build
pip install --extra-index-url https://pypi.org/simple /home/gabriel/Documents/PhD/Projects/Flow_cytometry/scBayesDec/dist/scBayesDec-0.2.tar.gz 

# rm dist/scBayesDec-0.2-cp39-*
# twine upload -u gatocor -p Pythonpypi.91299 --repository testpypi dist/*    
# pip install --no-cache --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple scBayesDec  