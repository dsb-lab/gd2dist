#!/bin/bash

pip uninstall scBayesDeconv
# python -m build
# pip install --extra-index-url https://pypi.org/simple /home/gabriel/Documents/PhD/Projects/Flow_cytometry/scBayesDec/dist/scBayesDeconv-0.1.tar.gz

# #Test
# rm dist/scBayesDeconv-0.1-cp39-*
# twine upload -u gatocor -p Pythonpypi.91299 --repository testpypi dist/*    
# pip install --no-cache --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple scBayesDeconv  

#Real
rm dist/scBayesDeconv-0.1-cp39-*
twine upload -u gatocor -p Pythonpypi.91299 --repository pypi dist/*    
pip install --no-cache --index-url https://pypi.org/simple/ --extra-index-url https://pypi.org/simple scBayesDeconv  