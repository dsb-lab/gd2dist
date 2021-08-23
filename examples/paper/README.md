# Paper gaussian deconvolution package

This example goes over the example analysis performed for the publication of the **scBayesDec** package.

The folder contains:

1. **Data** folder: All the datasets employed for the analysis.
2. **auxiliar_functions.py**: Python functions for the comparison and performance scoring of the different methods of deconvolution.
3. **synthetic_data_distributions.py**: Python functions for generation of the random sampling distributions employed in the generation of the synthetic data.
4. **synthetic_data_pdf.py**: Python functions for generation of the probability density functions employed in the evaluation of the evaluation of the synthetic datasets.
5. **Paper_analysis.ipynb**: Jupyter notebook where the analysis all the analysis of the paper are done.
6. **requirements.txt**: A txt file with all the packages installed in the environment at the time of the execution of the analysis.
7. **Output** folder: Plots of the analysis as well as tables.
8. **Simulations** folder: Sampled results of the Paper_analysis notebook for the different sections. If you want to recompute the results, set `False`, where necessary.