# Analysis present in scBayesDeconv paper.

This example goes over the example analysis performed for the publication of the **scBayesDec** package.

The folder contains:

1. **Paper_analysis.ipynb**: Jupyter notebook where the analysis all the analysis of the paper are done.
2. **requirements.txt**: A txt file with all the required packages intalled in the environment at the time of the execution of the analysis.
3. **auxiliar_metrics.py**: Python functions for quantifying the proximity to a ground truth distribution.
4. **auxiliar_neumman.py**: Python functions with the implementeation of the Neumman deconvolution method.
5. **auxiliar_synthetic_data_distributions.py**: Python functions for generation of the random sampling distributions employed in the generation of the synthetic data.
6. **auxiliar_synthetic_data_pdf.py**: Python functions for generation of the probability density functions employed in the evaluation of the evaluation of the synthetic datasets.
7. **Data** folder: All the datasets employed for the analysis.
8. **Output** folder: All data generated during the execution of the Jupyter Notebook. It is divided in `Plots`, `Tables` and `Samples`. `Samples` are the results of the bayesian simulations and it is already populated for reducing the time of execution. If you want to recompute the results, set `recompute=False` in the header of the notebook.