# Experimental evaluation of ensemble classifiers for imbalance in Big Data

[https://doi.org/10.1016/j.asoc.2021.107447](https://doi.org/10.1016/j.asoc.2021.107447)

Mario Juez-Gil <<mariojg@ubu.es>>, Álvar Arnaiz-González, Juan J. Rodríguez, and César Ignacio García-Osorio.

**Affiliation:**\
Departamento de Ingeniería Informática\
Universidad de Burgos\
[ADMIRABLE Research Group](http://admirable-ubu.es/)\
Burgos\
Spain

## Abstract

Datasets are growing in size and complexity at a pace never seen before, forming ever larger datasets known as Big Data. A common problem for classification, especially in Big Data, is that the numerous examples of the different classes might not be balanced. Some decades ago, imbalanced classification was therefore introduced, to correct the tendency of classifiers that show bias in favor of the majority class and that ignore the minority one. To date, although the number of imbalanced classification methods have increased, they continue to focus on normal-sized datasets and not on the new reality of Big Data. In this paper, in-depth experimentation with ensemble classifiers is conducted in the context of imbalanced Big Data classification, using two popular ensemble families (Bagging and Boosting) and different resampling methods. All the experimentation was launched in Spark clusters, comparing ensemble performance and execution times with statistical test results, including the newest ones based on the Bayesian approach. One very interesting conclusion from the study was that simpler methods applied to unbalanced datasets in the context of Big Data provided better results than complex methods. The additional complexity of some of the sophisticated methods, which appear necessary to process and to reduce imbalance in normal-sized datasets were not effective for imbalanced Big Data.

## Experiments

The experiments are available in [this notebook](experiments.ipynb).

## Aknowledgements

The project leading to these results has received funding from “la Caixa” Foundation, Spain, under agreement LCF/PR/PR18/51130007. This work was supported by the *Junta de Castilla y León, Spain* under project BU055P20 (JCyL/FEDER, UE) co-financed through European Union FEDER funds, and by the *Consejería de Educación of the Junta de Castilla y León and the European Social Fund, Spain* through a pre-doctoral grant (EDU/1100/2017). This material is based upon work supported by Google Cloud, United States .

## Citation policy

Please cite this research as:

```
@ARTICLE{juezgil2021imbalancebd,
title = {Experimental evaluation of ensemble classifiers for imbalance in Big Data},
author = {Juez-Gil, Mario and Arnaiz-Gonz{\'a}lez, {\'A}lvar and Rodr{\'\i}guez, Juan J and Garc{\'\i}a-Osorio, C{\'e}sar},
journal = {Applied Soft Computing},
year = {2021},
month = {sep},
volume = {108},
pages = {107447},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2021.107447},
url = {https://www.sciencedirect.com/science/article/pii/S1568494621003707},
keywords = {Unbalance, Imbalance, Ensemble, Resampling, Big Data, Spark},
publisher = {Elsevier}
}
```