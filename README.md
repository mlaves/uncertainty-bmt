# Quantifying the uncertainty of deep learning-based computer-aided diagnosis for patient safety
#### Max-Heinrich Laves, Sontje Ihler, Lüder A. Kahrs, Tobias Ortmaier

In this work, we discuss epistemic uncertainty estimation obtained by Bayesian inference in diagnostic classifiers and show that the prediction uncertainty highly correlates with *goodness* of prediction.
We train the ResNet-18 image classifier on a dataset of 84.484 optical coherence tomography scans showing four different retinal conditions.
Dropout is added before every building block of ResNet, creating an approximation to a Bayesian classifier.
Monte Carlo sampling is applied with dropout at test time for uncertainty estimation.
In Monte Carlo experiments, multiple forward passes are performed to get a posterior distribution of the class labels.
The variance and the entropy of the posterior is used as metrics for uncertainty.
Our results show strong correlation with $ \rho=0.99 $ between prediction uncertainty and prediction error.
Mean uncertainty of incorrectly diagnosed cases was significantly higher than mean uncertainty of correctly diagnosed cases.
Modeling of the prediction uncertainty in computer-aided diagnosis with deep learning yields more reliable results and is therefore expected to increase patient safety.
This can help to transfer such systems into clinical routine and to increase the acceptance of physicians and patients for machine learning in diagnosis.
