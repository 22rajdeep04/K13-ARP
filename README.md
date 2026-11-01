# K13-ARP: Kelch13 Artemisinin Resistance Predictor
**A tool to predict and classify resistance in _PfKelch13_ mutant strains from sequence and structure derived descriptor**
Implemented code and data collected for the study: **"Leave-One-Out-Cross-Validation Based Machine Learning Algorithms for Efficient, Scalable, and Robust Prediction of Artemisinin Resistance in __Plasmodium falciparum__ Kelch13 Mutant Strains"**, Malaria Journal, DOI: , 2026

**Authors: Das, Rajdeep; Dutta, Smritikana; Bhattacharjee, Souvik**
## Overview
Artemisinin (ART) and Artemisinin Combination Therapy (ACT) are the first line of defence against Malaria caused by _P. falciparum_, that are activated by Heme residues bound to an essential protein Kelch13 (K13). Recent surveys and studies have shown a high correlation between K13 Mutations and ART-Resistance. K13-ARP is a comprehensive tool that efficiently classifies such mutants for their degree of resistance. 
This repository provides the code and data used to build K13-ARP, that runs a simple and interpretable Logistic Regression and Linear Regression to predit ART-R from the mutation name alone. The models run entirely on a set of ten physicochemical and structural descriptors alone.

|**TASK**| **TARGET**| **MODEL**|
| :---: | :---: | :---: |
|Classification | ART-R Vs. ART-S| L2-Regularised Logistic Regression|
|Regression | Predict Parasite Clearance Half-life | Lasso Regression|

Due to the small sample size (n=40), we applied the models using a LOOCV (Leave one out cross validation) framework, to efficiently validate the performance and prevent overfitting on the small dataset. 

### Descriptors
|**Feature**|**Description**|
|:---:|:---:|
|Propensity| Chou-Fasman's secondary structure prediction|
|Isoelectric Point| pH at which a peptide has a neutral charge|
|GRAVY| Quantifies Hyrdropathicity|
|Aliphatic Index|Quantifies the aliphatic nature of the local environment|
|Instability Index|Quantifies instability|
|Steric Hindrance| Clashes in the Amino acid side chains|
|Solvent Accessible Surface Area| Per residue surface area|
|Hydrogen Bonds|No of hydrogen bonds in the total 3D structure|
|Bond Dissociation Energy| Thermodynamic stability |
|Root Mean Square Deviation| Structural deviation of mutants from WT|
## Contact
Dr. Souvik Bhattacharjee, Special Centre for Molecular Medicine, Jawaharlal Nehru University, New Delhi, India- 110067 [souvik@jnu.ac.in]. Repository Maintainer: @22rajdeep04.
