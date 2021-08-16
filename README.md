# SAME-ECOS
Spectrum analysis for multiple exponentials via experimental condition oriented simulation (https://arxiv.org/abs/2009.06761 to be updated)

## Introduction to SAME-ECOS
- A data-driven analysis workflow for multi-echo relaxation data acquired in magnetic resonance experiments (e.g. myelin water imaging sequences).
- Decompose each imaging voxel's T2 relaxation data into a T2 spectrum.
- Developed based on resolution limit and machine learning neural network algorithms.
- Tailored for different MR experimental conditions such as SNR.

## How does SAME-ECOS work
The SAME-ECOS workflow takes 4 steps: **simulate, train, test, and deploy.**
1. **Simulate** sufficient ground truth training examples of random T2 spectra and their MR signals (obeying the T2 resolution limit and experimental conditions)
2. **Train** a neural network model to learn the mapping between the simulated decay signals and the ground truth spectra
3. **Test** the trained model using customized tests (e.g. compare with baseline method) and adjust Step 1 & 2 until obtaining satisfactory test results 
4. **Deploy** the trained model to experimental data and get T2 spectrum for each imaging voxel

## What are the files in this repository
This Repository provides one specific example (in-vivo 32-echo sequence) as a paradigm to demonstrate the usage of SAME-ECOS. The following files can be downloaded:
- *SAME_ECOS_functions.py* contains all the functions that are required by the SAME-ECOS workflow. 
- *example_usage.ipynb* contains the SAME-ECOS workflow. Change the variable default values accordingly based on experimental conditions (e.g. SNR range, T2 range, echo times, flip angle etc.)
- *EPG_decay_library_32echo.mat* is a pre-computed library for the 32-echo spin echo decay sequence using extended phase graph (EPG) algorithm. Using a pre-computed EPG library is more efficient, compared with invoking the EPG functions at every simulation realization.
- *NN_model_example.h5* is the trained model that takes 32-echo input data and output a T2 spectrum depicted by 40 basis t2s.

In general, we recommend the users to use their own datasets to train their models adaptively instead of using the example trained model provided in this repository, although it might work as well.

## Package dependencies
The following packages need to be installed and imported properly:
- Numpy
- Scipy
- Tensorflow
- Keras
- SKlearn

File *package-list.txt* contains the full list of all packages installed in this project environment.

## Resource and reference
- Conventional analysis method non-negative least squares (NNLS) and the extended phase graph (EPG) algorithm can be requested at this URL: https://mriresearch.med.ubc.ca/news-projects/myelin-water-fraction/
- Please reference this paper: https://arxiv.org/abs/2009.06761 (to be updated)
