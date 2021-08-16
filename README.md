# SAME-ECOS
Spectrum analysis for multiple exponentials via experimental condition oriented simulation (https://arxiv.org/abs/2009.06761 to be updated)

## Introduction to SAME-ECOS
- A data-driven analysis workflow for multi-echo relaxation data acquired in magnetic resonance experiments (e.g. myelin water imaging sequences).
- Decompose each imaging voxel's T2 relaxation data into a T2 spectrum.
- Developed based on resolution limit (please see the **Relevant math** section below) and machine learning neural network algorithms.
- Tailored for different MR experimental conditions such as SNR.

## How does SAME-ECOS work
The SAME-ECOS workflow takes 4 steps: **simulate, train, test, and deploy.**
1. **Simulate** sufficient ground truth training examples of random T2 spectra and their MR signals (obeying the T2 resolution limit and experimental conditions)
2. **Train** a neural network model to learn the mapping between the simulated decay signals and the ground truth spectra
3. **Test** the trained model using customized tests (e.g. compare with baseline method) and adjust Step 1 & 2 until obtaining satisfactory test results 
4. **Deploy** the trained model to experimental data and get T2 spectrum for each imaging voxel

## What is in this Repository
This Repository provides one specific example (in-vivo 32-echo sequence) as a paradigm to demonstrate the usage of SAME-ECOS. The following files can be downloaded:
- *SAME_ECOS_functions.py* contains all the functions that are required by the SAME-ECOS workflow. 
- *example_usage.ipynb* contains the SAME-ECOS workflow. Change the variable default values accordingly based on experimental conditions (e.g. SNR range, T2 range, echo times, flip angle etc.)
- *EPG_decay_library.mat* is a pre-computed library for the 32-echo spin echo decay sequence using extended phase graph (EPG) algorithm. Using a pre-computed EPG library is more efficient, compared with invoking the EPG functions at every simulation realization.
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
- Conventional analysis method non-negative least squares (NNLS) and EPG functions can be requested at this URL: https://mriresearch.med.ubc.ca/news-projects/myelin-water-fraction/
- Please reference this paper: https://arxiv.org/abs/2009.06761 (to be updated)

## Relevant math (if you are interested…)

**Derivation of the T2 resolution limit and the maximum number of resolvable T2 components**:

- Based on the previous information theory works (https://aip.scitation.org/doi/abs/10.1063/1.348634 and https://aip.scitation.org/doi/abs/10.1063/1.1149581)

- In time domain, the MR signal function can be expressed as

$$
f(t)=\int_{0}^{\infty} e^{-\gamma t} g(\gamma) d \gamma
\\t\geq0 \ and \  \gamma\geq0
$$

​	where $g(\gamma)$ is the continuous function of the T2 spectrum distribution.

- Transform the above equation to logarithmic scale by letting $x=\ln t \text { and } y=-\ln \gamma$
  $$
  f\left(e^{x}\right)=\int_{-\infty}^{+\infty} e^{-e^{x-y}} \ g\left(e^{-y}\right) e^{-y} d y
  $$
  This is the convolution of two functions: $e^{-e^{x}}$ and $g\left(e^{-y}\right) e^{-y}$

- Use Fourier transformation and deconvolution:
  $$
  e^{-y} g\left(e^{-y}\right)=F^{-1}\left\{\frac{F\left\{f\left(e^{x}\right)\right\}}{F\left\{e^{-e^{x}}\right\}}\right\}
  $$

- In the case of discrete $g(\gamma)=\sum_{n} a_{n} \delta(\gamma-\gamma_n)$ and $f\left(t\right)=\sum_{n} a_{n} e^{-\gamma_nt}+v(t)$, where $v(t)$ is the noise term, then change the variables to have: $f\left(e^{x}\right)=\sum_{n} a_{n} e^{-e^{x-y_{n}}}+v\left(e^{x}\right)$

- Sub $f(e^x)$ into the above equation and use the shift property of FT (phase shift by $i\omega y_n$) to have
  $$
  \begin{aligned}
  e^{-y} g\left(e^{-y}\right) &=F^{-1}\left\{\frac{\sum_{n} a_{n} e^{i \omega y_{n}}F\{e^{-e^x}\}+F\left\{v\left(e^{x}\right)\right\}}{F\left\{e^{-e^{x}}\right\}}\right\} \\
  &=F^{-1}\left\{\frac{\sum_{n} a_{n} e^{i \omega y_{n}} \Gamma(i \omega)+F\left\{v\left(e^{x}\right)\right\}}{\Gamma(i \omega)}\right\}
  \end{aligned}
  $$

- $F\{e^{-e^x}\}=\int_{-\infty}^{+\infty} e^{-e^{x}} e^{-i\omega x} dx=\int_{0}^{+\infty}e^{-\theta} \theta^{i\omega-1}d\theta=\Gamma(i\omega)$, where $e^x=\theta$ was substituted.

- Only those frequency components that are large compared with the noise term will be properly reconstructed in the inverse Fourier transform. This means that
  $$
  a_{n}^{2}|\Gamma(i \omega)|^{2} \gg \left|F\left\{v\left(e^{x}\right)\right\}\right|^{2}
  $$
  where the phase factor $e^{i\omega y_n}$ is demodulated. 

- A known property of gamma function (proof:https://proofwiki.org/wiki/Modulus_of_Gamma_Function_of_Imaginary_Number):  
  $$
  |\Gamma(i \omega)|^{2}=\frac{\pi}{\omega \sinh(\pi \omega)}
  $$
  And assuming a constant white noise with amplitude $v_0$, then
  $$
  a_{n}^{2} \frac{\pi}{\omega \sinh (\pi \omega)} \gg v_{0}^{2}
  $$

- Therefore, the frequency component has a maximum limit of
  $$
  \frac{\omega_{max}}{\pi}\ {\sinh (\pi \omega_{max})} = \left(\frac{a_n}{v_{0}}\right)^{2}
  $$
  If the signal consists of $M$ exponentials with equal amplitudes $a_0$, then $S=f(t=0)$, $a_0=S/M$, and $SNR=S/v_0$ to give
  $$
  \frac{\omega_{\max }}{\pi} \sinh \left(\pi \omega_{\max }\right)=\left(\frac{\mathrm{SNR}}{M}\right)^{2}
  $$

- In the logarithmic scale, $\omega_{max}$ is the most oscillatory component to construct the T2 spectrum, so adjacent T2 components must have a minimum separation
  $$
  y_n-y_{n-1}=\frac{\pi}{\omega_{max}}
  $$
  Then, in the linear scale
  $$
  \ln \gamma_n-\ln\gamma_{n-1}=\frac{\pi}{\omega_{max}}
  $$
  So the resolution limit 
  $$
  \delta=\frac{\gamma_{n}}{\gamma_{n-1}}=e^{\pi / \omega_{\max }}
  $$
  There are up to M components can be accommodated in the T2 range. So 
  $$
  \delta=\left(T_{\max } / T_{\min }\right)^{1 / M}=e^{\pi / \omega_{\max }}
  $$

- Combine the two equations (derived in the last two bullet points) to have an expression that correlates T~2~ range, SNR, and M resolvable exponentials.
  $$
  \frac{M}{\ln \left(\frac{T_{2}^{\max }}{T_{2}^{\min }}\right)} \times \sinh \left(\frac{\pi^{2} \times M}{\ln \left(\frac{T_{2}^{\max }}{T_{2}^{\min }}\right)}\right)=\left(\frac{S N R}{M}\right)^{2}
  $$

The SAME-ECOS simulation obeys this derived expression.

