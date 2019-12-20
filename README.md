# mc_uncertainty


This repository implements functions from https://arxiv.org/abs/1712.01293 (**Probabilistic treatment of the uncertainty from the finite size of weighted Monte Carlo data**) and a follow-up paper https://arxiv.org/abs/1902.08831 (**A unified perspective on modified Poisson likelihoods for limited Monte Carlo data**).

Context:
Limited Monte Carlo data includes a statistical uncertainty which can be taken into account by switching (generalizing) the Poisson distribution with generalized Poisson-gamma mixture (generalized negative binomial) probability distributions. 

This leads to widening of related likelihood scans, as shown in the gif below (red - standard Poisson / green - generalized Poisson-gamma mixture)

<img src="https://github.com/thoglu/mc_uncertainty/raw/master/img/2sec_small.gif" width="640">

The generalized distributions are integrals over approximations of the Compound Poisson distribution - the distribution of the sum of weights - and can be solved analytically. All major approaches in the literature, including Frequentist solutions like the Barlow/Beeston  (1993) or Chirkin (2012) Ansatz, or previous probabilistic approaches by Argüelles et al (2019) can be shown to be doing the same thing under the hood (1902.08831). Implementations of these methods are included in this repository as well.

The unified viewpoint also suggests how to incorporate extra prior information into the likelihood. In situations where the background simulation is limited, this can be crucial, as can be seen in the following coverage plots

<img src="https://github.com/thoglu/mc_uncertainty/raw/master/img/output_small2.gif" width="640">

Three generalized probability distributions are discussed, generalization (2) is seen to perform best in such coverage tests in comparison to all other methods.

Some examples are collected in the Jupyter notebooks. The formulas are partially implemented in c to speed up calculation but can be called via cython. The scripts also include some functions which implement different ways to calculate the fourth Lauricella function FD or the Carlson-R function.

Installation (cython): in folder llh_defs, call "python setup.py build_ext --inplace", if necessary rename .so files to "llh_fast.so" and "poisson_gamma_mixtures.so"

Requirements: The code should work with python 2.X and 3.X and should not require external packages other than scipy/numpy

Questions/Something does not work out of the box: thorsten.gluesenkamp@fau.de


