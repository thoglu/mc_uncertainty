# mc_uncertainty


Probabilistic treatment of the uncertainty from the finite size of weighted Monte Carlo data

<img src="https://github.com/thoglu/mc_uncertainty/raw/master/img/2sec_small.gif" width="640">

This repository implements a collection of functions defined in https://arxiv.org/abs/1712.01293, in particular the functions
shown in the summary table. The alternative method based on nuisance parameter minimization from https://arxiv.org/abs/1304.0735 is also included. Jupyter notebooks for the Poisson-like and multinomial-like likelihoods demonstrate the usage. The scripts also include some functions which implement different ways to calculate the fourth Lauricella function FD in certain parameter regimes.

**Most important use case:**

Simple replacement of the Poisson PDF/likelihood to check for the influence of too little MC simulation (the `w's` are the weights from individual MC events, while `k` is the observed data count):

![Formula gif not found](https://github.com/thoglu/mc_uncertainty/raw/master/img/finite_poisson.gif)

with the iterative definition

![Formula gif not found](https://github.com/thoglu/mc_uncertainty/raw/master/img/iterative_sum_expl.gif)

It is implemented by `poisson_general_weights_direct(k, weights)` in `poisson.py`.
If some weights are equal, they can be combined in the formula (see eq. 2.22 in https://arxiv.org/abs/1712.01293). The above formula uses the "unique" prior, while. eq. 2.22 shows the general form with different possible prior assumptions. If all weights are equal, there
is a simpler expression that circumvents the sums (see eq. 2.6).

