import numpy
import scipy
import lauricella_fd

####### Relevant Poisson generalizations from the paper https://arxiv.org/abs/1712.01293
####### All formulas return the log-likelihood or log-probability

numpy.seterr(divide="warn")

#### single bin or multi bin expression
def poisson_infinite_statistics(k, lambd):
 
    return (-lambd+k*numpy.log(lambd)-scipy.special.gammaln(k+1)).sum()

### equal weight formula with equal weights per bin (eq. 21)
### multi bin expression, one k, one k_mc and one avg_weight item for each bin, all given by an array
def poisson_equal_weights(k,k_mc,avgweights,prior_factor=0.0):
    
    return (scipy.special.gammaln((k+k_mc+prior_factor)) -scipy.special.gammaln(k+1.0)-scipy.special.gammaln(k_mc+prior_factor) + (k_mc+prior_factor)* numpy.log(1.0/avgweights) - (k_mc+k+prior_factor)*numpy.log(1.0+1.0/avgweights)).sum()

### single-bin expression for general weights, i.e. k is a number and weights is an array (eq. 35)
def poisson_general_weights(k, weights, prior=0.0):
    
    ## treat each weight independently without any more checks (see github readme exaxmple)
   
    weight_prefactors=(-(1.0+prior/float(len(weights)))*numpy.log(weights)).sum()

    new_zs=1.+1./weights
        
    new_zs_log=numpy.log(new_zs)
    new_bs=numpy.ones(len(new_zs), dtype=float)
    new_bs+=prior*new_bs/float(len(weights))
    
    res=(-new_bs*new_zs_log).sum()
    
    cs=[res]
    if(k>0):
        
        lambdas=[]
        
        
        new_bs_log=numpy.log(new_bs)
        running_lambda_vec=new_bs_log
        
        for cur_ind in range(k):
            running_lambda_vec-=new_zs_log
            lambdas.append(scipy.misc.logsumexp(running_lambda_vec).sum())
            
            new_cs=scipy.misc.logsumexp( numpy.array(lambdas[::-1])+numpy.array(cs))-numpy.log(cur_ind+1)
            cs.append(new_cs)
       
    return weight_prefactors+cs[-1]


### calculation based on approximate lauricella function or residue at every point... not really needed other than for crosschecks
def poisson_general_weights_outdated(k, weights, lauricella_calc="exact", nthrows=100000, prior_factor=0.0):


    min_weight=min(weights)
    sort_mask=numpy.argsort(weights)
    sorted_weights=weights[sort_mask]
    
    zs=((1.0/min_weight)-(1.0/sorted_weights[1:]))/(1.0+1.0/min_weight)

    tot_mc=len(weights)

    individual_b_prior_factor=prior_factor/float(tot_mc)

    log_W=(1.0+individual_b_prior_factor)*numpy.sum( numpy.log(min_weight)-numpy.log(sorted_weights[1:]))
        
    
    tot_k=k
    log_fac1=-(prior_factor+tot_mc)*numpy.log(min_weight)
    log_fac2=-(tot_mc+tot_k+prior_factor)*numpy.log(1.0+(1.0/min_weight))
    log_fac3=scipy.special.gammaln(tot_k+tot_mc+prior_factor)-scipy.special.gammaln(tot_k+1)-scipy.special.gammaln(tot_mc+prior_factor)
   
    ## sort multiplicities
    item_counter=dict()
    for item in set(zs):
        if(item != 0.0):
            item_counter[item]=zs.tolist().count(item)

    
    new_bs=[]
    new_zs=[]

         
    for item in item_counter.keys():
        new_zs.append(item)
        new_bs.append(float(item_counter[item]))

    new_bs=numpy.array(new_bs)
    new_bs+=new_bs*individual_b_prior_factor

    lauri=None
    if(len(new_zs)==0):
        lauri=0.0 ## Lauricella function is equal to 1, or log equal to 0
    else:
        if(lauricella_calc=="exact"):
            if(prior_factor!=0.0):
                print "Exact form based on contour integral does not support priors!"
                exit(-1)
            lauri=lauricella_fd.log_lauricella_exact_cauchy(tot_k+tot_mc,new_bs,tot_mc,numpy.array(new_zs))
        elif(lauricella_calc=="montecarlo"):
            lauri=lauricella_fd.log_lauricella_dumb_integral_multi(tot_k+tot_mc+prior_factor,new_bs,tot_mc+prior_factor,numpy.array(new_zs), nthrows=nthrows)
        else:
            print "lauricella calc unkonwn"
            return None

    log_res=log_W+log_fac1+log_fac2+log_fac3+lauri
    return log_res


### The next functions involve the series representation of the Poisson expression for general weights (appendix A.4)
### It is a series respresentation that is steered by the *percentage* parameter, which decides when to stop (approximates eq. 35)

### Compared to the other formulas above, it is a pure multi-bin expression, i.e. 
### k is a list, all_weights is a list, and weight_index_list is a list of lists - one list for each bin containing the weight_indices for that which is jsed for *all_weights*
def poisson_general_weights_series(k, all_weights, weight_index_list, percentage=0.99, prior_factor=0.0,verbose=False):

    log_sum=0.0

    for ind, cur_windex_max in enumerate(weight_index_list):
        cur_weights=all_weights[cur_windex_max]
        cur_weights_sorted=numpy.sort(cur_weights)

        delta_vec_log, sum_log_C=get_delta_factor_n_logC_percentage(percentage, cur_weights_sorted, prior_factor=prior_factor,verbose=verbose)
        

        ## plus 1 because alphavec contains only ones and last entry 2, -1 because we need to calculate k_mc
        alpha_plus_s_equiv_to_k=numpy.arange(len(delta_vec_log))+len(cur_weights_sorted) 
        alpha_plus_s_equiv_to_k=numpy.array(alpha_plus_s_equiv_to_k, dtype=float) 

        alpha_plus_s_equiv_to_k+=float(prior_factor)/float(len(cur_weights_sorted))

        inner_logsum=poisson_llh_exp_value_gamma_equal_singleweight_n_delta(k[ind], min(cur_weights_sorted), alpha_plus_s_equiv_to_k, delta_vec_log)
        
        log_sum+=sum_log_C
     
        log_sum+=scipy.misc.logsumexp(inner_logsum)

        if(verbose):
            if(ind % len(weight_index_list) == 0 ):
                print 100.0*float(ind)/float(len(weight_index_list)), " percent complete..."
    
    return log_sum


############################## HELPER Functions for series representation of the Poisson expression for general weights
#######################################################################################################################

def get_gamma_factor_log(k, alpha_vec, sum_vector_for_gamma_log):

    return scipy.misc.logsumexp( numpy.log(alpha_vec) + k*(sum_vector_for_gamma_log)) - numpy.log(k)

def get_delta_factor_n_logC_percentage(percentage, sorted_weights, prior_factor=0.0,verbose=False):

    w_0=1.0

    sum_vector_for_gamma=1.0-(sorted_weights.min()/sorted_weights)

    alpha_vec=numpy.ones(len(sum_vector_for_gamma), dtype=float)

    alpha_vec+=float(prior_factor)/float(len(alpha_vec))

    sum_vector_for_gamma_log=scipy.misc.logsumexp( [numpy.zeros(len(sorted_weights)),( numpy.log(sorted_weights.min())-numpy.log(sorted_weights))], b=numpy.array([[1.0],[-1.0]]), axis=0) 
    
    delta_vector=[1.0]
    delta_vector_log=[0.0]

    C=numpy.prod((sorted_weights.min()/sorted_weights)**alpha_vec)
    sum_log_C=((numpy.log(sorted_weights.min()) - numpy.log(sorted_weights) )*alpha_vec).sum()

   
    percentage_reached=numpy.sum(delta_vector)*C
  
    percentage_reached_log=numpy.exp(scipy.misc.logsumexp(delta_vector_log)+sum_log_C)

    

    while(percentage_reached_log<percentage):

        i_vector=numpy.arange(len(delta_vector_log))+1
        
      
        gammalist=[]
        for i in i_vector:
            gammalist.append(get_gamma_factor_log(i, alpha_vec, sum_vector_for_gamma_log))
        full_gammas_up_till_now_log=numpy.array(gammalist)
        #full_gammas_up_till_now_log=numpy.array([get_gamma_factor_log(i, alpha_vec, sum_vector_for_gamma_log) for i in i_vector])
        new_entry_log=scipy.misc.logsumexp(numpy.array(delta_vector_log)[::-1]+full_gammas_up_till_now_log+numpy.log(i_vector)) - numpy.log(float(len(delta_vector_log)))
        delta_vector_log.append(new_entry_log)
        
        percentage_reached_log=numpy.exp(scipy.misc.logsumexp(delta_vector_log)+sum_log_C)


    if(verbose):
        if(len(delta_vector_log)>10):
            #print sum(delta_vector_log)
            print "PERCENTAGE REACHED...", percentage_reached_log, percentage_reached, " with weights ", sorted_weights, " in %d steps " % (len(delta_vector_log))
            
    return numpy.array(delta_vector_log), sum_log_C


def poisson_llh_exp_value_gamma_equal_singleweight_n_delta(k,single_weight, alpha_equiv, delta_factors_log):
    return scipy.special.gammaln(k+alpha_equiv) -scipy.special.gammaln(k+1)-scipy.special.gammaln(alpha_equiv) + (alpha_equiv)* numpy.log(1.0/single_weight) + delta_factors_log - (alpha_equiv+k)*numpy.log(1.0+1.0/single_weight)

##################################### end helper functions series representation
##########################################################

### Implementation of a frequentist alternative formula derived in https://arxiv.org/abs/1304.0735
### The results are practically equivalent to the general weights formulation above, but typically much faster
###########################################################################################

def poisson_general_weights_chirkin_13(data, all_weights, weight_indices):
    """
    Returns the positive log-likelihood value between data and simulation
    taking into account the finite statistics.
    """
    def func(x, w, d):
        """
        Reweighting function: 1/sum(w/(1+xw))
        The function should be equal to (1-x)/N_exp
        for reweighting variable x. Note, that w is an array
        since it is the (i,j) entry of w_hist.
        """
        return 1./numpy.sum(w/(1. + x*w)) - (1. - x)/d


    if data.ndim == 1:
        # array of reweighting factors
        lagrange = numpy.array([(scipy.optimize.brentq(func, -0.999999/max(all_weights[w]), 1., args=(all_weights[w],d), full_output=False)\
                    if d else 1.) if (len(w)>0) else 0. for (d, w) in zip(data, weight_indices)])
        # llh with new weights
        llh = numpy.array([numpy.sum(numpy.log(1. + lagrange[i]*all_weights[w])) if(len(w)>0) else 0 for (i,w) in enumerate(weight_indices)])\
              + data * numpy.log(1.-(lagrange-(lagrange == 1.)))
    else:
        raise NotImplementedError("`data` has more than 1 dimensions.")


    return -llh.sum()

