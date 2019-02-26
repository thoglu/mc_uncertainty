from autograd import numpy as numpy
import scipy
import lauricella_fd
import llh_fast
import poisson_gamma_mixtures
import copy


########################################################################################
####### Relevant Poisson generalizations from the paper https://arxiv.org/abs/1712.01293
####### and the newer one https://arxiv.org/abs/1902.08831
####### All formulas return the log-likelihood or log-probability
####### Formulas are not optimized for speed, but for clarity (except c implementations to some extent).
####### They can definately be sped up by smart indexing etc., and everyone has to adjust them to their use case anyway.
####### Formulas are not necessarily vetted, please try them out yourself first.
####### Any questions: thorsten.gluesenkamp@fau.de
########################################################################################

numpy.seterr(divide="warn")

######################################
#### standard Poisson likelihood
######################################
def poisson(k, lambd):
 
    return (-lambd+k*numpy.log(lambd)-scipy.special.gammaln(k+1)).sum()

################################################################
### Simple Poisson-Gamma mixture with equal weights (eq. 21 - https://arxiv.org/abs/1712.01293)
### multi bin expression, one k, one k_mc and one avg_weight item for each bin, all given by an array
def pg_equal_weights(k,k_mc,avgweights,prior_factor=0.0):
    
    return (scipy.special.gammaln((k+k_mc+prior_factor)) -scipy.special.gammaln(k+1.0)-scipy.special.gammaln(k_mc+prior_factor) + (k_mc+prior_factor)* numpy.log(1.0/avgweights) - (k_mc+k+prior_factor)*numpy.log(1.0+1.0/avgweights)).sum()


#################################################################################
## Standard poisson-gamma mixture, https://arxiv.org/abs/1712.01293 eq.  ("L_G")
################################################################################# 
def pg_log_python(k, weights, alpha_individual=0.0, extra_prior_counter=0.0, extra_prior_weight=1.0):
    
    log_deltas=[0.0]
    log_inner_factors=[]
    log_weight_prefactors=0.0

    log_weight_prefactors+=-((1.0+alpha_individual)*numpy.log(1.0+weights)).sum()

    first_fac=1.0+alpha_individual
    
    log_first_fac= numpy.log(first_fac)
    log_first_var=-numpy.log(1.0+1.0/weights)

    running_factor_vec=0.0

    if(k>0):
        
        for i in (numpy.arange(k)+1):
            
            running_factor_vec+=log_first_var
            
            # summing assuming all summands positive .. which is the only well-defined region here
            
            res=scipy.misc.logsumexp(log_first_fac+running_factor_vec)
            
            log_inner_factors.append(res)
            new_delta=scipy.misc.logsumexp( numpy.array(log_inner_factors[::-1])+numpy.array(log_deltas))-numpy.log(i)
            log_deltas.append(new_delta)
 
            #log_inner_factors.append((first_fac*(first_var**i)-second_fac*(second_var**i)).sum())
            #log_deltas.append( (numpy.array(inner_factors)*numpy.array(deltas[::-1])).sum()/float(i))
            
    return log_deltas[-1]+log_weight_prefactors

## fast methods that employ c implementations and only fall back to log-python from above when certain accuracy is required 
def fast_pg_single_bin(k, weights, mean_adjustment_per_weight):
    
    # alphas array corresponds to alpha/N in paper
    ## gamma poisson mixture based on gamma-poisson priors - general


    betas=1.0/weights
    alphas=numpy.ones(len(weights), dtype=float)+mean_adjustment_per_weight
    ret=poisson_gamma_mixtures.c_generalized_pg_mixture(k, alphas, betas)

    if(ret>1e-300 and len(weights)>0):
        return numpy.log(ret)
    else:
        
        return pg_log_python(k,weights, alpha_individual=mean_adjustment_per_weight)

################## end standard Poisson mixture ####################
####################################################################

########################################
### generalization (1) https://arxiv.org/abs/1902.08831, eq. 35 / eq. 97
########################################

def pgpg_log_python(k, weights, mean_adjustment):
    
    ## fix extra adjustment in standard PG to 0
    alpha_individual=0.0
    
    log_weight_prefactors=-(alpha_individual*numpy.log(1.0+weights)).sum()
   
    Cs=1.0/(2+2*weights)

    log_weight_prefactors+=-((1.0+mean_adjustment)*numpy.log(2.0-2*Cs)).sum()
    
    log_E_s=-numpy.log(1.+1./weights)
    #one_minus_c=(1+2*weights)/(2+2*weights)
    
    first_fac=1.0+mean_adjustment
    signs_first=numpy.where(first_fac>0, 1.0, -1.0)
    log_first_fac=numpy.where(signs_first>0, numpy.log(first_fac), numpy.log(-first_fac))
    
    second_fac=-(1.0+mean_adjustment-alpha_individual)
    signs_second=numpy.where(second_fac>0, 1.0, -1.0)
    log_second_fac=numpy.where(signs_second>0, numpy.log(second_fac), numpy.log(-second_fac))

    log_first_var=log_E_s-numpy.log(1.0-Cs)
    log_second_var=log_E_s

    log_deltas=[0.0]
    log_inner_factors=[]
    
    running_factor_vec_first=0.0
    running_factor_vec_second=0.0
    
    if(k>0):
        
        for i in (numpy.arange(k)+1):
            
            running_factor_vec_first+=log_first_var
            running_factor_vec_second+=log_second_var
            
            sum1,sign1=scipy.misc.logsumexp(log_first_fac+running_factor_vec_first, b=signs_first,return_sign=True)
            sum2,sign2=scipy.misc.logsumexp(log_second_fac+running_factor_vec_second, b=signs_second,return_sign=True)
            
            res=scipy.misc.logsumexp([sum1, sum2], b=[sign1,sign2, 1.0])
            
            log_inner_factors.append(res)
            new_delta=scipy.misc.logsumexp( numpy.array(log_inner_factors[::-1])+numpy.array(log_deltas))-numpy.log(i)
            log_deltas.append(new_delta)
 
            #log_inner_factors.append((first_fac*(first_var**i)-second_fac*(second_var**i)).sum())
            #log_deltas.append( (numpy.array(inner_factors)*numpy.array(deltas[::-1])).sum()/float(i))
            
    
    return log_deltas[-1]+log_weight_prefactors

def fast_pgpg_single_bin(k, weights, mean_adjustment):
    
    ## fast calculation in c without logarithm .. if return value is too small, go to 
    ## more time consuming calculation in log space in python
    ## gamma poisson mixture based on gamma-poisson priors - general

    gammas=1.0/weights
    deltas=numpy.ones(len(weights), dtype=float)+mean_adjustment
    epsilons=numpy.ones(len(deltas), dtype=float)
    
    ret=poisson_gamma_mixtures.c_generalized_pg_mixture_marginalized(k, gammas, deltas, epsilons)

    if(ret>1e-300 and len(weights)>0):
        return numpy.log(ret)
    else:
        return pgpg_log_python(k,weights, mean_adjustment=mean_adjustment)


####################
### end generalization (1)
####################


##############################################################
### generalization (2) - allbin expression, https://arxiv.org/abs/1902.08831 eq. 47
##############################################################

## the next 3 functions are used to calculate the convolution of N poisson-gamma mixtures in a safe way
### PG conv PG conv PG ... etc
import itertools
def bars_and_stars_iterator(tot_k, num_bins):

    for c in itertools.combinations(range(tot_k+num_bins-1), num_bins-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(tot_k+num_bins-1,))]


## second way to calculate generalized pg mixture, based on iterative sum
def generalized_pg_mixture_2nd(k, alphas, betas):
    
    iters=[numpy.array(i) for i in bars_and_stars_iterator(int(k), len(betas))]
    
    log_res=[]
    for it in iters:
        
        log_res.append(calc_pg(it, alphas, betas).sum())
    
    return scipy.misc.logsumexp(log_res)


## calculate c-based version .. if it doesnt suffice in precision, go to direct convolution
def fast_pgmix(k, alphas, betas):
    ret=poisson_gamma_mixtures.c_generalized_pg_mixture(k, alphas, betas)
    
    if(ret>1e-300):
        return numpy.log(ret)
    else:
        return generalized_pg_mixture_2nd(k, alphas, betas)

def poisson_gen2(data, individual_weights_dict, mean_adjustments, larger_weight_variance=False):

    tot_llh=0.0

    for cur_bin_index, _ in enumerate(individual_weights_dict.values()[0]):

        alphas=[]
        betas=[]

        for src in individual_weights_dict.keys():

            this_weights=individual_weights_dict[src][cur_bin_index]

            if(len(this_weights)>0):
                kmc=float(len(this_weights))
                mu=float(len(this_weights))
                
                exp_w=0.0
                #print "ind .. ", cur_bin_index, " ", src
                
                exp_w=numpy.mean(this_weights)
                var_w=0.0

                if(larger_weight_variance):
                    var_w=(this_weights**2).sum()/float(len(this_weights))
                else:
                    var_w=((this_weights-exp_w)**2).sum()/(float(len(this_weights)))

                var_z=(var_w+exp_w**2)

                beta=exp_w/var_z
                trad_alpha=(exp_w**2)/var_z

                #sumw=this_weights.sum()
                #sqrw=(this_weights**2).sum()
                

                extra_fac=mean_adjustments[src]

                alphas.append( (mu+extra_fac)*trad_alpha)
                betas.append(beta)

               

        if(len(alphas)>0):

            tot_llh+=fast_pgmix(data[cur_bin_index], numpy.array(alphas), numpy.array(betas))

   
    return tot_llh

########################################
# end generalization (2)
########################################


####################################################
### effective generalization (2) - allbin expression, https://arxiv.org/abs/1902.08831 , eq 48
####################################################

def poisson_gen2_effective(data, individual_weights_dict, mean_adjustments):

    tot_llh=0.0

    for cur_bin_index, _ in enumerate(individual_weights_dict.values()[0]):

        mus=[]
        all_weights=[]

        for src in individual_weights_dict.keys():

            this_weights=individual_weights_dict[src][cur_bin_index]

            if(len(this_weights)>0):
                
                all_weights.extend(this_weights.tolist())
                
                mus.append(float(len(this_weights))+mean_adjustments[src])


        all_weights=numpy.array(all_weights)

        if(len(all_weights)>0):
            #print all_weights
            kmc=float(len(all_weights))
           

            exp_w=numpy.mean(all_weights)
            var_w=((exp_w-all_weights)**2).sum()/kmc

            var_z=(var_w+exp_w**2)

            #print "expw,varw,varz", exp_w,var_w,var_z
            beta=exp_w/var_z
            trad_alpha=(exp_w**2)/var_z
            
            alpha=sum(mus)*trad_alpha
            k=data[cur_bin_index]

            #print "alpha,beta,k", alpha,beta,k

            this_llh= (scipy.special.gammaln((k+alpha)) -scipy.special.gammaln(k+1.0)-scipy.special.gammaln(alpha) + (alpha)* numpy.log(beta) - (alpha+k)*numpy.log(1.0+beta)).sum()

            tot_llh+=this_llh

    return tot_llh

##################################
### generalization (3) functions, https://arxiv.org/abs/1902.08831, eq. 51
##################################

### Genertes stirling numbers (logarithmic) up to maximum no of max_val
### returns a lower-triangular matrix
def generate_log_stirling(max_val=1000):

    arr=numpy.zeros(shape=(max_val+1, max_val+1))*(-numpy.inf)
    arr[0][0]=0.0
    for i in range(max_val+1):
        for j in range(i+1):
            if(i==0 and j==0):
                arr[i][j]=0.0
                continue
            
            if(j==0):
                arr[i][j]=-numpy.inf
                continue
            if(i==j):
                arr[i][j]=0.0
                continue
          
            arr[i][j]=scipy.misc.logsumexp([numpy.log(i-1.0)+arr[i-1][j],arr[i-1][j-1]])
    
    return arr

def poisson_gen3(k, individual_weights_dict, mean_adjustments, log_stirlings,s_factor=1.0, larger_weight_variance=False):

    tot_llh=0.0

    num_sources=len(individual_weights_dict.keys())

    for cur_bin_index, _ in enumerate(individual_weights_dict.values()[0]):

        As=[]
        Bs=[]
        Qs=[]
        kmcs=[]
        gammas=[]

        for src in individual_weights_dict.keys():

            this_weights=individual_weights_dict[src][cur_bin_index]

            if(len(this_weights)>0):
                kmc=float(len(this_weights))
                mu=float(len(this_weights))

                Q=0.0
                if(larger_weight_variance>0):
                    exp_w=numpy.mean(this_weights)
                    var_w=0.0

                    ## pdf is a mixture of gammas
                    var_w=(this_weights**2).sum()/float(len(this_weights))
                    
                    var_z=(var_w+exp_w**2)
                    beta=exp_w/var_z
                    trad_alpha=(exp_w**2)/var_z
                    Q=trad_alpha
                else:

                    sumw=this_weights.sum()
                    sumw2=(this_weights**2).sum()
                    
                    beta=sumw/sumw2

                    trad_alpha=sumw**2/sumw2
                    Q=(1.0/kmc)*(trad_alpha)
                
                A=beta/(1.0+beta)
                B=1.0/(1.0+beta)

                extra_fac=mean_adjustments[src]


                As.append(A)
                Bs.append(B)
                Qs.append(Q)

                kmcs.append( (mu+extra_fac)/s_factor)
                gammas.append(1.0/s_factor)

        As=numpy.array(As)
        Bs=numpy.array(Bs)
        Qs=numpy.array(Qs)
        kmcs=numpy.array(kmcs)
        gammas=numpy.array(gammas)

        tot_llh+=poisson_gamma_mixtures.c_multi_pgg(int(k[cur_bin_index]), As,Bs,Qs,kmcs,gammas, log_stirlings)

    return tot_llh          

######### ##################
#### end generalization (3)
############################

############################################
### generic preprocessing to include prior information and empty bins as described in the paper
############################################

def generic_pdf(k_list, dataset_weights, type="basic_pg", empty_bin_strategy=0, empty_bin_weight="max", mean_adjustment=False, s_factor=1.0, larger_weight_variance=False, log_stirling=None):
    """
    k_list - a numpy array of counts for each bin
    dataset_weights_list - a dictionary of lists of numpy arrays. Each list corresponds to a dataset and contains numpy arrays with weights for a given bin. empty bins here mean an empty array
    type - old/gen1/gen2/gen2_effective/gen3 - handles the various formulas from the two papers
    empty_bin_strategy - 0 (no filling), 1 (fill up bins which have at least one event), 2 (fill up all bins)
    empty_bin_weight - what weight to use for pseudo counts in empty  bins? "max" , maximum of all weights of dataset (used in paper) .. could be mean etc
    mead_adjustment - apply mean adjustment as implemented in the paper? yes/no
    weight_moments - change to more "unbiased" way of determining weight distribution moments as implemented in the paper
    """

    ## calculate number of mc events / bin / dataset
    kmc_dict=dict()
    max_weights=dict()
    for dsname in dataset_weights.keys():
        mw=max([max(w) if len(w)>0 else 0 for w in dataset_weights[dsname]])
        kmc_dict[dsname]=numpy.array([len(w) for w in dataset_weights[dsname]])
        max_weights[dsname]=mw

    ## calculate mean adjustment per dataset
    mean_adjustments=dict()
    for dsname in kmc_dict.keys():

        avg_kmc=numpy.mean(kmc_dict[dsname])
                            
        delta_alpha=0.0
        if(avg_kmc<1.0):
            delta_alpha=-(1.0-avg_kmc)+1e-3
        mean_adjustments[dsname]=delta_alpha


    ## fill in empty bins - update the weights
    new_weights=copy.deepcopy(dataset_weights)

    ## strategy 1 - fill up only bins that have at least 1 mc event from any dataset
    if(empty_bin_strategy==1):

        for bin_index in range(len(k_list)):
            
            weight_found=False
            for dsname in kmc_dict.keys():
                if(kmc_dict[dsname][bin_index] > 0):
                    weight_found=True

            if(weight_found):

                for dsname in kmc_dict.keys():
                    if(kmc_dict[dsname][bin_index]==0):
                        new_weights[dsname][bin_index]=numpy.array([max_weights[dsname]])

    # strategy 2 - fill up all bins
    elif(empty_bin_strategy==2):
        for bin_index in range(len(k_list)):
            
            for dsname in kmc_dict.keys():
                if(kmc_dict[dsname][bin_index]==0):
                    new_weights[dsname][bin_index]=numpy.array([max_weights[dsname]])


    ## now loop through all bins and call respective likelihood
    ## manifest mean adjustment possible only in gen2 and gen3 (see table in paper)

    llh_res=0.0

    if(type=="gen3"):
        if(log_stirling is None):
            log_stirling=generate_log_stirling(max_val=max([max(k_list),1]))

        llh_res=poisson_gen3(k_list, new_weights, mean_adjustments, log_stirling,s_factor=s_factor, larger_weight_variance=larger_weight_variance)
        
    elif(type=="gen2"):
        llh_res=poisson_gen2(k_list, new_weights, mean_adjustments, larger_weight_variance=larger_weight_variance)
    elif(type=="gen2_effective"):
        llh_res=poisson_gen2_effective(k_list, new_weights, mean_adjustments)
    else:

        ## calculate an effective mean adjustment per weight (no manifest mean adjustment possible)

        for bin_index in range(len(k_list)):
            total_weights=[]
            individual_mean_adjustments=[]

            for dsname in new_weights.keys():
                this_weights=new_weights[dsname][bin_index].tolist()
                this_len=len(this_weights)
                total_weights.extend(this_weights)

                ## alpha* = alpha/N
                if(this_len>0):
                    individual_mean_adjustments.extend(this_len*[mean_adjustments[dsname]/float(this_len)])

            if(len(total_weights)>0):

                total_weights=numpy.array(total_weights)
                individual_mean_adjustments=numpy.array(individual_mean_adjustments)

                if(type=="basic_pg"):
                    llh_res+=fast_pg_single_bin(k_list[bin_index], total_weights, individual_mean_adjustments)
                elif(type=="gen1"):
                    llh_res+=fast_pgpg_single_bin(k_list[bin_index], total_weights, individual_mean_adjustments)
                


    return llh_res

#################################
### arXiv:1901.04645, Arg+elles et al.
##################################
## a=1 is effective version, recommended by the authors
##########################################
def asy_llh(data,all_weights,weight_indices, a_prior=1):
    """
    data - array of data counts of size J
    all_weights - array of all weights of size N
    weight_indices - list of numpy arrays of size J (one index array per bin, picks out the wieghts from all_weights)
    """
    tot_llh=0.0
    for ind, cur_weight_mask in enumerate(weight_indices):

        cur_weights=all_weights[cur_weight_mask]
        weight_sum=cur_weights.sum()
        sqr_weight_sum=(cur_weights**2).sum()

        alpha=(weight_sum**2)/sqr_weight_sum+a_prior
        beta=weight_sum/sqr_weight_sum
        tot_llh+= (scipy.special.gammaln((data[ind]+alpha)) -scipy.special.gammaln(data[ind]+1.0)-scipy.special.gammaln(alpha) + (alpha)* numpy.log(beta) - (alpha+data[ind])*numpy.log(1.0+beta)).sum()

    return tot_llh    


##########################
###### Barlow/Beeston (https://www.sciencedirect.com/science/article/pii/001046559390005W) (without const terms)
#########################

def barlow_beeston_llh(data, weights_dict, indices_dict):
    """
    data - array of data counts of size J
    all_weights - array of all weights of size N
    weight_indices - list of numpy arrays of size J (one index array per bin, picks out the wieghts from all_weights)
    """

    def func(x, w, d):
        #print "x ", x
        #print "w ", w
        #print "d ", d
        """
        Reweighting function: 1/sum(w/(1+xw))
        The function should be equal to (1-x)/N_exp
        for reweighting variable x. Note, that w is an array
        since it is the (i,j) entry of w_hist.
        """
        return 1./numpy.sum(w/(1. + x*w)) - (1. - x)/d


    if data.ndim == 1:
        # array of reweighting factors

        new_avg_weights=[]
        new_kmcs=[]

        ## ind goes over all bins
        
        for ind in numpy.arange(len(indices_dict.values()[0])):
            these_weights=[]

            for src_key in indices_dict.keys():
                #print all_weights_src_list[ind_src][weight_indices_src_list[ind_src][ind]]
                
                if(len(indices_dict[src_key][ind])>0):
                    these_src_weights=weights_dict[src_key][indices_dict[src_key][ind]]

                    these_weights.extend([numpy.mean(these_src_weights)]*len(these_src_weights))
            
            new_avg_weights.append(numpy.array(these_weights))
            

        lagrange = numpy.array([(scipy.optimize.brentq(func, -0.999999/max(w), 1., args=(w,d), full_output=False)\
                    if d else 1.) if (len(w)>0) else 0. for (d, w) in zip(data, new_avg_weights)])


        # llh with new weights
        llh = numpy.array([numpy.sum((numpy.log(1. + lagrange[i]*w))) if(len(w)>0) else 0 for (i,w) in enumerate(new_avg_weights)])\
              + data * numpy.log(1.-(lagrange-(lagrange == 1.)))


    else:
        raise NotImplementedError("`data` has more than 1 dimensions.")


    return -llh.sum()
        


##############################################################################
### Chirkin (https://arxiv.org/abs/1304.0735)
###########################################################################################


def chirkin_llh(data, all_weights, weight_indices):
    """
    data - array of data counts of size J
    all_weights - array of all weights of size N
    weight_indices - list of numpy arrays of size J (one index array per bin, picks out the wieghts from all_weights)
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
        print "dima lagrange ", lagrange
        llh = numpy.array([numpy.sum(numpy.log(1. + lagrange[i]*all_weights[w])) if(len(w)>0) else 0 for (i,w) in enumerate(weight_indices)])\
              + data * numpy.log(1.-(lagrange-(lagrange == 1.)))
        raise NotImplementedError("`data` has more than 1 dimensions.")


    return -llh.sum()



