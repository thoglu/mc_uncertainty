import numpy
import scipy.misc
import scipy
import itertools
import carlson_r
import cdecimal
cdecimal.getcontext().prec=100
import mpmath

def custom_logsumexp_mpmath(logs, signs):

    positive_mask=signs>0

    positive_logs=numpy.array(logs, dtype=object)[positive_mask]
    negative_logs=numpy.array(logs, dtype=object)[positive_mask==False]

  
    res_pos=None
    res_neg=None

    if(len(positive_logs)>0):
        res_pos=max(positive_logs)+mpmath.log(sum([ mpmath.exp(i-max(positive_logs)) for i in positive_logs]))
        
    if(len(negative_logs)>0):
        res_neg=max(negative_logs)+mpmath.log(sum([ mpmath.exp(i-max(negative_logs)) for i in negative_logs]))
    
    #print "repos/resneg", res_pos, res_neg
    
    if(res_neg is None):
        return res_pos, 1.0
    if(res_pos is None):
        return res_neg, -1.0
   
    if(res_pos==res_neg):
        print "not enough precision!!!..."
        exit(-1)
        return None, None
    elif(res_pos==res_neg and res_pos==0):
        print "0?!"
        print logs
        exit(-1)
    if(res_neg<res_pos):
        ## easy case .. subtracted number is smaller 
        return res_neg + mpmath.log(mpmath.exp(res_pos-res_neg) - 1), 1.0
    else:
        ## A-B < 0 -> A-B = - (B-A)
        #print "the usual case... respos/resneg ", res_pos, res_neg 
        return res_pos + mpmath.log(mpmath.exp(res_neg-res_pos) - 1), -1.0



# Numerical evaluation of integral representation 1, multi-dimensional integral over simplex
def log_lauricella_dumb_integral_multi(a, bs, c, zs, nthrows=1000):
    """
    First known Lauricella representation as an integral over the k-1 simplex. Only valid if c>sum(b_i).
    Everything done logarithmically to avoid numerical problems.
    """
    
    ### sampling from a simplex in n+1 dimensional space ... exponential random variables, normalized.... 
    ### this gives a uniform sample on the n simplex in n+1 dimension space (sum x=1).. but we want open (sum x <= 1) on n simplex in n dimensiona space ... easy, just drop last coordinate
    if(type(bs)==list):
        bs=numpy.array(bs)
    if(type(zs)==list):
        zs=numpy.array(zs)
    
    ## normalizing factor of gamma functions
    log_normalizing_factor=scipy.special.gammaln(numpy.array([c]))-sum(scipy.special.gammaln(bs))-scipy.special.gammaln(numpy.array([c])-sum(bs))-scipy.special.gammaln(len(zs)+1)

    ## random numbers for numerical integration
    random_nos=-numpy.log(numpy.random.uniform(size=(nthrows,len(zs)+1)))
    rsum=numpy.reshape(random_nos.sum(axis=1), (nthrows,1 ))
    random_nos=random_nos/rsum
      
    ## use average integration volume per throw
    log_integration_volume_per_throw=-numpy.log(float(nthrows))
    
    ## subtract last random no to land in k-1 simplex
    final_randoms=random_nos[:,:-1]
    
    
    ## the dirichlet part
    basic_dirichlet_logintegrand=( numpy.resize((bs-1), final_randoms.shape)*numpy.log(final_randoms) ).sum(axis=1) +(c-numpy.sum(bs)-1)*numpy.log(1.0-numpy.sum(final_randoms, axis=1))
    
   
    ## the extra integrand part
    log_integrand_vals2=-a*numpy.log((1.0-numpy.exp( scipy.misc.logsumexp(numpy.log(final_randoms)+numpy.resize(numpy.log(zs), final_randoms.shape), axis=1))) ) 
    
    ## the log of the total integrated integrand
    log_result= scipy.misc.logsumexp(log_integration_volume_per_throw+basic_dirichlet_logintegrand+log_integrand_vals2)
    
    return log_result+log_normalizing_factor
    
# Numerical evaluation of Integral representation 2 - 1-d integral over the line
def log_lauricella_dumb_integral_single(a,bs, c, zs, nthrows=1000):
    """
    Second known integral lauricella representation in form of a single integral over [0,1]. Only valid if c>a.
    """
    if(a==c):
        ## get out since this is a limiting case and not defined in the integral rep
        return (-bs*numpy.log(1.0-zs)).sum()

    random_vals=numpy.random.uniform(size=nthrows)
    random_vals2=numpy.resize(random_vals, (len(bs),nthrows)).T
    
    log_res=(a-1.0)*numpy.log(random_vals) + (c-a-1.0)* numpy.log(1.0-random_vals) + (  -bs*numpy.log(1.0-random_vals2*zs)).sum( axis=1)- numpy.log(float(nthrows))

    log_normfactor=scipy.special.gammaln(c)-scipy.special.gammaln(a)-scipy.special.gammaln(c-a)

    return scipy.misc.logsumexp(log_normfactor+log_res)


## Calculate all combinations of tot_k distributed over num_bins, such that sum k_i = tot_k
def bars_and_stars_iterator(tot_k, num_bins):

    for c in itertools.combinations(range(tot_k+num_bins-1), num_bins-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(tot_k+num_bins-1,))]


# Exact solution of F_D for a > c, both integer (calculation of Contour integral to solve combinatorial representation of Carlson_R implicitly) - see eq. 23
# Contour integral solved via 
# Ma et. al 2014, "Efficient Recursive Methods for Partial Fraction Expansion of General Rational Functions",
# http://dx.doi.org/10.1155/2014/895036.
########################################################################################################
# TODO: Currently Section 2 for factorized functions and higher-order Poles. Should check out Section 3, the two other algorithms!
## Simple poles, which the majority of poles is, should work fine with simple increased precision.
########################################################################################################################
def log_lauricella_exact_cauchy(a,bs,c,zs):

    def new_z_minus_z(new_z, z_vec, b_vec):
        #signs=numpy.ones(len(b_vec))
        signs=numpy.where(new_z<z_vec, -1.0, 1.0)
        sum_signs=numpy.where( (signs < 0) & (b_vec%2==1)  , -1.0, 1.0)
        neg=(sum_signs==-1).sum()%2!=0

        sign=1.0
        if(neg):
            sign=-1.0 
        decimals= numpy.array([ mpmath.mpf(new_z)-mpmath.mpf(z_vec[ind]) for ind in range(len(z_vec))])#[ ( cdecimal.Decimal(int(b_vec[ind]))*( (cdecimal.Decimal(new_z)-cdecimal.Decimal(z_vec[ind]))*cdecimal.Decimal(int(signs[ind]))).ln() ) for ind in range(len(z_vec)) ]

        return numpy.prod(decimals**b_vec), sign
   
    def get_high_order_residue(new_z, z_vec, other_b_vec, order, base_residue, base_residue_sign, upper_degree):

        signs=numpy.where(z_vec<new_z, -1.0, 1.0)
        if(new_z<0):
            signs=numpy.append(signs, -1.0)
        else:
            signs=numpy.append(signs, 1.0)


     

        #basic_one_over_log=numpy.append([ -mpmath.log((mpmath.mpf(i)-mpmath.mpf(new_z))*signs[ind]) for ind, i in enumerate(z_vec)], [-mpmath.log(new_z*signs[-1])])
        #after_exp_correction=numpy.zeros(len(basic_one_over_log),dtype=object)
        #after_exp_correction[-1]=mpmath.log(upper_degree)
        #after_exp_correction[:-1]=numpy.array([mpmath.log(i) for i in other_b_vec])#numpy.log(other_b_vec)

        basic_one_over_log=numpy.append(-numpy.log((z_vec-new_z)*signs[:-1]), -numpy.log(new_z*signs[-1]))
        after_exp_correction=numpy.zeros(len(basic_one_over_log),dtype=float)
        after_exp_correction[-1]=numpy.log(upper_degree)
        after_exp_correction[:-1]=numpy.array([numpy.log(i) for i in other_b_vec])#numpy.log(other_b_vec)

        lambdas=[]

        cs=[]
        cs.append( (base_residue,base_residue_sign) )

        ## extra orders
        for o in numpy.arange(int(order)-1)+1:
            # o is only factorial, thats why +1

            ## generate the lambdas .. 
            if(o%2==0):
                lambdas.append(scipy.misc.logsumexp(o*basic_one_over_log+after_exp_correction, b=numpy.ones(len(signs)-1).tolist() + [-1.0], return_sign=True))
                #print "oooo"
                #print "even: "
                #print "input ", o*basic_one_over_log+after_exp_correction, " signs.. ", numpy.ones(len(signs)-1).tolist() + [-1.0]
                #print scipy.misc.logsumexp(o*basic_one_over_log+after_exp_correction, b=numpy.ones(len(signs)-1).tolist() + [-1.0], return_sign=True)
                #print custom_logsumexp_mpmath(o*basic_one_over_log+after_exp_correction, numpy.array(numpy.ones(len(signs)-1).tolist() + [-1.0]) )
                #print  "ooooo"
                #lambdas.append(custom_logsumexp_mpmath(o*basic_one_over_log+after_exp_correction, numpy.array(numpy.ones(len(signs)-1).tolist() + [-1.0]) ))
            else:
                lambdas.append(scipy.misc.logsumexp(o*basic_one_over_log+after_exp_correction, b=signs, return_sign=True))
                #print "----"
                #print "odd:"
                #print "input ", o*basic_one_over_log+after_exp_correction, " signs.. ", numpy.ones(len(signs)-1).tolist() + [-1.0]
                #print scipy.misc.logsumexp(o*basic_one_over_log+after_exp_correction, b=signs, return_sign=True)
                #print custom_logsumexp_mpmath(o*basic_one_over_log+after_exp_correction, signs)
                #print "----"
                #lambdas.append(custom_logsumexp_mpmath(o*basic_one_over_log+after_exp_correction, signs))
            new_cs_list=[]
            new_cs_sign=[]

 
            for sumlen in range(o):
                new_cs_list.append( lambdas[sumlen][0]+cs[o-1-sumlen][0]    )
                new_cs_sign.append(1.0) if (lambdas[sumlen][1] == cs[o-1-sumlen][1] ) else  new_cs_sign.append(-1.0)

            
            new_cs_sign=numpy.array(new_cs_sign)
            new_cs_list=numpy.array([float(c) for c in new_cs_list])

            cur_logsum, cur_sign=scipy.misc.logsumexp(new_cs_list, b=new_cs_sign, return_sign=True)
            #cur_logsum, cur_sign=custom_logsumexp_mpmath(new_cs_list, new_cs_sign)
            cur_logsum-=numpy.log(o)

            cs.append( (cur_logsum,cur_sign))
 
        return cs[-1]


    extended_zs=[]
    
    if(not 0.0 in zs):
        extended_zs=[1.0]
    extended_zs.extend((1.0-zs).tolist())
    extended_zs=numpy.array(extended_zs)**-1

    ## TODO - want to get rid of this step: Tuning precision is not optimal ....
    k_tot=a-c

    extended_bs=numpy.append([c-sum(bs)], bs)
    

    sorta=numpy.argsort(extended_zs)
    extended_zs=extended_zs[sorta]
    extended_bs=extended_bs[sorta]
    diffs=extended_zs[2:]-extended_zs[1:-1]

    log_normalizing_factor=scipy.special.gammaln(a-c+1)+scipy.special.gammaln(c)-scipy.special.gammaln(a)
    

    pol_deg=sum(extended_bs)+k_tot-1

    #### TUNE PRECISION PROPORTIONAL TO NUMERATOR DEGREE
    mpmath.mp.dps=15+200*pol_deg

    tot_sum_logs=[]
    tot_sum_logs_numpy=[]
    tot_signs=[]

    tot_res_sum=0.0

    if(len(extended_zs)==1 and extended_bs[0]==1):
        tot_sum_logs.append(mpmath.mpf(pol_deg*numpy.log(extended_zs[0])))
        tot_signs.append(1.0)
    else:
        for ind in range(len(extended_zs)):

            fac_nolog,sign=new_z_minus_z(extended_zs[ind], numpy.array(extended_zs[:ind].tolist()+extended_zs[ind+1:].tolist()), numpy.array(extended_bs[:ind].tolist()+extended_bs[ind+1:].tolist()))


            upper_sign=1.0
            if(extended_zs[ind]<0):
                upper_sign=-1.0
            

            basic_residue=pol_deg*mpmath.log(extended_zs[ind])-mpmath.log((fac_nolog*sign))
            
            sign=float(sign)*float(upper_sign)

            ## multiplicity of 1 ... only need basic residue
            if(extended_bs[ind]==1):
                
                tot_sum_logs.append(basic_residue)
                tot_signs.append(sign)
            else:
                ## multiplicity higher than 1 .. need higher order terms, following http://dx.doi.org/10.1155/2014/895036
                high_residue,high_sign=get_high_order_residue(extended_zs[ind], numpy.array(extended_zs[:ind].tolist()+extended_zs[ind+1:].tolist()),  numpy.array(extended_bs[:ind].tolist()+extended_bs[ind+1:].tolist())   ,   extended_bs[ind], float(basic_residue),sign , pol_deg )

                tot_sum_logs.append(mpmath.mpf(high_residue))
                tot_signs.append(high_sign)

    tot_signs=numpy.array(tot_signs)

    #print "final input: ", tot_sum_logs
    
    tot_sum_log, overall_sign=custom_logsumexp_mpmath(tot_sum_logs, tot_signs)

    k_mc_prefacs=(extended_bs*numpy.log(extended_zs)).sum()

    return float(k_mc_prefacs+log_normalizing_factor+tot_sum_log)


def log_lauricella_exact_via_explicit_contour_integral(a,bs,c,zs):

    
    extended_zs=[]
    
    if(not 0.0 in zs):
        extended_zs=[1.0]
    extended_zs.extend((1.0-zs).tolist())
    extended_zs=numpy.array(extended_zs)**-1

    ## TODO - want to get rid of this step: Tuning precision is not optimal ....
    k_tot=a-c

    extended_bs=numpy.append([c-sum(bs)], bs)
    

    sorta=numpy.argsort(extended_zs)
    extended_zs=extended_zs[sorta]
    extended_bs=extended_bs[sorta]
    diffs=extended_zs[2:]-extended_zs[1:-1]

    log_normalizing_factor=scipy.special.gammaln(a-c+1)+scipy.special.gammaln(c)-scipy.special.gammaln(a)
    

    log_contour_integral=contour_integral(extended_zs, extended_bs, k_tot)
    

    k_mc_prefacs=(extended_bs*numpy.log(extended_zs)).sum()

    return float(k_mc_prefacs+log_normalizing_factor+log_contour_integral)


def contour_integral(poles, multiplicities, k_tot):

    if(type(poles)==list):
        poles=numpy.array(poles)
    if(type(multiplicities)==list):
        multiplicities=numpy.array(multiplicities)
    def new_z_minus_z(new_z, z_vec, b_vec):
        #signs=numpy.ones(len(b_vec))
        signs=numpy.where(new_z<z_vec, -1.0, 1.0)
        sum_signs=numpy.where( (signs < 0) & (b_vec%2==1)  , -1.0, 1.0)
        neg=(sum_signs==-1).sum()%2!=0

        sign=1.0
        if(neg):
            sign=-1.0 
        decimals= numpy.array([ mpmath.mpf(new_z)-mpmath.mpf(z_vec[ind]) for ind in range(len(z_vec))])#[ ( cdecimal.Decimal(int(b_vec[ind]))*( (cdecimal.Decimal(new_z)-cdecimal.Decimal(z_vec[ind]))*cdecimal.Decimal(int(signs[ind]))).ln() ) for ind in range(len(z_vec)) ]

        return numpy.prod(decimals**b_vec), sign
   
    def get_high_order_residue(new_z, z_vec, other_b_vec, order, base_residue, base_residue_sign, upper_degree):

        signs=numpy.where(z_vec<new_z, -1.0, 1.0)
        if(new_z<0):
            signs=numpy.append(signs, -1.0)
        else:
            signs=numpy.append(signs, 1.0)


     

        #basic_one_over_log=numpy.append([ -mpmath.log((mpmath.mpf(i)-mpmath.mpf(new_z))*signs[ind]) for ind, i in enumerate(z_vec)], [-mpmath.log(new_z*signs[-1])])
        #after_exp_correction=numpy.zeros(len(basic_one_over_log),dtype=object)
        #after_exp_correction[-1]=mpmath.log(upper_degree)
        #after_exp_correction[:-1]=numpy.array([mpmath.log(i) for i in other_b_vec])#numpy.log(other_b_vec)

        basic_one_over_log=numpy.append(-numpy.log((z_vec-new_z)*signs[:-1]), -numpy.log(new_z*signs[-1]))
        after_exp_correction=numpy.zeros(len(basic_one_over_log),dtype=float)
        after_exp_correction[-1]=numpy.log(upper_degree)
        after_exp_correction[:-1]=numpy.array([numpy.log(i) for i in other_b_vec])#numpy.log(other_b_vec)

        lambdas=[]

        cs=[]
        cs.append( (base_residue,base_residue_sign) )

        ## extra orders
        for o in numpy.arange(int(order)-1)+1:
            # o is only factorial, thats why +1

            ## generate the lambdas .. 
            if(o%2==0):
                lambdas.append(scipy.misc.logsumexp(o*basic_one_over_log+after_exp_correction, b=numpy.ones(len(signs)-1).tolist() + [-1.0], return_sign=True))
                #print "oooo"
                #print "even: "
                #print "input ", o*basic_one_over_log+after_exp_correction, " signs.. ", numpy.ones(len(signs)-1).tolist() + [-1.0]
                #print scipy.misc.logsumexp(o*basic_one_over_log+after_exp_correction, b=numpy.ones(len(signs)-1).tolist() + [-1.0], return_sign=True)
                #print custom_logsumexp_mpmath(o*basic_one_over_log+after_exp_correction, numpy.array(numpy.ones(len(signs)-1).tolist() + [-1.0]) )
                #print  "ooooo"
                #lambdas.append(custom_logsumexp_mpmath(o*basic_one_over_log+after_exp_correction, numpy.array(numpy.ones(len(signs)-1).tolist() + [-1.0]) ))
            else:
                lambdas.append(scipy.misc.logsumexp(o*basic_one_over_log+after_exp_correction, b=signs, return_sign=True))
                #print "----"
                #print "odd:"
                #print "input ", o*basic_one_over_log+after_exp_correction, " signs.. ", numpy.ones(len(signs)-1).tolist() + [-1.0]
                #print scipy.misc.logsumexp(o*basic_one_over_log+after_exp_correction, b=signs, return_sign=True)
                #print custom_logsumexp_mpmath(o*basic_one_over_log+after_exp_correction, signs)
                #print "----"
                #lambdas.append(custom_logsumexp_mpmath(o*basic_one_over_log+after_exp_correction, signs))
            new_cs_list=[]
            new_cs_sign=[]

 
            for sumlen in range(o):
                new_cs_list.append( lambdas[sumlen][0]+cs[o-1-sumlen][0]    )
                new_cs_sign.append(1.0) if (lambdas[sumlen][1] == cs[o-1-sumlen][1] ) else  new_cs_sign.append(-1.0)

            
            new_cs_sign=numpy.array(new_cs_sign)
            new_cs_list=numpy.array([float(c) for c in new_cs_list])

            cur_logsum, cur_sign=scipy.misc.logsumexp(new_cs_list, b=new_cs_sign, return_sign=True)
            #cur_logsum, cur_sign=custom_logsumexp_mpmath(new_cs_list, new_cs_sign)
            cur_logsum-=numpy.log(o)

            cs.append( (cur_logsum,cur_sign))
 
        return cs[-1]

    pol_deg=sum(multiplicities)+k_tot-1

    #### TUNE PRECISION PROPORTIONAL TO NUMERATOR DEGREE
    mpmath.mp.dps=15+200*pol_deg


    tot_sum_logs=[]
    tot_sum_logs_numpy=[]
    tot_signs=[]

    tot_res_sum=0.0

    if(len(poles)==1 and multiplicities[0]==1):
        tot_sum_logs.append(mpmath.mpf(pol_deg*numpy.log(poles[0])))
        tot_signs.append(1.0)
    else:
        for ind in range(len(poles)):

            fac_nolog,sign=new_z_minus_z(poles[ind], numpy.array(poles[:ind].tolist()+poles[ind+1:].tolist()), numpy.array(multiplicities[:ind].tolist()+multiplicities[ind+1:].tolist()))


            upper_sign=1.0
            if(poles[ind]<0):
                upper_sign=-1.0
            

            basic_residue=pol_deg*mpmath.log(poles[ind])-mpmath.log((fac_nolog*sign))
            
            sign=float(sign)*float(upper_sign)

            ## multiplicity of 1 ... only need basic residue
            if(multiplicities[ind]==1):
                
                tot_sum_logs.append(basic_residue)
                tot_signs.append(sign)
            else:
                ## multiplicity higher than 1 .. need higher order terms, following http://dx.doi.org/10.1155/2014/895036
                high_residue,high_sign=get_high_order_residue(poles[ind], numpy.array(poles[:ind].tolist()+poles[ind+1:].tolist()),  numpy.array(multiplicities[:ind].tolist()+multiplicities[ind+1:].tolist())   ,   multiplicities[ind], float(basic_residue),sign , pol_deg )

                tot_sum_logs.append(mpmath.mpf(high_residue))
                tot_signs.append(high_sign)

    tot_signs=numpy.array(tot_signs)

    #print "final input: ", tot_sum_logs
    
    tot_sum_log, overall_sign=custom_logsumexp_mpmath(tot_sum_logs, tot_signs)

    return float(tot_sum_log)

