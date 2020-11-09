import mc1
import bcjr1
import numpy as np
def bcjr_markov(gen_prob,del_prob,noise_variance,number_blocks,length):
    p=gen_prob
    d=del_prob
    n=number_blocks
    k=length
    v=noise_variance
    i=mc1.markov.mcarr(n,k,p)
    i1=np.reshape(i,(n,k))
    i2=mc1.block.bpskmod(i1,1)
    i3=mc1.deletion.deletion(i2,d)
    i4=mc1.block.addnoise(i3,v)
    q=mc1.deletion.sample_prob(p,d)
    i5,i6,i7=bcjr1.markov_bcjr.bcjr_arr_out(i4,q,v)
    i8=mc1.deletion.deletion_mc_pad(i6,k,p,d)
    return i,i8
