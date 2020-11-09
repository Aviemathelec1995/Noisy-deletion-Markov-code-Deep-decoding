

import numpy as np

class watermark:
    
    def gen_prob(del_prob,insertion_prob,max_insert_length,past_drift,drift):
        b=drift
        a=past_drift
        I=max_insert_length
        d=del_prob
        i=insertion_prob
        t=1-d-i
        if I==0:
            n=0
        else:
            n=1/(1-i**I)
        
        if b-a==-1:
            p=d
        elif b==a:
            p=n*i*d+t
        elif 0<b-a<I :
            p=n*(np.power(i,(b-a+1))*d+np.power(i,(b-a))*t)
        elif b-a==I:
            p=n*np.power(i,I)*t
        else:
            p=0
        return p
    
    def gen_output_dist(del_prob,insertion_prob,effective_subs,max_insert_length,past_drift,drift,position,received_vector,watermark_vector):
        r=received_vector
        w=watermark_vector
        b=drift
        a=past_drift
        I=max_insert_length
        d=del_prob
        f=effective_subs
        i=insertion_prob
        t=1-d-i
        
        j=position
        if b-a>=0 and j+b+1<=len(r):
            y=r[j+a:j+b+1]
        if i==0:
            i1=0
            i2=0
        else:
            i1=np.power(i,(b-a+1))
            i2=np.power(i,(b-a))
        p=watermark.gen_prob(d,i,I,a,b)
        
        if j+b+1<=len(r)and i>0:
            if b-a>=0 and j+b>=0  and b-a<I :
                if r[b+j]==w[j]:
                    q= i1*d*(1/(2**(b-a+1)))+i2*t*(1-f)*(1/(2**(b-a)))
                else:
                    q= i1*d*(1/(2**(b-a+1)))+i2*t*f*(1/(2**(b-a)))
            elif b-a==-1:
                q=p
            elif b-a==I and j+b>=0:
                if r[b+j]==w[j]:
                    q= i2*t*(1-f)*(1/(2**(b-a)))
                else:
                    q= i2*t*f*(1/(2**(b-a)))
                
            
            else:
                q=0
        elif i==0 and j+b+1<=len(r):
            if b-a==0 and j+b>=0 :
                if r[b+j]==w[j]:
                    q=t*(1-f)
                else:
                    q=t*f
            elif b-a==-1:
                q=p
            else:
                q=0
        else:
            q=0
        if p!=0:
            return q/p
        else:
            return 0
   
    def forward_recur(del_prob,insertion_prob,effective_subs,max_insert_length,received_vector,watermark_vector):
        r=received_vector
        w=watermark_vector
        
        I=max_insert_length
        d=del_prob
        s=effective_subs
        i=insertion_prob
        t=1-d-i
        l=len(w)
        
        f=np.zeros(shape=((I+2)*(l+1),l+1))
        ra=(I+2)*(l+1)
        f[1+l][0]=1
        for k1 in range(0,ra):
            if k1-1-l<=I:
                f[k1][1]=f[1+l][0]*watermark.gen_prob(d,i,I,l+1-1,k1-1)*watermark.gen_output_dist(d,i,s,I,l+1-1-l,k1-1-l,0,r,w)
            else:
                f[k1][1]=0
        for k2 in range(2,l+1):
            for k1 in range(0,ra):
                
                if k1-I>=-1 and k1+1<=ra-1:
                    for u in range(k1-I,k1+2):
                        f[k1][k2]=f[k1][k2]+f[u][k2-1]*watermark.gen_prob(d,i,I,u-l-1,k1-l-1)*watermark.gen_output_dist(d,i,s,I,u-l-1,k1-l-1,k2-1,r,w)
                elif k1-I<1 :
                    for u in range(0,k1+2):
                        f[k1][k2]=f[k1][k2]+f[u][k2-1]*watermark.gen_prob(d,i,I,u-l-1,k1-l-1)*watermark.gen_output_dist(d,i,s,I,u-l-1,k1-l-1,k2-1,r,w)
                elif k1==ra-1:
                    for u in range(0,k1+1):
                        f[k1][k2]=f[k1][k2]+f[u][k2-1]*watermark.gen_prob(d,i,I,u-l-1,k1-l-1)*watermark.gen_output_dist(d,i,s,I,u-l-1,k1-l-1,k2-1,r,w)
                
        return f
    def backward_recur(del_prob,insertion_prob,effective_subs,max_insert_length,received_vector,watermark_vector):
        r=received_vector
        w=watermark_vector
        
        I=max_insert_length
        d=del_prob
        s=effective_subs
        i=insertion_prob
        t=1-d-i
        l=len(w)
        
        b=np.zeros(shape=((I+2)*(l+1),l+1))
        ra=(I+2)*(l+1)
        diff=len(r)-len(w)
        
        b[l+1+diff][l]=1
        for k2 in range(0,l):
            for k1 in range(0,ra):
                if k1+I<=ra-1 and k1-1>=0:
                    for u in range(k1-1,k1+I+1):
                        b[k1][l-k2-1]=b[k1][l-k2-1]+b[u][l-k2]*watermark.gen_prob(d,i,I,k1-1-l,u-1-l)*watermark.gen_output_dist(d,i,s,I,k1-1-l,u-1-l,l-k2-1,r,w)
                
        
        return b
    def likelihood_gen(forward_mat,backward_mat,del_prob,insertion_prob,substitution_prob,effective_subs,max_insert_length,received_vector,sparse_watermark_vector,watermark_vector,position,block_size):
        r=received_vector
        sp=sparse_watermark_vector
        n=block_size
        e=effective_subs
        w=watermark_vector
        
        I=max_insert_length
        d=del_prob
        s=substitution_prob
        i=insertion_prob
        t=1-d-i
        l=len(w)
        p=position
        p1=p+block_size
        ra=(I+2)*(l+1)
        f=forward_mat
        b=backward_mat
        f1=np.zeros(shape=np.shape(f))
        
        f1[:,p]=f[:,p]
        sp1=watermark.vec_add(sp,w[p:p1])
        sp2=np.concatenate([np.zeros(p),sp1,np.zeros(l-p1)])
        
        for k2 in range(p+1,p1+1):
            for k1 in range(0,ra):
                
                if k1-I>=-1 and k1+1<=ra-1:
                    for u in range(k1-I,k1+2):
                        f1[k1][k2]=f1[k1][k2]+f1[u][k2-1]*watermark.gen_prob(d,i,I,u-l-1,k1-l-1)*watermark.gen_output_dist(d,i,s,I,u-l-1,k1-l-1,k2-1,r,sp2)
                elif k1-I<1 :
                    for u in range(0,k1+2):
                        f1[k1][k2]=f1[k1][k2]+f1[u][k2-1]*watermark.gen_prob(d,i,I,u-1,k1-1)*watermark.gen_output_dist(d,i,s,I,u-1,k1-1,k2-1,r,sp2)
                elif k1==ra-1:
                    for u in range(0,k1+1):
                        f1[k1][k2]=f1[k1][k2]+f1[u][k2-1]*watermark.gen_prob(d,i,I,u-1,k1-1)*watermark.gen_output_dist(d,i,s,I,u-1,k1-1,k2-1,r,sp2)
                
        z=np.multiply(b,f1)
        count=0
        for k1 in range(0,ra):
            count=count+z[k1][p1]
        return count
    
    
    def bcjr_dec(forward_mat,backward_mat,del_prob,insertion_prob,substitution_prob,effective_subs,max_insert_length,received_vector,sparse_list,watermark_vector):
        r=received_vector
        sp=sparse_list
        e=effective_subs
        w=watermark_vector
        f=forward_mat
        b=backward_mat
        I=max_insert_length
        d=del_prob
        s=substitution_prob
        i=insertion_prob
        t=1-d-i
        l=len(w)
        sh1=np.shape(sp)[0]
        sh2=np.shape(sp)[1]
        ra=int(l/sh2)
        o=[]
        for k1 in range(0,ra):
            c=[]
            for k2 in range(0,sh1):
                c.append(watermark.likelihood_gen(f,b,d,i,s,e,I,r,sp[k2],w,k1*sh2,sh2))
            o=np.concatenate([o,sp[np.argmax(c)]])
                       
        return o

    def sparse_gen(n):
        l=len(n)
        r=int(l/2)
        o=[]
        for i in range(0,r):
            if n[2*i]==0 and n[2*i+1]==0:
                o=np.concatenate([o,[0,0,0,0]])
            if n[2*i]==0 and n[2*i+1]==1:
                o=np.concatenate([o,[0,1,0,0]])
            if n[2*i]==1 and n[2*i+1]==0:
                o=np.concatenate([o,[0,0,1,0]])
            if n[2*i]==1 and n[2*i+1]==1:
                o=np.concatenate([o,[0,0,0,1]])
        return o
    def sparse_gen1(n):
        l=len(n)
        r=int(l)
        o=[]
        for i in range(0,r):
            if n[i]==0:
                o=np.concatenate([o,[0,0,0,0]])
            if n[i]==1:
                o=np.concatenate([o,[0,0,0,1]])
        return o
    
    def vec_add(n1,n2):
        l=len(n1)
        o=[]
        for i in range(0,l):
            o.append(int(np.mod(n1[i]+n2[i],2)))
        return np.array(o)
            
    def watermark_add(inp,w):
        l1=np.shape(inp)[0]
        o=[]
        for i in range(0,l1):
            o.append(watermark.vec_add(inp[i],w))
        return o
    def bcjr_arr(del_prob,insertion_prob,substitution_prob,effective_subs,max_insert_length,received_vector,sparse_list,watermark_vector):
        r=received_vector
        l1=np.shape(r)[0]
        o=[]
        for i in range(0,l1):
            f=watermark.forward_recur(del_prob,insertion_prob,effective_subs,max_insert_length,r[i],watermark_vector)
            b=watermark.backward_recur(del_prob,insertion_prob,effective_subs,max_insert_length,r[i],watermark_vector)
            o.append(watermark.bcjr_dec(f,b,del_prob,insertion_prob,substitution_prob,effective_subs,max_insert_length,r[i],sparse_list,watermark_vector))
        return o
    
    
    def sparse_back(n,sparse_list):
        sp=sparse_list
        l=len(n)
        r=int(l/4)
        o=[]
        for k1 in range(0,r):
            if np.array_equal(n[k1*4:k1*4+4],sp[0])==True:
                o=np.concatenate([o,[0,0]])
            if np.array_equal(n[k1*4:k1*4+4],sp[1])==True:
                o=np.concatenate([o,[0,1]])

            if np.array_equal(n[k1*4:k1*4+4],sp[2])==True:
                o=np.concatenate([o,[1,0]])

            if np.array_equal(n[k1*4:k1*4+4],sp[3])==True:
                o=np.concatenate([o,[1,1]])
        return o
    def sparse_gen2(n):
        l=len(n)
        r=int(l/2)
        o=[]
        for i in range(0,r):
            if n[2*i]==0 and n[2*i+1]==0:
                o=np.concatenate([o,[0,0,0,0,0,0]])
            if n[2*i]==0 and n[2*i+1]==1:
                o=np.concatenate([o,[0,0,0,1,0,0]])
            if n[2*i]==1 and n[2*i+1]==0:
                o=np.concatenate([o,[0,0,0,0,1,0]])
            if n[2*i]==1 and n[2*i+1]==1:
                o=np.concatenate([o,[0,0,0,0,0,1]])
        return o
            
            
    
        
            
            
                
                
        
    
    
    
    

 
   
        
