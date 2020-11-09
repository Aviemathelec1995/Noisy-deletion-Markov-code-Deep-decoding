import numpy as np
class markov_bcjr:
    def int_dist(p1):
        return np.array([p1,1-p1])
    def state_change(p2,p3):
        return np.array([[1-p2,p2],[p3,1-p3]])
    def likelihood_zero(u):
        return (np.exp(u)/(np.exp(u)+1))
    def likelihood_one(u):
        return (1/(np.exp(u)+1))
    def soft_prob(variance,x,a):
            
        return (1/np.sqrt(2*np.pi*variance))*np.exp(-(x*x+a*a-2*a*x)/(2*variance))
        
    def hard_dec(x):
        if x<0:
            return 1
        else:
            return 0
    
    def hard_prob(e,x,a):
        if x==1 and a==1:
            return (1-e)
        if x==1 and a==0:
            return e
        if x==0 and a==1:
            return e
        if x==0 and a==0:
            return (1-e)
        
    def compute_gamma(inp,p2,p3,e):
        g0=np.zeros((np.shape(inp)[0],2))
        g1=np.zeros((np.shape(inp)[0],2))
        s=markov_bcjr.state_change(p2,p3)
        
        g0[0][0]=np.log(0.5*markov_bcjr.hard_prob(e,inp[0],0))
        g1[0][1]=np.log(0.5*markov_bcjr.hard_prob(e,inp[0],1))
        l=len(inp)
        for i in range(1,l):
            g0[i][0]=np.log(s[0][0]*markov_bcjr.hard_prob(e,inp[i],0))
            g0[i][1]=np.log(s[0][1]*markov_bcjr.hard_prob(e,inp[i],1))
            g1[i][0]=np.log(s[1][0]*markov_bcjr.hard_prob(e,inp[i],0))
            g1[i][1]=np.log(s[1][1]*markov_bcjr.hard_prob(e,inp[i],1))
            continue
        return g0,g1
    
        
        
    def forward_recursion(inp,p2,p3,e):
        g0,g1=markov_bcjr.compute_gamma(inp,p2,p3,e)
        a0=np.zeros(np.shape(inp)[0])
        a1=np.zeros(np.shape(inp)[0])
        l=len(inp)
        a0[0]=np.log(1*np.exp(g0[0][0]))
        a1[0]=np.log(1*np.exp(g1[0][1]))
        
        for i in range(1,l):
            a0[i]=np.log(np.exp(a0[i-1]+g0[i][0])+np.exp(a1[i-1]+g1[i][0]))
            a1[i]=np.log(np.exp(a0[i-1]+g0[i][1])+np.exp(a1[i-1]+g1[i][1]))
            
            continue
        return a0,a1
    def backward_recursion(inp,p2,p3,e):
        g0,g1=markov_bcjr.compute_gamma(inp,p2,p3,e)
        l=len(inp)
        b0=np.zeros(np.shape(inp)[0])
        b1=np.zeros(np.shape(inp)[0])
        b0[l-1]=np.log(1*(p3/(p2+p3)))
        b1[l-1]=np.log(1*(p2/(p2+p3)))
       
        for i in range(0,l-1):
            b0[l-2-i]=np.log(np.exp(b0[l-1-i]+g0[l-i-1][0])+np.exp(b1[l-1-i]+g0[l-1-i][1]))
            b1[l-2-i]=np.log(np.exp(b0[l-1-i]+g1[l-i-1][0])+np.exp(b1[l-1-i]+g1[l-1-i][1]))
            
            continue
        return b0,b1
    def bcjr_out(inp,p2,p3,e):
       
        if len(inp)==0:
            return inp
        else:
            g0,g1=markov_bcjr.compute_gamma(inp,p2,p3,e)
            a0,a1=markov_bcjr.forward_recursion(inp,p2,p3,e)
            b0,b1=markov_bcjr.backward_recursion(inp,p2,p3,e)
            l=len(inp)
            l1=[]
            l2=[]
            l3=[]
            r0=g0[0][0]+b0[0]
            r1=g1[0][1]+b1[0]
            r=(r0-r1)
            l1.append((r))
            l2.append(markov_bcjr.hard_dec(l1[0]))
            l3.append(1-markov_bcjr.likelihood_zero(r))
            for i in range(1,l):
                r0=np.log(np.exp(a0[i-1]+g0[i][0]+b0[i])+np.exp(a1[i-1]+g1[i][0]+b0[i]))
                r1=np.log(np.exp(a0[i-1]+g0[i][1]+b1[i])+np.exp(a1[i-1]+g1[i][1]+b1[i]))
                r=(r0-r1)
                l1.append(r)
                l2.append(markov_bcjr.hard_dec(l1[i]))
                l3.append(1-markov_bcjr.likelihood_zero(r))
                continue
            return l1,l2,l3
    def bcjr_arr_out(in_data,p2,p3,e):
        s1=np.shape(in_data)[0]
        o1=[]
        o2=[]
        o3=[]
        for i in range(0,s1):
            if len(in_data[i])==0:
                o1.append(in_data[i])
                o2.append(in_data[i])
                o3.append(in_data[i])
            else:
                l1,l2,l3=markov_bcjr.bcjr_out(in_data[i],p2,p3,e)
                o1.append(l1)
                o2.append(l2)
                o3.append(l3)
                continue
        return o1,o2,o3
    
       
        
            
            
            
    
            
            
            
        
