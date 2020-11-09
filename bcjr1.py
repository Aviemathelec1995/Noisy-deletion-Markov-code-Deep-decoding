import numpy as np
class markov_bcjr:
    def int_dist(p1):
        return np.array([p1,1-p1])
    def state_change(p):
        return np.array([[1-p,p],[p,1-p]])
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
    
    def form_path(inp,variance,p,p1):
        t=markov_bcjr.state_change(p)
        k=markov_bcjr.int_dist(p1)
        
        l=len(inp)
        v=variance
        
        l0=[]
        l1=[]
        l2=[]
        d=[]
        d1=[]
        h=inp[0]
        f=[]
        s=[]
        
        l0.append(np.log(k[0]*markov_bcjr.soft_prob(v,h,-1)))
        l1.append(np.log(k[1]*markov_bcjr.soft_prob(v,h,+1)))        
        d.append(max([l0[0],l1[0]]))
        a=np.array([l0[0],l1[0]])
        d1.append(2*(a.argmax())-1)
        l2.append(l0[0]-l1[0])
        f.append(markov_bcjr.hard_dec(-l2[0]))
        s.append(markov_bcjr.hard_dec(l2[0]))
        for j in range(1,l):
            a=[]
            h=inp[j]
            
            e0=markov_bcjr.soft_prob(v,h,-1)
            e1=markov_bcjr.soft_prob(v,h,+1)
            f0=markov_bcjr.likelihood_zero(l2[j-1])
            f1=markov_bcjr.likelihood_one(l2[j-1])
            l0.append(np.log(t[0][0]*e0*f0+t[1][0]*e0*f1))
            l1.append(np.log(t[0][1]*e1*f0+t[1][1]*e1*f1))
            l2.append(np.log(t[0][0]*e0*f0+t[1][0]*e0*f1)-np.log(t[0][1]*e1*f0+t[1][1]*e1*f1))
            d.append(max([l0[j],l1[j]]))
            a=np.array([l0[j],l1[j]])
            d1.append(2*(a.argmax())-1)
            f.append(markov_bcjr.hard_dec(-l2[j]))
            s.append(markov_bcjr.likelihood_one(l2[j]))
            continue
        return l0,l1,d,d1,l2,f,s
    def markov_seq_path(inp,e,p,p1):
        l=np.shape(inp)[0]
        l1=np.shape(inp)[1]
        o0=[]
        o1=[]
        o2=[]
        o3=[]
        o4=[]
        o5=[]
        o6=[]
                          
        for i in range(0,l):
            k0,k1,d,d1,d2,d3,d4=markov_bcjr.form_path(inp[i],e,p,p1)
            
            o0.append(k0)
            o1.append(k1)
            o2.append(d)
            o3.append(d1)
            o4.append(d2)
            o5.append(d3)
            o6.append(d4)
        return o0,o1,o2,o3,o4,o5,o6
    def compute_gamma(inp,p,variance):
        g0=np.zeros((np.shape(inp)[0],2))
        g1=np.zeros((np.shape(inp)[0],2))
        s=markov_bcjr.state_change(p)
        v=variance
        g0[0][0]=np.log(0.5*markov_bcjr.soft_prob(v,inp[0],-1))
        g1[0][1]=np.log(0.5*markov_bcjr.soft_prob(v,inp[0],+1))
        l=len(inp)
        for i in range(1,l):
            g0[i][0]=np.log(s[0][0]*markov_bcjr.soft_prob(v,inp[i],-1))
            g0[i][1]=np.log(s[0][1]*markov_bcjr.soft_prob(v,inp[i],+1))
            g1[i][0]=np.log(s[1][0]*markov_bcjr.soft_prob(v,inp[i],-1))
            g1[i][1]=np.log(s[1][1]*markov_bcjr.soft_prob(v,inp[i],+1))
            continue
        return g0,g1
    
        
        
    def forward_recursion(inp,p,variance):
        g0,g1=markov_bcjr.compute_gamma(inp,p,variance)
        a0=np.zeros(np.shape(inp)[0])
        a1=np.zeros(np.shape(inp)[0])
        l=len(inp)
        a0[0]=np.log(1*np.exp(g0[0][0]))
        a1[0]=np.log(1*np.exp(g1[0][1]))
        
        for i in range(1,l):
            a0[i]=max(a0[i-1]+g0[i][0],a1[i-1]+g1[i][0])
            a1[i]=max(a0[i-1]+g0[i][1],a1[i-1]+g1[i][1])
            
            continue
        return a0,a1
    def backward_recursion(inp,p,variance):
        g0,g1=markov_bcjr.compute_gamma(inp,p,variance)
        l=len(inp)
        b0=np.zeros(np.shape(inp)[0])
        b1=np.zeros(np.shape(inp)[0])
        b0[l-1]=np.log(1*0.5)
        b1[l-1]=np.log(1*0.5)
       
        for i in range(0,l-1):
            b0[l-2-i]=max(b0[l-1-i]+g0[l-i-1][0],b1[l-1-i]+g0[l-1-i][1])
            b1[l-2-i]=max(b0[l-1-i]+g1[l-i-1][0],b1[l-1-i]+g1[l-1-i][1])
            
            continue
        return b0,b1
    def bcjr_out(inp,p,variance):
       
        if len(inp)==0:
            return inp
        else:
            g0,g1=markov_bcjr.compute_gamma(inp,p,variance)
            a0,a1=markov_bcjr.forward_recursion(inp,p,variance)
            b0,b1=markov_bcjr.backward_recursion(inp,p,variance)
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
                r0=max(a0[i-1]+g0[i][0]+b0[i],a1[i-1]+g1[i][0]+b0[i])
                r1=max(a0[i-1]+g0[i][1]+b1[i],a1[i-1]+g1[i][1]+b1[i])
                r=(r0-r1)
                l1.append(r)
                l2.append(markov_bcjr.hard_dec(l1[i]))
                l3.append(1-markov_bcjr.likelihood_zero(r))
                continue
            return l1,l2,l3
    def bcjr_arr_out(in_data,p,variance):
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
                l1,l2,l3=markov_bcjr.bcjr_out(in_data[i],p,variance)
                o1.append(l1)
                o2.append(l2)
                o3.append(l3)
                continue
        return o1,o2,o3
    
       
        
            
            
            
    
            
            
            
        
