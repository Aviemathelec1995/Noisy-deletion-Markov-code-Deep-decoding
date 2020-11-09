

import numpy as np

class markov:
    
    
    def mc(n,d1,d2):
        if n==0:
            pass
        else:
            p=np.random.randint(low=0,high=2,size=1)
            k=[]
            k.append(p[0])
            for i in range(1,n):
                if k[i-1]==0:
                    k.append(np.random.choice([0,1],p=[1-d1,d1]))
                elif k[i-1]==1 :
                    k.append(np.random.choice([0,1],p=[d2,1-d2]))
            
            return k
    
    def mc_flip(n,d1,d2):
        k=markov.mc(n,d1,d2)
        return np.flip(k)
    
    def mc_rept2(n,d):
        p=np.random.randint(low=0,high=2,size=1)
        k=[]
        k1=[]
        k1.append(p[0])
        k.append(p[0])
        k.append(p[0])
        for i in range(1,n):
            if k[2*i-1]==0:
                k.append(np.random.choice([0,1],p=[1-d1,d1]))
                k.append(np.random.choice([0,1],p=[1-d1,d1]))
                k1.append(np.random.choice([0,1],p=[1-d1,d1]))
            elif k[2*i-1]==1 :
                k.append(np.random.choice([0,1],p=[d2,1-d2]))
                k.append(np.random.choice([0,1],p=[d2,1-d2]))
                k1.append(np.random.choice([0,1],p=[d2,1-d2]))
        return k,k1
    def mcarr(n,k,d1,d2):
        o=[]
        for i in range(0,k):
            o=np.concatenate([o,markov.mc(n,d1,d2)])
            i=+1
        return o
    def mcarr_flip(n,k,d1,d2):
        o=[]
        for i in range(0,k):
            o=np.concatenate([o,markov.mc_flip(n,d1,d2)])
            i=+1
        return o
    
            
        
    
    def mcarr_rept2(n,k,d1,d2):
        o=[]
        o1=[]
        for i in range(0,k):
            p,p1=markov.mc_rept2(n,d1,d2)
            o=np.concatenate([o,p])
            o1=np.concatenate([o1,p1])
            i+=1
        return o,o1
class mod_demod:
    
    def addnoise(n,variance):
        n1=np.random.normal(0,np.sqrt(variance),n.shape)
        return n+n1
    def bpskmod(n,a):
        l=np.shape(n)[0]
        p=np.zeros(l)
        for i in range(0,l):
            if n[i]==0:
                p[i]=-1*a
            else:
                p[i]=1*a
        return p
    def bpskdemod(n):
        l=np.shape(n)[0]
        p=np.zeros(l)
        for i in range(0,l):
            if n[i]<0:
                p[i]=-1
            else:
                p[i]=1
        return p
    def abs(n):
        l=np.shape(n)[0]
        p=np.zeros(l)
        count=0
        for i in range(0,l):
            if n[i]==0:
                p[i]=0
                count+=0
            else:
                p[i]=1
                count+=1
        return p
class deletion:
    def deletion(data,k):
        l1=np.shape(data)[0]
        l2=np.shape(data)[1]
        o=[]
        
        for i in range(0,l1):
            d=[]
            s=[]
            s1=[]
            c=[]
            count=0
            for j in range(0,l2):
                d.append(np.random.choice([0,5],p=[1-k,k]))
            s=data[i]+d
            for j1 in range(0,l2):
                if s[j1]>3:
                    c.append(j1)
            for j2 in range(0,len(c)):
                s1=np.delete(s,c)
            o.append(s1)
            continue
        return o
    
    def deletion_mc_pad(data,max_pad,p1,p2):
        l1=np.shape(data)[0]
        
        o=[]
        for i in range(0,l1):
            l=len(data[i])
            n=max_pad-l
            if n==0:
                o.append(data[i])
            else:
                m=markov.mc_flip(n,p1,p2)
                o.append(np.concatenate([data[i],m]))
        return o
    def sample_prob(p,d):
        p1=1-p
        q1=1-((1-p1)/(1+d*(1-2*p1)))
        q=1-q1
        return q
    def del_discard(inp):
        s1=np.shape(inp)[0]
        o=[]
        for i in range(0,s1):
            if len(inp[i])>0:
                o.append(inp[i])
            continue
        return o
    def del_arr(inp):
        s1=np.shape(inp)[0]
        o=[]
        for i in range(0,s1):
            if len(inp[i])==0:
                o.append(i)
        return o
    def arr_discard(in_data,inp):
        a=deletion.del_arr(inp)
        s1=np.shape(inp)[0]
        o=[]
        for i in range(0,s1):
            if i in set(a):
                pass
            else:
                o.append(in_data[i])
        return o
    
class  block:
    def bpskmod(inp,a):
        s1=np.shape(inp)[0]
        o=[]
        for i in range(0,s1):
            o.append(mod_demod.bpskmod(inp[i],a))
            continue
        return o
    def addnoise(inp,variance):
        s1=np.shape(inp)[0]
        o=[]
        for i in range(0,s1):
            if len(inp[i])==0:
                o.append(inp[i])
            else:
                o.append(mod_demod.addnoise(inp[i],variance))
            continue
        return o 
    def bpskdemod(inp):
        s1=np.shape(inp)[0]
        o=[]
        for i in range(0,s1):
            o.append(mod_demod.bpkdemod(inp[i]))
            continue
        return o
    def substitution(inp,s):
        s1=np.shape(inp)[0]
        d=[]
        for i in range(0,s1):
            d.append(np.random.choice([inp[i],np.mod(inp[i]+1,2)],p=[1-s,s]))
            continue
        return d
    def subs_arr(inp,s):
        s1=np.shape(inp)[0]
        o=[]
        for i in range(0,s1):
            o.append(block.substitution(inp[i],s))
            continue
        return o
class conc:
    def rand_conc(inp,l):
        s1=np.shape(inp)[0]
        o=[]
        for x1 in range(0,s1):
            o.append(np.concatenate([inp[x1],np.random.randint(high=2,low=0,size=l)]))
        return o
class sticky_deletion:
    def deletion(data,k1,k2):
        l1=np.shape(data)[0]
        l2=np.shape(data)[1]
        o=[]

        for i in range(0,l1):
            s=[]
            s1=[]
            c=[]
            count=0
            d=markov.mc(l2,k1,k2)
            d1=[x*5 for x in d]
            s=data[i]+d1
            for j1 in range(0,l2):
                if s[j1]>3:
                    c.append(j1)
            
            s1=np.delete(s,c)
            o.append(s1)
            continue
        return o
    def state_tran_del(state_tran,k1,k2):
        p=state_tran
        o=np.zeros(np.shape(p))
        o+=p*(1-k1)+np.dot(p,p)*k1*k2
        f=np.dot(p,p)
        for x1 in range(0,10):
            f=np.dot(f,p)
            o+=f*k1*k2*np.power((1-k2),x1+1)
        return o
    def mc_pad(data,max_pad,q1,q2):
        l1=np.shape(data)[0]
        o=[]
        for i in range(0,l1):
            l=len(data[i])
            n=max_pad-l
            if n==0:
                o.append(data[i])
            else:
                m=markov.mc_flip(n,q1,q2)
                o.append(np.concatenate([data[i],m]))
        return o
class entropy:
    
    def entr(p1):
        h=-(p1*np.log(p1)+(1-p1)*np.log(1-p1))/np.log(2)
        return h
    def cal_entr(p1,p2):
        h=(p2/(p1+p2))*entropy.entr(p1)+(p1/(p1+p2))*entropy.entr(p2)
        return h
    def entropy_pair(p1,rate):
        l1=[]
        l2=[]
        for i in range(0,1000):
            l1.append(abs(entropy.cal_entr(p1,(0.5+(i/2000)))-rate))
            l2.append(0.5+(i/2000))
        l1=np.array(l1)
        m=np.argmin(l1)
        return(l2[m])
    def entropy_pair1(p1,rate,range1,range2):
        r1=range1
        r2=range2
        l1=[]
        l2=[]
        for i in range(0,1000):
            l1.append(abs(entropy.cal_entr(p1,(r1+((r2-r1)*i)/1000)-rate)))
            l2.append(r1+((r2-r1)*i)/1000)
        l1=np.array(l1)
        m=np.argmin(l1)
        return (l2[m])
        
        

            
        
    
            
            
    
