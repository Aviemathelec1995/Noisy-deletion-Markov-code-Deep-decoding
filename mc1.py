

import numpy as np

class markov:
    
    
    def mc(n,d):
        p=np.random.randint(low=0,high=2,size=1)
        k=[]
        k.append(p[0])
        for i in range(1,n):
            if k[i-1]==0:
                k.append(np.random.choice([0,1],p=[1-d,d]))
            elif k[i-1]==1 :
                k.append(np.random.choice([0,1],p=[d,1-d]))
        return k
    def mc_flip(n,d):
        k=markov.mc(n,d)
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
                k.append(np.random.choice([0,1],p=[1-d,d]))
                k.append(np.random.choice([0,1],p=[1-d,d]))
                k1.append(np.random.choice([0,1],p=[1-d,d]))
            elif k[2*i-1]==1 :
                k.append(np.random.choice([0,1],p=[d,1-d]))
                k.append(np.random.choice([0,1],p=[d,1-d]))
                k1.append(np.random.choice([0,1],p=[d,1-d]))
        return k,k1
    def mcarr(n,k,d):
        o=[]
        for i in range(0,k):
            o=np.concatenate([o,markov.mc(n,d)])
            i=+1
        return o
    def mcarr_flip(n,k,d):
        o=[]
        for i in range(0,k):
            o=np.concatenate([o,markov.mc_flip(n,d)])
            i=+1
        return o
    
            
        
    
    def mcarr_rept2(n,k,d):
        o=[]
        o1=[]
        for i in range(0,k):
            p,p1=markov.mc_rept2(n,d)
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
    def hard_dec(n):
        l=np.shape(n)[0]
        p=np.zeros(l)
        for i in range(0,l):
            if n[i]<0:
                p[i]=0
            else:
                p[i]=1
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
            
            s1=np.delete(s,c)
            o.append(s1)
            continue
        return o
    def deletion_mark(data,k):
        l1=np.shape(data)[0]
        l2=np.shape(data)[1]
        o=[]
        o1=[]
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
            
            s1=np.delete(s,c)
            o.append(s1)
            o1.append(c)
            continue
        return o,o1
    def deletion_mc_pad(data,max_pad,p,d):
        l1=np.shape(data)[0]
        p1=1-p
        q1=1-((1-p1)/(1+d*(1-2*p1)))
        q=1-q1
        o=[]
        for i in range(0,l1):
            l=len(data[i])
        
            n=max_pad-l
            if n==0:
                o.append(data[i])
            else:
                m=markov.mc_flip(n,q)
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
    def zero_pad(data,max_pad,p,d):
        l1=np.shape(data)[0]
        
        o=[]
        for i in range(0,l1):
            l=len(data[i])
        
            n=max_pad-l
            if n==0:
                o.append(data[i])
            else:
                m=np.zeros(n)
                o.append(np.concatenate([data[i],m]))
        return o
    def deletion_mc_modpad(data,max_pad,p,d):
        l1=np.shape(data)[0]
        p1=1-p
        q1=1-((1-p1)/(1+d*(1-2*p1)))
        q=1-q1
        o=[]
        for i in range(0,l1):
            l=len(data[i])
        
            n=max_pad-l
            if n==0:
                o.append(data[i])
            else:
                m=markov.mc_flip(n,q)
                m1=2*m-1
                o.append(np.concatenate([data[i],m1]))
        return o
    def del_str(del_list,length):
        d=del_list
        l=length
        l1=np.shape(d)[0]
        o=[]
        for i in range(0,l1):
            o1=np.zeros(l)
            c=d[i]
            for j in range(0,len(c)):
                o1[c[j]]=1
            o.append(o1)
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
            o.append(mod_demod.bpskdemod(inp[i]))
            continue
        return o
    def hard_dec(inp):
        s1=np.shape(inp)[0]
        o=[]
        for i in range(0,s1):
            o.append(mod_demod.hard_dec(inp[i]))
            continue
        return o
        
    def substitution(inp,s):
        s1=np.shape(inp)[0]
        d=[]
        for i in range(0,s1):
            d.append(np.random.choice([int(inp[i]),int(np.mod(inp[i]+1,2))],p=[1-s,s]))
            continue
        return np.array(d)
    def subs_arr(inp,s):
        s1=np.shape(inp)[0]
        o=[]
        for i in range(0,s1):
            o.append(block.substitution(inp[i],s))
            continue
        return o
    def error_comp(i1,t1,c):
        l=np.shape(i1)[0]
        err=0
        count=0
        for j in range(0,l):
            for k in range(0,len(c[j])):
                err+=abs(i1[j][c[j][k]]-t1[j][c[j][k]])
            count+=len(c[j])
        return err/count
    
    
class conc:
    def rand_conc(inp,l):
        s1=np.shape(inp)[0]
        o=[]
        for x1 in range(0,s1):
            o.append(np.concatenate([inp[x1],np.random.randint(high=2,low=0,size=l)]))
        return o
    def random_msg(length,number):
        l=length
        n=number
        o=[]
        for i in range(0,n):
            o.append(np.random.randint(low=0,high=2,size=l))
        return o
class insertion:
    def insertion(data,k):
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
            s1=data[i]
            for j2 in range(0,len(c)):
                
                s1=np.insert(s1,c[j2],s1[c[j2]])
            o.append(s1)
        return o
    def chop(data,length):
        l1=np.shape(data)[0]
        o=[]
        for i in range(0,l1):
            o.append(data[i][0:length])
        return o
    def insertion_rand(data,k):
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
            s1=data[i]
            for j2 in range(0,len(c)):
                t=np.random.choice([0,1],p=[0.5,0.5])
                s1=np.insert(s1,t,s1[c[j2]])
            o.append(s1)
        return o
class insertion_deletion:
    def insert_del(data,k1,k2):
        o=deletion.deletion(data,k1)
        l1=np.shape(data)[0]
        o1=[]
        for i in range(0,l1):
            d=[]
            s=[]
            s1=[]
            c=[]
            count=0
            l2=len(o[i])
            for j in range(0,l2):
                d.append(np.random.choice([0,5],p=[1-k2,k2]))
            s=o[i]+d
            for j1 in range(0,l2):
                if s[j1]>3:
                    c.append(j1)
            s1=o[i]
            for j2 in range(0,len(c)):
                t=np.random.choice([0,1],p=[0.5,0.5])
                s1=np.insert(s1,t,s1[c[j2]])
            o1.append(s1)
        return o1
    def chop_pad(data,max_length,p,k1):
        l1=np.shape(data)[0]
        o=[]
        for i in range(0,l1):
            if len(data[i])>max_length:
                o.append(data[i][0:max_length])
            else:
                o.append(data[i])
        
        o1=deletion.deletion_mc_pad(o,max_length,p,k1)
        return o1
    
            
class encode:
    def von_neu(x):
        l=np.shape(x)[0]
        s=[]
        
        r=int(l/2)
        for j in range(0,r):
            if x[2*j]!=x[2*j+1]:
                s.append(int(x[2*j+1]))
        return s
    def von_neu1(x):
        l=np.shape(x)[0]
        u=[]
        
        r=int(l/2)
        for j in range(0,r):
            u.append(int(np.mod(x[2*j]+x[2*j+1],2)))
        return u
        
    def von_neu2(x):
        l=np.shape(x)[0]
        v=[]
        
        r=int(l/2)
        for j in range(0,r):
            if x[2*j]==x[2*j+1]:
                v.append(int(x[2*j+1]))
        return v
    def von_arr(x):
        l1=np.shape(x)[0]
        l2=np.shape(x)[1]
        o=[]
        for i in range(0,l1):
            s=encode.von_neu(x[i])
            o.append(s)
        return o
    def peres(x,t):
        s=encode.von_neu(x)
        u=encode.von_neu1(x)
        v=encode.von_neu2(x)
        count=1
        s=np.array(s)
        s1=np.array(encode.von_neu(u))
        s2=np.array(encode.von_neu(v))
        o=np.concatenate([s,s1,s2])
        for count in range(1,t):
            
            u1=encode.von_neu1(o)
            v1=encode.von_neu2(o)
            s1=np.array(encode.von_neu(u1))
            s2=np.array(encode.von_neu(v1))
            o=np.concatenate([s,s1,s2])
            count+=1
        return o
    def peres_arr(x,t):
        l1=np.shape(x)[0]
        o=[]
        for i in range(0,l1):
            o.append(encode.peres(x[i],t))
        return o
class conv:
    def encode(x):
        l=len(x)
        o=[]
        x=np.insert(x,len(x),0)
        x=np.insert(x,len(x),0)
        for i in range(0,l+2):
            if i<2:
                o.append(x[i])
                if i==0:
                    o.append(x[i])
                else:
                    o.append(np.mod(x[i]+x[i-1],2))
            else:
                o.append(np.mod(x[i]+x[i-2],2))
                o.append(np.mod(x[i]+x[i-1]+x[i-2],2))
        return o
    
        
    
            
        
        
    
            
                
        
                
            
    
    
            
            
    
