"""Experiment 5 from MSDS458_Assignment1.ipynb, retrained without TensorFlow.

The notebook trains its networks with Keras but does not save them. This script rebuilds
the best one (Experiment 5) in plain NumPy so its weights can be exported and used
elsewhere, such as the browser demo at
https://jamesquantai.com/projects/digit-recognition.html

Same recipe as the notebook:
- 784 inputs -> Dense 128 ReLU -> batch norm -> Dense 64 ReLU -> batch norm
  -> Dense 32 ReLU -> batch norm -> Dense 10 softmax
- Adam (learning rate 0.001), batch size 32, 30 epochs
- first 5,000 training images held out for validation, 55,000 used for training

It reaches 98.1% test accuracy, matching the notebook's 98.08%.

Usage:
    python train_experiment5_numpy.py /path/to/mnist/

The folder must hold the four original MNIST files (train-images-idx3-ubyte.gz,
train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz).
The path needs its trailing slash. Training takes about 90 seconds on a laptop.

Output: digits_model.json with the weights (little-endian float32, base64, row-major
inputs x outputs), batch norm folded into a scale and shift per layer, the training
history, the test confusion matrix and ten sample test digits. A copy from one run is
saved in this repo as experiment5_weights.json.
"""
import numpy as np, gzip, json, base64, time, sys
D=sys.argv[1]
def imgs(f): return np.frombuffer(gzip.open(D+f).read(),np.uint8,offset=16).reshape(-1,784).astype(np.float32)/255
def labs(f): return np.frombuffer(gzip.open(D+f).read(),np.uint8,offset=8).astype(np.int64)
X=imgs('train-images-idx3-ubyte.gz'); y=labs('train-labels-idx1-ubyte.gz'); Xt=imgs('t10k-images-idx3-ubyte.gz'); yt=labs('t10k-labels-idx1-ubyte.gz')
Xv,yv,X,y=X[:5000],y[:5000],X[5000:],y[5000:]   # same split as the notebook
rng=np.random.default_rng(42)
sizes=[784,128,64,32,10]
W=[];b=[]
for i in range(4):
    lim=np.sqrt(6/(sizes[i]+sizes[i+1])); W.append(rng.uniform(-lim,lim,(sizes[i],sizes[i+1])).astype(np.float32)); b.append(np.zeros(sizes[i+1],np.float32))
G=[np.ones(s,np.float32) for s in sizes[1:4]]; Bt=[np.zeros(s,np.float32) for s in sizes[1:4]]
RM=[np.zeros(s,np.float32) for s in sizes[1:4]]; RV=[np.ones(s,np.float32) for s in sizes[1:4]]
MOM,EPS=0.99,1e-3
params=W+b+G+Bt
m=[np.zeros_like(p) for p in params]; v=[np.zeros_like(p) for p in params]; t=0
def forward(x,train):
    cache=[];a=x
    for i in range(3):
        z=a@W[i]+b[i]; h=np.maximum(z,0)
        if train:
            mu=h.mean(0); var=h.var(0); RM[i]=MOM*RM[i]+(1-MOM)*mu; RV[i]=MOM*RV[i]+(1-MOM)*var
        else: mu,var=RM[i],RV[i]
        hn=(h-mu)/np.sqrt(var+EPS); o=G[i]*hn+Bt[i]
        cache.append((a,z,h,mu,var,hn)); a=o
    z=a@W[3]+b[3]; z-=z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True)
    return p,cache,a
def acc(Xs,ys):
    p,_,_=forward(Xs,False); return float((p.argmax(1)==ys).mean()), float(-np.log(p[np.arange(len(ys)),ys]+1e-12).mean())
t0=time.time(); hist=[]
for ep in range(30):
    idx=rng.permutation(len(X)); tl=0;tc=0
    for s in range(0,len(X),32):
        bi=idx[s:s+32]; xb=X[bi]; yb=y[bi]; n=len(bi)
        p,cache,alast=forward(xb,True)
        tl+=-np.log(p[np.arange(n),yb]+1e-12).sum(); tc+=(p.argmax(1)==yb).sum()
        dz=p.copy(); dz[np.arange(n),yb]-=1; dz/=n
        gW=[None]*4; gb=[None]*4; gG=[None]*3; gB=[None]*3
        gW[3]=alast.T@dz; gb[3]=dz.sum(0); do=dz@W[3].T
        for i in (2,1,0):
            a,z,h,mu,var,hn=cache[i]
            gG[i]=(do*hn).sum(0); gB[i]=do.sum(0)
            dhn=do*G[i]; std=np.sqrt(var+EPS)
            dh=(dhn-dhn.mean(0)-hn*(dhn*hn).mean(0))/std
            dzz=dh*(z>0)
            gW[i]=a.T@dzz; gb[i]=dzz.sum(0)
            if i>0: do=dzz@W[i].T
        grads=gW+gb+gG+gB; t+=1
        lr=1e-3*np.sqrt(1-0.999**t)/(1-0.9**t)
        for k in range(len(params)):
            m[k]=0.9*m[k]+0.1*grads[k]; v[k]=0.999*v[k]+0.001*grads[k]*grads[k]
            params[k]-=lr*m[k]/(np.sqrt(v[k])+1e-7)
    va,vl=acc(Xv,yv); hist.append([round(float(tl/len(X)),4),round(float(tc/len(X)),4),round(vl,4),round(va,4)])
    print(ep+1,hist[-1],round(time.time()-t0),'s',flush=True)
ta,tl_=acc(Xt,yt); print('TEST acc',ta,'loss',tl_)
p,_,_=forward(Xt,False); pred=p.argmax(1)
cm=np.zeros((10,10),int)
for a_,b_ in zip(yt,pred): cm[a_,b_]+=1
def enc(a): return base64.b64encode(np.asarray(a,'<f4').tobytes()).decode()
# fold batch norm into a scale and shift per layer for inference
layers=[]
for i in range(3):
    sc=G[i]/np.sqrt(RV[i]+EPS); sh=Bt[i]-RM[i]*sc
    layers.append({'in':sizes[i],'out':sizes[i+1],'W':enc(W[i]),'b':enc(b[i]),'scale':enc(sc),'shift':enc(sh)})
layers.append({'in':32,'out':10,'W':enc(W[3]),'b':enc(b[3])})
samples=[]
for dgt in range(10):
    k=int(np.where(yt==dgt)[0][3]); samples.append({'label':dgt,'pixels':base64.b64encode((Xt[k]*255).astype(np.uint8).tobytes()).decode(),'probs':[round(float(q),4) for q in p[k]]})
json.dump({'testAccuracy':round(ta,4),'testLoss':round(tl_,4),'history':hist,'confusion':cm.tolist(),'layers':layers,'samples':samples},open('digits_model.json','w'))
