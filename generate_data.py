#!/usr/bin/env python3
"""Precompute training trajectories for the Session-1 webapp.
Outputs data.js defining window.PRECOMP with quantized decision-boundary frames,
embedding snapshots, accuracies and weight matrices. Pure numpy (reproducible)."""
import numpy as np, json, base64

rng_global = np.random.RandomState(7)

# ---------------- data ----------------
def make_rings(n=300, noise=0.055, seed=7):
    r = np.random.RandomState(seed); m = n//2
    t = r.rand(m)*2*np.pi; rin = 0.30 + r.randn(m)*noise
    Xi = np.c_[rin*np.cos(t), rin*np.sin(t)]
    t2 = r.rand(n-m)*2*np.pi; rout = 0.70 + r.randn(n-m)*noise
    Xo = np.c_[rout*np.cos(t2), rout*np.sin(t2)]
    X = np.vstack([Xi, Xo]).astype(np.float64)
    y = np.r_[np.zeros(m), np.ones(n-m)].astype(np.float64)
    return X, y

def make_noisy(n, seed):
    r = np.random.RandomState(seed)
    X = r.randn(n, 2)
    y = (X[:,0]*X[:,1] > 0).astype(np.float64)
    flip = r.rand(n) < 0.10
    y = np.where(flip, 1-y, y)
    return X, y

# ---------------- tiny MLP (numpy + Adam) ----------------
ACT = {
    'relu':   (lambda z: np.maximum(0,z),           lambda z: (z>0).astype(z.dtype)),
    'tanh':   (lambda z: np.tanh(z),                lambda z: 1-np.tanh(z)**2),
    'linear': (lambda z: z,                          lambda z: np.ones_like(z)),
}
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))

class MLP:
    def __init__(self, sizes, acts, seed=0):
        r = np.random.RandomState(seed)
        self.W=[]; self.b=[]; self.acts=acts
        for i in range(len(sizes)-1):
            fan = sizes[i]
            scale = np.sqrt(2.0/fan) if acts[i]=='relu' else np.sqrt(1.0/fan)
            self.W.append(r.randn(sizes[i], sizes[i+1])*scale)
            self.b.append(np.zeros(sizes[i+1]))
        self.mW=[np.zeros_like(w) for w in self.W]; self.vW=[np.zeros_like(w) for w in self.W]
        self.mb=[np.zeros_like(b) for b in self.b]; self.vb=[np.zeros_like(b) for b in self.b]
        self.t=0
    def forward(self, X):
        a=X; self.cache=[(None,X)]
        for i in range(len(self.W)-1):
            z=a@self.W[i]+self.b[i]; a=ACT[self.acts[i]][0](z); self.cache.append((z,a))
        zL=a@self.W[-1]+self.b[-1]; p=sigmoid(zL); self.cache.append((zL,p))
        return p.ravel()
    def step(self, X, y, lr=0.05):
        n=len(y); p=self.forward(X); p=p.reshape(-1,1); Y=y.reshape(-1,1)
        dz=(p-Y)/n  # BCE + sigmoid
        gW=[None]*len(self.W); gb=[None]*len(self.b)
        a_prev=self.cache[-2][1]
        gW[-1]=a_prev.T@dz; gb[-1]=dz.sum(0)
        da=dz@self.W[-1].T
        for i in range(len(self.W)-2, -1, -1):
            z=self.cache[i+1][0]; dz=da*ACT[self.acts[i]][1](z)
            a_prev=self.cache[i][1]
            gW[i]=a_prev.T@dz; gb[i]=dz.sum(0); da=dz@self.W[i].T
        # adam
        self.t+=1; b1,b2,eps=0.9,0.999,1e-8
        for i in range(len(self.W)):
            self.mW[i]=b1*self.mW[i]+(1-b1)*gW[i]; self.vW[i]=b2*self.vW[i]+(1-b2)*gW[i]**2
            mhat=self.mW[i]/(1-b1**self.t); vhat=self.vW[i]/(1-b2**self.t)
            self.W[i]-=lr*mhat/(np.sqrt(vhat)+eps)
            self.mb[i]=b1*self.mb[i]+(1-b1)*gb[i]; self.vb[i]=b2*self.vb[i]+(1-b2)*gb[i]**2
            mhb=self.mb[i]/(1-b1**self.t); vhb=self.vb[i]/(1-b2**self.t)
            self.b[i]-=lr*mhb/(np.sqrt(vhb)+eps)
    def prob(self, X): return self.forward(X)
    def acc(self, X, y): return float(((self.prob(X)>0.5).astype(float)==y).mean())

# ---------------- helpers ----------------
def frame_sched(total, k=11):
    # denser early; unique sorted epoch indices in [0,total]
    xs=np.unique(np.round(np.linspace(0,1,k)**1.7*total).astype(int))
    return list(xs)

def grid_probs(model, dom, G=40):
    a,b=dom; xs=np.linspace(a,b,G)
    XX,YY=np.meshgrid(xs,xs)
    pts=np.c_[XX.ravel(),YY.ravel()]
    p=model.prob(pts)             # row-major: j*G+i with y=rows
    q=np.clip(np.round(p*255),0,255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode()

def train_capture(model, X, y, dom, epochs, lr=0.05, G=40):
    sched=set(frame_sched(epochs))
    frames=[]; accs=[]
    for e in range(epochs+1):
        if e in sched:
            frames.append(grid_probs(model,dom,G)); accs.append(round(model.acc(X,y),4))
        if e<epochs: model.step(X,y,lr)
    return frames, accs

def pts_payload(X,y,maxn=None,seed=0):
    if maxn and len(X)>maxn:
        idx=np.random.RandomState(seed).choice(len(X),maxn,replace=False); X=X[idx]; y=y[idx]
    return {"x":[round(float(v),3) for v in X[:,0]],
            "y":[round(float(v),3) for v in X[:,1]],
            "c":[int(v) for v in y]}

OUT={"meta":{"G":40}}
DOM_R=[-1.05,1.05]; DOM_N=[-3.6,3.6]

# ===== S1-1 : activations =====
Xr,yr=make_rings(300,0.055,7)
s11={"domain":DOM_R,"points":pts_payload(Xr,yr),"models":{}}
m=MLP([2,1],['linear'],seed=1); f,a=train_capture(m,Xr,yr,DOM_R,70,0.06); s11["models"]["linear"]={"frames":f,"acc":a}
m=MLP([2,16,1],['relu','linear'],seed=2); f,a=train_capture(m,Xr,yr,DOM_R,70,0.05); s11["models"]["relu"]={"frames":f,"acc":a}
m=MLP([2,16,1],['tanh','linear'],seed=3); f,a=train_capture(m,Xr,yr,DOM_R,70,0.05); s11["models"]["tanh"]={"frames":f,"acc":a}
OUT["s11"]=s11

# ===== S1-2 : depth without nonlinearity =====
s12={"domain":DOM_R,"points":pts_payload(Xr,yr),"models":{}}
m=MLP([2,1],['linear'],seed=4); f,a=train_capture(m,Xr,yr,DOM_R,80,0.06); s12["models"]["lin1"]={"frames":f,"acc":a}
m5=MLP([2,8,8,8,8,1],['linear']*5,seed=5); f,a=train_capture(m5,Xr,yr,DOM_R,80,0.03); s12["models"]["lin5"]={"frames":f,"acc":a}
m=MLP([2,16,16,16,16,1],['relu']*4+['linear'],seed=6); f,a=train_capture(m,Xr,yr,DOM_R,80,0.04); s12["models"]["relu5"]={"frames":f,"acc":a}
# weight collapse from trained 5-linear net (exclude final sigmoid head's pre-activation is linear too)
prod=m5.W[0]
for W in m5.W[1:]: prod=prod@W
s12["collapse"]={"shapes":[list(W.shape) for W in m5.W],
                 "weff":[round(float(prod[0,0]),4), round(float(prod[1,0]),4)]}
OUT["s12"]=s12

# ===== S1-3 : embeddings from next-token =====
VOCAB=['cat','dog','cow','apple','mango','banana','eat','chase','see']
CATG=[0,0,0,1,1,1,2,2,2]
A=[0,1,2]; F=[3,4,5]; V=[6,7,8]
r=np.random.RandomState(3); xs=[]; ys=[]
for _ in range(2000):
    a=A[r.randint(3)]; v=V[r.randint(3)]; f=F[r.randint(3)]
    xs+=[a,v]; ys+=[v,f]
xs=np.array(xs); ys=np.array(ys)
D=2; Vn=9
E=np.random.RandomState(5).randn(Vn,D)*0.30
Wo=np.random.RandomState(6).randn(D,Vn)*0.10; bo=np.zeros(Vn)
def softmax(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
EP3=140; lr3=0.40
sched3=set(frame_sched(EP3,16))
emb_frames=[]; loss_frames=[]; pur_frames=[]
def purity(E):
    ok=0
    for i in range(Vn):
        d=((E-E[i])**2).sum(1); d[i]=1e9; j=int(d.argmin()); ok+=CATG[j]==CATG[i]
    return ok
for e in range(EP3+1):
    h=E[xs]; logits=h@Wo+bo; p=softmax(logits)
    if e in sched3:
        ll=-np.log(p[np.arange(len(ys)),ys]+1e-12).mean()
        emb_frames.append([[round(float(E[i,0]),3),round(float(E[i,1]),3)] for i in range(Vn)])
        loss_frames.append(round(float(ll),3)); pur_frames.append(int(purity(E)))
    if e<EP3:
        pp=p.copy(); pp[np.arange(len(ys)),ys]-=1; g=pp/len(ys)
        gWo=h.T@g; gbo=g.sum(0); gh=g@Wo.T
        Wo-=lr3*gWo; bo-=lr3*gbo
        np.add.at(E, xs, -lr3*gh)
OUT["s13"]={"vocab":VOCAB,"cat":CATG,"emb":emb_frames,"loss":loss_frames,"purity":pur_frames}

# ===== S1-4 : memorization vs generalization =====
Xte,yte=make_noisy(2000,999)
s14={"domain":DOM_N,"test_points":pts_payload(Xte,yte,maxn=500,seed=1),"sizes":{}}
for n,seed,ep,lr in [(20,101,400,0.03),(200,102,200,0.02),(2000,103,90,0.02)]:
    Xtr,ytr=make_noisy(n,seed)
    m=MLP([2,64,64,1],['relu','relu','linear'],seed=10+n)
    sched=set(frame_sched(ep,11)); frames=[]; tr=[]; te=[]
    for e in range(ep+1):
        if e in sched:
            frames.append(grid_probs(m,DOM_N,40))
            tr.append(round(m.acc(Xtr,ytr),4)); te.append(round(m.acc(Xte,yte),4))
        if e<ep: m.step(Xtr,ytr,lr)
    s14["sizes"][str(n)]={"frames":frames,"train":tr,"test":te,
                          "points":pts_payload(Xtr,ytr,maxn=400,seed=2)}
OUT["s14"]=s14

js="window.PRECOMP="+json.dumps(OUT,separators=(',',':'))+";"
open("data.js","w").write(js)
# quick report
print("bytes data.js:", len(js))
print("S1-1 linear acc end", OUT['s11']['models']['linear']['acc'][-1],
      "| relu", OUT['s11']['models']['relu']['acc'][-1],
      "| tanh", OUT['s11']['models']['tanh']['acc'][-1])
print("S1-2 lin1", OUT['s12']['models']['lin1']['acc'][-1],
      "| lin5", OUT['s12']['models']['lin5']['acc'][-1],
      "| relu5", OUT['s12']['models']['relu5']['acc'][-1],
      "| weff", OUT['s12']['collapse']['weff'])
print("S1-3 loss end", OUT['s13']['loss'][-1], "| purity end", OUT['s13']['purity'][-1], "/9")
for n in ['20','200','2000']:
    d=OUT['s14']['sizes'][n]; print(f"S1-4 n={n} train {d['train'][-1]} test {d['test'][-1]} gap {round(d['train'][-1]-d['test'][-1],3)}")
print("frames per ring model:", len(OUT['s11']['models']['linear']['frames']))
