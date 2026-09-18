"""Observable-only temporal tasks with chronological train/validation/test splits."""
import numpy as np
from .spin import chemical_filter


def features(y, include_chemistry=True):
    if not include_chemistry:
        return y[:,None]
    return np.column_stack([y]+[chemical_filter(y,tau,1.) for tau in [1.,5.,20.]])


def targets(u,max_delay=30):
    labels=[f'delay_{d}' for d in range(1,max_delay+1)]+['product_1_2','square_5']
    target=np.zeros((len(u),len(labels)))
    for d in range(1,max_delay+1):
        target[d:,d-1]=u[:-d]
    target[2:,-2]=u[1:-1]*u[:-2]
    target[5:,-1]=u[:-5]**2
    return target,labels


def r2_score(y,pred):
    variance=np.mean((y-y.mean(axis=0))**2,axis=0)
    return 1-np.mean((y-pred)**2,axis=0)/np.maximum(variance,1e-20)


def fit_scores(x,y,washout=300,train=2400,valid=800):
    """Each target selects ridge on validation only. Test never tunes anything."""
    x=x[washout:];y=y[washout:]
    tr=slice(0,train);va=slice(train,train+valid);te=slice(train+valid,None)
    xm=x[tr].mean(0);xs=x[tr].std(0);xs=np.maximum(xs,1e-12)
    a=(x-xm)/xs
    ym=y[tr].mean(0)
    gram=a[tr].T@a[tr]/train;rhs=a[tr].T@(y[tr]-ym)/train
    weights=[];val=[]
    alphas=np.array([1e-8,1e-6,1e-4,1e-2,1e-1])
    for alpha in alphas:
        w=np.linalg.solve(gram+alpha*np.eye(x.shape[1]),rhs)
        weights.append(w);val.append(r2_score(y[va],a[va]@w+ym))
    best=np.argmax(np.array(val),axis=0)
    w=np.column_stack([weights[k][:,j] for j,k in enumerate(best)])
    pred=a[te]@w+ym
    return r2_score(y[te],pred),alphas[best],pred,y[te]


def esn(u,seed,dimension=4):
    """Small algorithmic reference, not a physiologically calibrated circuit."""
    rng=np.random.default_rng(seed)
    w=rng.normal(size=(dimension,dimension))
    w*=.8/max(abs(np.linalg.eigvals(w)))
    win=rng.normal(0,.7,size=dimension);bias=rng.normal(0,.1,size=dimension)
    x=np.zeros(dimension);states=[]
    for value in u:
        x=.6*x+.4*np.tanh(w@x+win*value+bias)
        states.append(x.copy())
    return np.array(states)


def observe_yield(y,events,rng):
    if events is None:
        return y.copy()
    if np.min(y)<-1e-10 or np.max(y)>1+1e-10:
        raise ArithmeticError('reaction yield outside [0,1]')
    return rng.binomial(events,np.clip(y,0,1))/events
