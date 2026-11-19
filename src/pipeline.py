"""Reproducible high-dimensional geometry experiments on 20 Newsgroups.
All reported results are calculated, never hard-coded. Run from project root.
"""
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'

def savefig(name):
    plt.tight_layout();plt.savefig(OUT/f'{name}.png',dpi=170);plt.close()

def lb_id(x,ks=(5,10,20)):
    distances=NearestNeighbors(n_neighbors=max(ks)+1,metric='euclidean').fit(x).kneighbors(x)[0][:,1:]
    results={}
    for k in ks:
        t=distances[:,:k];r=np.maximum(t[:,-1,None],1e-12)/np.maximum(t[:,:-1],1e-12)
        denom=np.log(r).sum(axis=1)
        estimates=(k-2)/np.where(denom>1e-12,denom,np.nan)
        results[str(k)]=float(np.nanmean(estimates))
    return results

def corr_dim(x):
    ds=pdist(x);ds=ds[ds>0]
    radii=np.geomspace(np.percentile(ds,1),np.percentile(ds,90),70)
    counts=np.searchsorted(np.sort(ds),radii,side='right')/len(ds)
    valid=(counts>0)&(counts<0.95)
    lr=np.log(radii[valid]);lc=np.log(counts[valid]);
    if len(lr)<12:return {'estimate':None,'span_decades':None}
    # Diagnostic only: narrow automatic windows do not establish a scaling regime.
    width=10; candidates=[]
    for i in range(len(lr)-width+1):
        sl=np.diff(lc[i:i+width])/np.diff(lr[i:i+width]);candidates.append(np.var(sl))
    start=int(np.argmin(candidates));end=start+width
    slope=float(np.polyfit(lr[start:end],lc[start:end],1)[0]);span=float((lr[end-1]-lr[start])/np.log(10))
    plt.plot(lr,lc,'.',markersize=3);plt.plot(lr[start:end],lc[start:end],linewidth=3,label=f'slope={slope:.2f}; span={span:.3f} decades');plt.legend();plt.xlabel('log radius');plt.ylabel('log correlation integral');savefig('correlation_integral')
    return {'estimate':slope,'span_decades':span,'warning':'Unreliable if scaling interval is narrow'}

def pca_stats(x,prefix="real"):
    model=PCA(svd_solver='full').fit(x);ratio=model.explained_variance_ratio_;c=np.cumsum(ratio)
    plt.plot(np.arange(1,len(ratio)+1),ratio);plt.xlabel('PC');plt.ylabel('Explained variance ratio');savefig(f'{prefix}_scree')
    plt.plot(np.arange(1,len(c)+1),c);plt.axhline(.9,linestyle='--');plt.xlabel('PCs');plt.ylabel('Cumulative explained variance');savefig(f'{prefix}_cumulative_variance')
    return {'pc90':int(np.searchsorted(c,.90)+1),'pc95':int(np.searchsorted(c,.95)+1),'pc99':int(np.searchsorted(c,.99)+1),'kaiser':int((model.explained_variance_>model.explained_variance_.mean()).sum())}

def embedding_plot(x,y,method,name):
    if method=='pca':z=PCA(n_components=2,random_state=42).fit_transform(x)
    elif method=='tsne':z=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=42).fit_transform(x)
    else:
        import umap
        z=umap.UMAP(n_neighbors=15,min_dist=.1,random_state=42).fit_transform(x)
    plt.figure(figsize=(8,6));plt.scatter(z[:,0],z[:,1],c=y,s=8,cmap='tab20',alpha=.75);plt.xlabel('Axis 1');plt.ylabel('Axis 2');savefig(name)
    return float(silhouette_score(z,y))

def jl_sweep(x,y,rng,dimensions=(20,40,80,120,200,300),reps=3):
    base=pdist(x);nbr=NearestNeighbors(n_neighbors=11).fit(x).kneighbors(x,return_distance=False)[:,1:]
    rows=[]
    for k in dimensions:
        for rep in range(reps):
            z=GaussianRandomProjection(n_components=k,random_state=42+rep).fit_transform(x)
            ratio=pdist(z)/np.maximum(base,1e-12)
            new=NearestNeighbors(n_neighbors=11).fit(z).kneighbors(z,return_distance=False)[:,1:]
            overlap=np.mean([len(set(a)&set(b))/10 for a,b in zip(nbr,new)])
            rows.append({'k':k,'seed':rep,'mean_ratio':ratio.mean(),'max_abs_error':np.max(abs(ratio-1)),'nn_overlap':overlap,'silhouette':silhouette_score(z,y,sample_size=min(len(y),600),random_state=42)})
    df=pd.DataFrame(rows);df.to_csv(OUT/'jl_sweep.csv',index=False)
    agg=df.groupby('k').mean(numeric_only=True)
    plt.plot(agg.index,agg.max_abs_error,marker='o',label='Max distortion');plt.plot(agg.index,agg.nn_overlap,marker='s',label='10-NN overlap');plt.xscale('log');plt.legend();plt.xlabel('Projected dimensions');savefig('jl_sweep')
    return df.groupby('k')[['max_abs_error','nn_overlap','silhouette']].agg(['mean','std']).round(4).to_dict()

def concentration(x,rng):
    rows=[]
    for d in [10,25,50,100,200,x.shape[1]]:
        if d>x.shape[1]:continue
        for rep in range(5):
            ix=rng.choice(x.shape[1],size=d,replace=False)
            distances=pdist(x[:,ix]);rows.append({'d':d,'rep':rep,'spread':float((distances.max()-distances.min())/max(distances.min(),1e-12)),'cv':float(distances.std()/distances.mean())})
    df=pd.DataFrame(rows);df.to_csv(OUT/'distance_concentration.csv',index=False)
    a=df.groupby('d').cv.agg(['mean','std']);plt.errorbar(a.index,a['mean'],yerr=a['std'],marker='o');plt.xscale('log');plt.xlabel('Dimensions');plt.ylabel('Distance coefficient of variation');savefig('distance_concentration')
    return a.round(5).to_dict()

def spectrum(x,rng):
    # Empirical null: independently permute each column, preserving marginals.
    def eigen(a):return np.linalg.eigvalsh(np.cov(a,rowvar=False))
    ev=eigen(x);sh=x.copy()
    for j in range(sh.shape[1]):rng.shuffle(sh[:,j])
    null=eigen(sh)
    gamma=x.shape[1]/(x.shape[0]-1)
    edge=(1+np.sqrt(gamma))**2
    plt.hist(ev,bins=70,alpha=.5,label='Observed');plt.hist(null,bins=70,alpha=.5,label='Column-shuffle null');plt.axvline(edge,color='black',linestyle='--',label='MP edge (unit-variance iid)');plt.yscale('log');plt.legend();plt.xlabel('Covariance eigenvalue');savefig('spectrum')
    return {'gamma':gamma,'mp_edge_unit_variance':edge,'observed_above_edge':int((ev>edge).sum()),'null_above_edge':int((null>edge).sum()),'caveat':'MP edge assumes independent unit-variance entries; compare shuffled null and verify scaling'}

def anomalies(x):
    model=LedoitWolf().fit(x);d=model.mahalanobis(x)
    plt.hist(d,bins=40);plt.xlabel('Squared Mahalanobis distance');savefig('mahalanobis')
    # Empirical rank, not a falsely calibrated chi-square significance test.
    order=np.argsort(d)[-10:][::-1];pd.DataFrame({'sample_index':order,'distance_squared':d[order]}).to_csv(OUT/'top_mahalanobis.csv',index=False)
    return {'shrinkage':float(model.shrinkage_),'min':float(d.min()),'max':float(d.max()),'mean':float(d.mean()),'note':'Rankings only; no chi-square outlier claim'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=800);ap.add_argument('--features',type=int,default=5000);ap.add_argument('--svd',type=int,default=300);ap.add_argument('--seed',type=int,default=42);args=ap.parse_args()
    OUT.mkdir(exist_ok=True);rng=np.random.default_rng(args.seed)
    corpus=fetch_20newsgroups(subset='all',remove=('headers','footers','quotes'),shuffle=False)
    selected=np.sort(rng.choice(len(corpus.data),size=min(args.samples,len(corpus.data)),replace=False))
    docs=[corpus.data[i] for i in selected];y=np.asarray(corpus.target)[selected]
    vec=TfidfVectorizer(max_features=args.features,min_df=2,stop_words='english',norm='l2',sublinear_tf=True)
    sparse=vec.fit_transform(docs);nonempty=np.asarray(sparse.getnnz(axis=1)>0).ravel();sparse=sparse[nonempty];y=y[nonempty]
    # TruncatedSVD is not PCA: it does not center the sparse TF-IDF matrix.
    ncomp=min(args.svd,sparse.shape[0]-2,sparse.shape[1]-1)
    svd=TruncatedSVD(n_components=ncomp,random_state=args.seed);x=svd.fit_transform(sparse)
    x=StandardScaler().fit_transform(x)
    pd.DataFrame({'category':np.bincount(y,minlength=20)}).to_csv(OUT/'class_balance.csv',index_label='class_id')
    report={'dataset':{'selected':len(selected),'nonempty':len(y),'tfidf_shape':list(sparse.shape),'svd_shape':list(x.shape),'nnz':int(sparse.nnz),'tfidf_density':float(sparse.nnz/(sparse.shape[0]*sparse.shape[1]))},'settings':vars(args)}
    report['pca']=pca_stats(x);report['lb_id']=lb_id(x);report['correlation_dimension']=corr_dim(x)
    report['embedding_silhouette']={m:embedding_plot(x,y,m,f'embedding_{m}') for m in ['pca','tsne','umap']}
    report['jl']=jl_sweep(x,y,rng);report['concentration']=concentration(x,rng);report['spectrum']=spectrum(x,rng);report['anomalies']=anomalies(x)
    # Matched baselines use the same sample count and analysis dimension.
    noise=rng.standard_normal(x.shape)
    z=rng.standard_normal((len(x),5));q,_=np.linalg.qr(rng.standard_normal((x.shape[1],5)));manifold=z@q.T+0.001*rng.standard_normal(x.shape)
    report['baselines']={'gaussian':{'pca':pca_stats(noise,'gaussian'),'lb':lb_id(noise)},'five_dimensional_linear':{'pca':pca_stats(manifold,'linear_manifold'),'lb':lb_id(manifold)}}
    with open(OUT/'summary.json','w') as f:json.dump(report,f,indent=2,default=lambda v:float(v) if isinstance(v,np.floating) else int(v))
    print('Completed. Results saved to',OUT)
if __name__=='__main__':main()
