"""Original Elliptic transaction dataset: chronological supervised benchmark.

No model artifact uploaded by a user is deserialized. Artifacts are generated only
by the local training command. Anonymized features cannot be reconstructed from a
wallet address through this interface.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import joblib
import sklearn
import sys
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, auc, confusion_matrix, precision_recall_curve, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.getenv('ELLIPTIC_ARTIFACT_DIR', str(ROOT/'artifacts'/'elliptic')))
router = APIRouter(prefix='/elliptic', tags=['Elliptic supervised evaluation'])
MAX_UPLOAD = 32*1024*1024


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_features(source):
    frame=pd.read_csv(source,header=None,dtype=str)
    if not len(frame):raise ValueError('Empty feature file')
    if str(frame.iloc[0,0]).lower() in ('txid','tx_id'):
        frame=frame.iloc[1:].reset_index(drop=True)
    if frame.shape[1]!=167:
        raise ValueError(f'Original Elliptic requires 167 columns: txId, time_step, and 165 numeric features. Received {frame.shape[1]}.')
    ids=frame.iloc[:,0].astype(str)
    if ids.duplicated().any():raise ValueError('Duplicate txId values in feature file')
    if ids.str.len().eq(0).any() or ids.str.len().gt(128).any():raise ValueError('Invalid transaction identifier')
    numeric=frame.iloc[:,1:].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():raise ValueError('Features must be finite numeric values; missing values are not accepted')
    times=numeric[:,0]
    if ((times<1)|(times>49)|(times!=np.floor(times))).any():raise ValueError('Original Elliptic time_step must be an integer between 1 and 49')
    return ids.to_numpy(),times.astype(int),numeric[:,1:].astype(np.float32)


def read_classes(source):
    frame=pd.read_csv(source,dtype=str)
    if set(frame.columns)!={'txId','class'}:raise ValueError('Classes CSV must contain exactly txId,class; 1=illicit, 2=licit, unknown=unlabeled')
    if frame.txId.duplicated().any():raise ValueError('Duplicate transaction IDs in classes file')
    if not frame['class'].isin(['1','2','unknown']).all():raise ValueError('Classes must be 1, 2 or unknown. 0 is not the original Elliptic licit class.')
    return dict(zip(frame.txId,frame['class']))


def metrics(labels, scores, threshold):
    labels=np.asarray(labels,dtype=int);scores=np.asarray(scores,dtype=float)
    if len(labels)!=len(scores):raise ValueError('Labels and predictions must be aligned')
    known=labels>=0;y=labels[known];s=scores[known]
    result={'total':int(len(labels)),'labeled':int(known.sum()),'unknown':int((~known).sum()),
            'positives':int((y==1).sum()),'negatives':int((y==0).sum()),'threshold':float(threshold),
            'precision':None,'recall':None,'f1':None,'average_precision':None,'pr_auc_trapezoidal':None,
            'confusion_matrix':None,'curve':[],'warnings':[]}
    if not len(y):result['warnings']=['No known ground-truth labels; metrics are unavailable.'];return result
    pred=(s>=threshold).astype(int)
    p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    result.update(precision=float(p),recall=float(r),f1=float(f),
                  confusion_matrix={'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},prevalence=float(y.mean()))
    if len(np.unique(y))==2:
        precision,recall,_=precision_recall_curve(y,s)
        result.update(average_precision=float(average_precision_score(y,s)),pr_auc_trapezoidal=float(auc(recall,precision)))
        # Display-only subsampling; full arrays are used in both area calculations.
        indices=np.unique(np.linspace(0,len(precision)-1,min(300,len(precision)),dtype=int))
        result['curve']=[{'precision':float(precision[i]),'recall':float(recall[i])} for i in indices]
    else:result['warnings'].append('Only one labeled class: PR-AUC/AP withheld. F1 is not a representative two-class evaluation.')
    if not pred.any():result['warnings'].append('No predicted positives; undefined precision is reported as 0.')
    if not y.any():result['warnings'].append('No positive labels; undefined recall is reported as 0.')
    if (~known).any():result['warnings'].append('Unknown labels are excluded from all evaluation metrics.')
    return result


def choose_threshold(y,scores):
    p,r,t=precision_recall_curve(y,scores)
    f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    # Break ties conservatively: choose the highest threshold among equal F1s.
    return float(t[np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]])


def train(folder:Path, output:Path=ARTIFACTS):
    started=perf_counter()
    paths={name:folder/f'elliptic_txs_{suffix}.csv' for name,suffix in [('features','features'),('classes','classes'),('edges','edgelist')]}
    for p in paths.values():
        if not p.is_file():raise ValueError(f'Missing dataset file: {p}')
    ids,times,x=read_features(paths['features']);classes=read_classes(paths['classes'])
    if set(classes)-set(ids):raise ValueError('Classes contain IDs not present in the features')
    labels=np.array([1 if classes.get(i)=='1' else 0 if classes.get(i)=='2' else -1 for i in ids])
    train_mask=(times<=29)&(labels>=0);val_mask=(times>=30)&(times<=34)&(labels>=0);test_mask=times>=35
    for name,mask in [('training',train_mask),('validation',val_mask),('test',test_mask&(labels>=0))]:
        if len(np.unique(labels[mask]))!=2:raise ValueError(f'{name} split must contain both known classes')
    print(f'Loaded {len(ids):,} transactions; training={train_mask.sum():,}, validation={val_mask.sum():,}, test known={((labels>=0)&test_mask).sum():,}',flush=True)
    candidates={
        'Logistic regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42)),
        'Histogram gradient boosting':HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=31,learning_rate=.08,l2_regularization=1,class_weight='balanced',early_stopping=False,random_state=42),
    }
    comparisons=[];best=None;best_ap=-1
    for name,model in candidates.items():
        t0=perf_counter();print(f'Training {name}…',flush=True);model.fit(x[train_mask],labels[train_mask])
        scores=model.predict_proba(x[val_mask])[:,1];ap=float(average_precision_score(labels[val_mask],scores))
        threshold=choose_threshold(labels[val_mask],scores)
        comparisons.append({'model':name,'validation_ap':ap,'validation_f1':metrics(labels[val_mask],scores,threshold)['f1'],'threshold':threshold,'fit_seconds':round(perf_counter()-t0,2)})
        if ap>best_ap:best=(name,model,threshold);best_ap=ap
    name,model,threshold=best
    print(f'Selected {name} by validation AP. Evaluating the untouched test split…',flush=True)
    scores=model.predict_proba(x)[:,1]
    edges=pd.read_csv(paths['edges'],dtype=str)
    if set(edges.columns)!={'txId1','txId2'}:raise ValueError('Edge file needs txId1,txId2 columns')
    edge_pairs=edges[['txId1','txId2']].to_numpy()
    if not set(edge_pairs.ravel()).issubset(set(ids)):raise ValueError('Edges reference IDs absent from features')
    manifest={'project':'Chain Forensics','dataset':'Original Elliptic Bitcoin transaction dataset','model':name,
              'trained_at':datetime.now(UTC).isoformat(),'threshold':threshold,'feature_count':165,
              'excluded_columns':['txId','time_step','class'],'positive_class':'illicit (original class 1)',
              'class_mapping':{'1':1,'2':0,'unknown':'excluded from fitting and evaluation'},
              'splits':{'train':'time steps 1–29, known labels only','validation':'30–34, known labels only','test':'35–49; metrics on known labels only'},
              'selection_policy':'Model selected by validation Average Precision; threshold maximizes validation F1. No test-set tuning.',
              'candidates':comparisons,'validation':metrics(labels[val_mask],scores[val_mask],threshold),
              'test':metrics(labels[test_mask],scores[test_mask],threshold),
              'counts':{'transactions':len(ids),'edges':len(edges),'training':int(train_mask.sum()),'validation':int(val_mask.sum())},
              'file_sha256':{key:digest(path) for key,path in paths.items()},
              'limitations':['Anonymized dataset features; cannot score raw wallet addresses.', 'Published aggregate features may include neighborhood information unavailable at an online decision time; this is an offline temporal benchmark.', 'No claim of real-time blockchain detection or verified ownership.'],
              'training_seconds':round(perf_counter()-started,2),
              'software':{'python':sys.version.split()[0],'scikit_learn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},
              'random_state':42, 'label_provenance':'Published Elliptic class labels; not independently re-verified',
              'source':'https://www.kaggle.com/datasets/ellipticco/elliptic-data-set'}
    output.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':model,'threshold':threshold,'manifest':manifest},output/'model.joblib')
    np.savez_compressed(output/'predictions.npz',ids=ids.astype(str),times=times,labels=labels,scores=scores)
    edges.to_csv(output/'edges.csv',index=False)
    (output/'benchmark.json').write_text(json.dumps(manifest,indent=2))
    # A small unlabeled held-out feature file plus separate labels for live evaluation.
    indices=np.flatnonzero(test_mask&(labels>=0))[:200]
    pd.DataFrame(np.column_stack([ids[indices],times[indices],x[indices]])).to_csv(output/'judge-features.csv',header=False,index=False)
    pd.DataFrame({'txId':ids[indices],'class':[classes[i] for i in ids[indices]]}).to_csv(output/'judge-labels.csv',index=False)
    print(json.dumps({'model':name,'threshold':threshold,'test':{k:v for k,v in manifest['test'].items() if k!='curve'},'seconds':manifest['training_seconds']},indent=2),flush=True)
    return manifest


@lru_cache(maxsize=1)
def artifact():
    if not (ARTIFACTS/'model.joblib').is_file():raise HTTPException(409,'Elliptic model not trained. Provide the original dataset files and run the training command.')
    return joblib.load(ARTIFACTS/'model.joblib')


@lru_cache(maxsize=1)
def predictions():
    if not (ARTIFACTS/'predictions.npz').is_file():raise HTTPException(409,'Elliptic model not trained')
    with np.load(ARTIFACTS/'predictions.npz',allow_pickle=False) as data:return {k:data[k] for k in data.files}


@router.get('/status')
def status():
    if not (ARTIFACTS/'benchmark.json').is_file():return {'ready':False,'required_files':['elliptic_txs_features.csv','elliptic_txs_classes.csv','elliptic_txs_edgelist.csv']}
    return {'ready':True,**artifact()['manifest']}


@router.get('/transactions')
def transactions(query:str='',flagged_only:bool=False,offset:int=Query(0,ge=0),limit:int=Query(20,ge=1,le=100)):
    data=predictions();threshold=artifact()['threshold'];mask=data['times']>=35
    if query:mask &= np.char.find(data['ids'],query)>=0
    if flagged_only:mask &= data['scores']>=threshold
    indices=np.flatnonzero(mask);indices=indices[np.argsort(-data['scores'][indices],kind='stable')]
    rows=[{'txid':str(data['ids'][i]),'time_step':int(data['times'][i]),'score':float(data['scores'][i]),'label':int(data['labels'][i]),'flagged':bool(data['scores'][i]>=threshold)} for i in indices[offset:offset+limit]]
    return {'total':len(indices),'rows':rows,'threshold':threshold,'scope':'held-out time steps 35–49; unknown labels retained for exploration'}


@router.post('/predict')
async def predict(features:UploadFile=File(...),labels:UploadFile|None=File(None)):
    started=perf_counter();bundle=artifact()
    raw=await features.read(MAX_UPLOAD+1)
    if len(raw)>MAX_UPLOAD:raise HTTPException(413,'Feature file limit is 32 MiB')
    try:
        ids,times,x=read_features(io.BytesIO(raw))
        if len(ids)>10000:raise ValueError('Live evaluation accepts at most 10,000 transactions')
        supplied={};label_sha=None
        if labels is not None:
            label_raw=await labels.read(MAX_UPLOAD+1)
            if len(label_raw)>MAX_UPLOAD:raise ValueError('Labels file limit is 32 MiB')
            supplied=read_classes(io.BytesIO(label_raw));label_sha=hashlib.sha256(label_raw).hexdigest()
            if set(supplied)-set(ids):raise ValueError('Label file contains transaction IDs not present in the uploaded feature file')
        y=np.array([1 if supplied.get(i)=='1' else 0 if supplied.get(i)=='2' else -1 for i in ids])
        scores=bundle['model'].predict_proba(x)[:,1]
    except (ValueError,TypeError,pd.errors.ParserError) as error:raise HTTPException(422,str(error)) from error
    known=predictions();fit_ids=set(known['ids'][(known['times']<=34)&(known['labels']>=0)])
    overlap=sum(i in fit_ids for i in ids)
    result=metrics(y,scores,bundle['threshold'])
    if overlap:result['warnings'].append(f'{overlap} uploaded records overlap training/validation IDs. These metrics are not an independent test.')
    return {'model':bundle['manifest']['model'],'evaluation':result,'predictions':[{'txid':str(i),'score':float(p),'flagged':bool(p>=bundle['threshold']),'label':int(label)} for i,p,label in zip(ids,scores,y)],
            'feature_sha256':hashlib.sha256(raw).hexdigest(),'label_sha256':label_sha,'training_validation_overlap':overlap,
            'elapsed_ms':round((perf_counter()-started)*1000,2),'threshold_frozen':True,
            'disclaimer':'Elliptic-compatible anonymized features only; scores are not calibrated criminality probabilities.'}


@router.get('/graph/{txid}')
def graph(txid:str):
    data=predictions();matches=np.flatnonzero(data['ids']==txid)
    if not len(matches):raise HTTPException(404,'Transaction not present in Elliptic dataset')
    edges=pd.read_csv(ARTIFACTS/'edges.csv',dtype=str)
    connected=edges[(edges.txId1==txid)|(edges.txId2==txid)]
    shown=connected.head(30);ids=set(shown.to_numpy().ravel())|{txid}
    lookup={str(i):idx for idx,i in enumerate(data['ids']) if i in ids}
    return {'nodes':[{'id':i,'score':float(data['scores'][lookup[i]]),'label':int(data['labels'][lookup[i]])} for i in sorted(ids)],
            'edges':[{'source':r.txId1,'target':r.txId2} for r in shown.itertuples()],
            'truncated':len(connected)>30,'total_edges':len(connected),'description':'Published directed payment-flow edges, one-hop neighborhood; no ownership inference.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,default=ARTIFACTS)
    args=parser.parse_args();train(args.data_dir,args.output_dir)
