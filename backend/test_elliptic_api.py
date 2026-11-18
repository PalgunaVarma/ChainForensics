import io
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend import elliptic

class FrozenModel:
    def predict_proba(self,x):
        score=np.clip(x[:,0],0,1)
        return np.column_stack([1-score,score])

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(elliptic,'artifact',lambda:{'model':FrozenModel(),'threshold':.5,'manifest':{'model':'Test fixture only'}})
    monkeypatch.setattr(elliptic,'predictions',lambda:{'ids':np.array(['training-id','validation-id']),'times':np.array([1,32]),'labels':np.array([1,0])})
    app=FastAPI();app.include_router(elliptic.router)
    with TestClient(app) as c:yield c

def feature_rows(ids=('101','102','103','104')):
    return '\n'.join(f'{key},35,{score},'+','.join(['0.0']*164) for key,score in zip(ids,[.9,.8,.2,.1]))

def test_live_upload_metrics_and_label_id_alignment(client):
    r=client.post('/elliptic/predict',files={'features':('f.csv',feature_rows()),'labels':('l.csv','txId,class\n104,2\n103,1\n102,2\n101,1\n')})
    assert r.status_code==200,r.text
    result=r.json();assert result['evaluation']['f1']==.5
    assert result['evaluation']['average_precision']==pytest.approx(5/6)
    assert result['threshold_frozen'] is True and result['training_validation_overlap']==0
    assert result['feature_sha256'] and result['label_sha256']

def test_unlabeled_upload_never_fabricates_metrics(client):
    r=client.post('/elliptic/predict',files={'features':('f.csv',feature_rows())})
    assert r.status_code==200
    assert r.json()['evaluation']['f1'] is None
    assert r.json()['evaluation']['unknown']==4

def test_upload_rejects_incompatible_features_and_unknown_label_ids(client):
    assert client.post('/elliptic/predict',files={'features':('f.csv','1,2,3')}).status_code==422
    r=client.post('/elliptic/predict',files={'features':('f.csv',feature_rows()),'labels':('l.csv','txId,class\nnot-in-features,1\n')})
    assert r.status_code==422

def test_training_overlap_is_reported(client):
    r=client.post('/elliptic/predict',files={'features':('f.csv',feature_rows(('training-id','102','103','104')))})
    assert r.status_code==200
    assert r.json()['training_validation_overlap']==1
    assert any('not an independent test' in w for w in r.json()['evaluation']['warnings'])

def test_single_class_labels_withhold_curve_metrics(client):
    r=client.post('/elliptic/predict',files={'features':('f.csv',feature_rows()),'labels':('l.csv','txId,class\n101,1\n102,1\n103,1\n104,1\n')})
    assert r.status_code==200
    assert r.json()['evaluation']['average_precision'] is None
