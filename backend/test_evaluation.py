import io
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from backend.elliptic import metrics, choose_threshold, read_features, read_classes
from backend.main import evaluate_dataset, Record


def test_metrics_known_values():
    result=metrics([1,0,1,0],[.9,.8,.2,.1],.5)
    assert result['f1']==.5
    assert result['precision']==.5
    assert result['recall']==.5
    assert result['confusion_matrix']==dict(tn=1,fp=1,fn=1,tp=1)
    assert result['average_precision']==pytest.approx(5/6)
    assert result['pr_auc_trapezoidal']==pytest.approx(19/24)


def test_unknown_labels_are_excluded_not_negative():
    base=metrics([1,0],[.9,.1],.5)
    added=metrics([1,0,-1],[.9,.1,.99],.5)
    for key in ['f1','precision','recall','average_precision','pr_auc_trapezoidal','confusion_matrix']:
        assert base[key]==added[key]
    assert added['unknown']==1
    assert metrics([-1],[.8],.5)['f1'] is None


def test_single_class_withheld_and_no_positive_prediction():
    r=metrics([0,0],[.1,.2],.5)
    assert r['average_precision'] is None and r['pr_auc_trapezoidal'] is None
    assert r['f1']==0 and r['warnings']


def test_threshold_uses_supplied_validation_labels_only():
    assert choose_threshold([0,0,1,1],np.array([.1,.4,.6,.9]))==.6


def test_elliptic_format_and_class_mapping():
    raw='99,35,'+','.join(['0.1']*165)+'\n'
    ids,times,x=read_features(io.StringIO(raw))
    assert ids.tolist()==['99'] and times.tolist()==[35] and x.shape==(1,165)
    classes=read_classes(io.StringIO('txId,class\n99,1\n98,2\n97,unknown\n'))
    assert classes=={'99':'1','98':'2','97':'unknown'}
    with pytest.raises(ValueError):read_classes(io.StringIO('txId,class\n99,0\n'))
    with pytest.raises(ValueError):read_features(io.StringIO('99,35,0.1\n'))
    with pytest.raises(ValueError):read_features(io.StringIO(raw+raw))
    with pytest.raises(ValueError):read_features(io.StringIO(raw.replace('0.1','NaN',1)))


def test_transaction_fields_do_not_require_relay_metadata():
    row={'txid':'a'*64,'input_addresses':['in'],'output_addresses':['out'],'input_amounts':[10000],'output_amounts':[9900],'label':1}
    r=Record.model_validate(row)
    assert r.fee==100 and r.src_ip is None and r.timestamp is None
    for bad in [True,'1',2]:
        with pytest.raises(ValueError):Record.model_validate({**row,'label':bad})


def test_synthetic_evaluation_not_a_self_label():
    data={'id':'example','model_version':'test','source_sha256':'hash','synthetic':True,
          'alerts':[{'txid':'a','flagged':True,'anomaly_score':.9},{'txid':'b','flagged':False,'anomaly_score':.1}]}
    assert evaluate_dataset(data)['status']=='labels_required'
    data['labels']={'a':0,'b':1}
    result=evaluate_dataset(data)
    assert result['f1']==0
    assert result['confusion_matrix']==dict(tn=0,fp=1,fn=1,tp=0)
