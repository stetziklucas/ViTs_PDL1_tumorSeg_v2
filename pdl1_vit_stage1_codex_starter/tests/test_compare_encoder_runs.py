import json
from pathlib import Path
from scripts import compare_encoder_runs

def test_compare(tmp_path, monkeypatch):
    out=tmp_path/'outputs'; (out/'reports_training_a').mkdir(parents=True); (out/'reports_training_b').mkdir(parents=True)
    base={'aggregate_metrics':{'precision':0.1,'false_positive_px':1,'false_negative_px':2,'sensitivity':0.2,'f1':0.3,'training_log_loss_total':1.2,'annotated_positive_px':10,'annotated_negative_px':20,'annotated_total_px':30},'encoder_provenance':{'encoder_id':'current_timm'}}
    cand={'aggregate_metrics':{'precision':0.2,'false_positive_px':2,'false_negative_px':1,'sensitivity':0.3,'f1':0.4,'training_log_loss_total':1.0,'annotated_positive_px':10,'annotated_negative_px':20,'annotated_total_px':30},'encoder_provenance':{'encoder_id':'hibou_b'}}
    (out/'reports_training_a'/'training_summary.json').write_text(json.dumps(base))
    (out/'reports_training_b'/'training_summary.json').write_text(json.dumps(cand))
    target=out/'cmp'
    monkeypatch.setattr('sys.argv',['x','--baseline-tag','a','--candidate-tag','b','--outputs-root',str(out),'--output-dir',str(target)])
    compare_encoder_runs.main()
    assert (target/'encoder_comparison_summary.json').exists()
