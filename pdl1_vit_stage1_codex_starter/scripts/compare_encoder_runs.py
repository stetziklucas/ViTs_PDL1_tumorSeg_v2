from __future__ import annotations
import argparse, json
from pathlib import Path

METRICS=["false_positive_px","false_negative_px","precision","sensitivity","f1","training_log_loss_total","annotated_positive_px","annotated_negative_px","annotated_total_px"]

def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline-tag", required=True)
    p.add_argument("--candidate-tag", required=True)
    p.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    p.add_argument("--output-dir", type=Path, required=True)
    a=p.parse_args()
    b=_load(a.outputs_root/f"reports_training_{a.baseline_tag}"/"training_summary.json")
    c=_load(a.outputs_root/f"reports_training_{a.candidate_tag}"/"training_summary.json")
    ba=b.get("aggregate_metrics",{}) ; ca=c.get("aggregate_metrics",{})
    side={m:{"baseline":ba.get(m),"candidate":ca.get(m),"candidate_minus_baseline":(ca.get(m)-ba.get(m)) if isinstance(ba.get(m),(int,float)) and isinstance(ca.get(m),(int,float)) else None} for m in METRICS}
    out={"baseline_tag":a.baseline_tag,"candidate_tag":a.candidate_tag,"baseline_encoder_provenance":b.get("encoder_provenance"),"candidate_encoder_provenance":c.get("encoder_provenance"),"aggregate_metrics":side,"caveat":"Annotated-region development metrics only; not whole-slide validation and not clinical performance."}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir/"encoder_comparison_summary.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=["# Encoder Comparison Summary","",f"baseline_tag: {a.baseline_tag}",f"candidate_tag: {a.candidate_tag}","",out["caveat"],"","| metric | baseline | candidate | delta |","| --- | ---: | ---: | ---: |"]
    for k,v in side.items(): lines.append(f"| {k} | {v['baseline']} | {v['candidate']} | {v['candidate_minus_baseline']} |")
    (a.output_dir/"encoder_comparison_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
