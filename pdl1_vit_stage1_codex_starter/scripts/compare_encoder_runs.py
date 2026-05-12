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
    be=b.get("encoder_provenance") or (next(iter((b.get("per_run_encoder_provenance") or {}).values()), None) if isinstance(b.get("per_run_encoder_provenance"), dict) else None)
    ce=c.get("encoder_provenance") or (next(iter((c.get("per_run_encoder_provenance") or {}).values()), None) if isinstance(c.get("per_run_encoder_provenance"), dict) else None)
    out={"baseline_tag":a.baseline_tag,"candidate_tag":a.candidate_tag,"baseline_encoder_provenance":be,"candidate_encoder_provenance":ce,"aggregate_metrics":side,"caveat":"Annotated-region development metrics only; not whole-slide validation and not clinical performance."}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir/"encoder_comparison_summary.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=["# Encoder Comparison Summary","",f"baseline_tag: {a.baseline_tag}",f"candidate_tag: {a.candidate_tag}","",out["caveat"],"", "Baseline encoder:", f"- display_name: {(be or {}).get("encoder_display_name")}", f"- encoder_id: {(be or {}).get("encoder_id")}", f"- backend: {(be or {}).get("encoder_backend")}", f"- model_name: {(be or {}).get("encoder_model_name")}", f"- embedding_dim: {(be or {}).get("embedding_dim")}", "", "Candidate encoder:", f"- display_name: {(ce or {}).get("encoder_display_name")}", f"- encoder_id: {(ce or {}).get("encoder_id")}", f"- backend: {(ce or {}).get("encoder_backend")}", f"- model_name: {(ce or {}).get("encoder_model_name")}", f"- embedding_dim: {(ce or {}).get("embedding_dim")}", "", "| metric | baseline | candidate | delta |","| --- | ---: | ---: | ---: |"]
    for k,v in side.items(): lines.append(f"| {k} | {v['baseline']} | {v['candidate']} | {v['candidate_minus_baseline']} |")
    (a.output_dir/"encoder_comparison_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
