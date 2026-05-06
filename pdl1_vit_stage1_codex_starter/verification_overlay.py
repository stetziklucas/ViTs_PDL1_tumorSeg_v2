"""Helpers for generating annotated-region cropped verification masks."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw
from scripts.report_metrics import load_working_supervision_and_prediction_masks

ANNOTATION_LABEL_MAPPING={"background":0,"Positive_Tumor":1,"Negative_Tumor":2,"NonTumor":3,"Ignore":4}
PREDICTION_LABEL_MAPPING={"background":0,"pred_on_positive_tumor":1,"pred_on_negative_tumor":2,"pred_on_nontumor":3,"pred_on_ignore":4,"pred_outside_annotated_roi":5}

def _crop_bounds(mask,pad):
 ys,xs=np.nonzero(mask)
 if ys.size==0:return None
 h,w=mask.shape
 y0=max(0,int(ys.min())-pad); y1=min(h,int(ys.max())+1+pad); x0=max(0,int(xs.min())-pad); x1=min(w,int(xs.max())+1+pad)
 return y0,x0,y1-y0,x1-x0

def _polygon_mask(shape,verts):
 c=Image.new('L',(shape[1],shape[0]),0); d=ImageDraw.Draw(c); d.polygon([(float(x),float(y)) for y,x in verts],fill=1,outline=1); return np.asarray(c)>0

def _components(mask):
 h,w=mask.shape; vis=np.zeros_like(mask,bool); out=[]
 for y in range(h):
  for x in range(w):
   if not mask[y,x] or vis[y,x]: continue
   q=[(y,x)]; vis[y,x]=True; pts=[]
   while q:
    cy,cx=q.pop(); pts.append((cy,cx))
    for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
      if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not vis[ny,nx]: vis[ny,nx]=True; q.append((ny,nx))
   out.append(np.array(pts,dtype=int))
 return out

def _write_region_preview(base_rgb,ann_mask,pred_mask,out_path,thumb_path):
 img=base_rgb.copy(); img=np.clip(img*0.9+ann_mask[...,None]*np.array([60,90,255])[None,None,:],0,255).astype(np.uint8)
 tp=ann_mask&pred_mask; fn=ann_mask&(~pred_mask); fp=(~ann_mask)&pred_mask; img[tp]=[0,220,0]; img[fn]=[255,60,60]; img[fp]=[255,200,40]
 Image.fromarray(img).save(out_path); Image.fromarray(img).resize((128,128),Image.Resampling.NEAREST).save(thumb_path)

def generate_verification_overlay(*,image_id,run_tag,scribble_labels_path,positive_mask_path,output_dir,label_encoding,crop_padding_px=64,annotation_metadata_path=None,overlay_base_path=None):
 a=load_working_supervision_and_prediction_masks(image_id=image_id,scribble_labels_path=scribble_labels_path,positive_mask_path=positive_mask_path)
 scribble=a['scribble_working']; pred=a['positive_mask']>0
 annotated=np.isin(scribble,[int(label_encoding[k]) for k in ('Positive_Tumor','Negative_Tumor','NonTumor','Ignore') if k in label_encoding])
 bounds=_crop_bounds(annotated,crop_padding_px); output_dir.mkdir(parents=True,exist_ok=True)
 summary={"verification_regions_available":False,"verification_regions_path":None,"verification_region_count":0,"verification_regions_warning":None,"image_id":image_id,"run_tag":run_tag,"verification_overlay_available":False,"verification_overlay_mode":"positive_mask_working_crop"}
 overlay_path=output_dir/'verification_overlay.png'; annotation_labels_path=output_dir/'verification_annotation_labels.png'; pred_labels_path=output_dir/'verification_prediction_labels.png'; regions_json=output_dir/'verification_regions.json'; summary_path=output_dir/'verification_overlay_summary.json'
 if bounds is None:
  summary['verification_regions_warning']='No annotated pixels found in scribble_labels; overlay unavailable.'
 else:
  y0,x0,ch,cw=bounds
  ann=np.zeros_like(scribble,dtype=np.uint8)
  for k,v in ANNOTATION_LABEL_MAPPING.items():
   if k!='background' and label_encoding.get(k) is not None: ann[scribble==int(label_encoding[k])]=int(v)
  ann_crop=ann[y0:y0+ch,x0:x0+cw]; pred_crop=pred[y0:y0+ch,x0:x0+cw]
  Image.fromarray((pred_crop*255).astype(np.uint8)).save(overlay_path); Image.fromarray(ann_crop).save(annotation_labels_path)
  pl=np.zeros_like(ann_crop,dtype=np.uint8); pl[pred_crop&(ann_crop==1)]=1; pl[pred_crop&(ann_crop==2)]=2; pl[pred_crop&(ann_crop==3)]=3; pl[pred_crop&(ann_crop==4)]=4; pl[pred_crop&(ann_crop==0)]=5; Image.fromarray(pl).save(pred_labels_path)
  summary.update({"verification_overlay_available":True,"verification_overlay_path":overlay_path.as_posix(),"verification_annotation_labels_available":True,"verification_annotation_labels_path":annotation_labels_path.as_posix(),"verification_prediction_labels_available":True,"verification_prediction_labels_path":pred_labels_path.as_posix(),"crop_y0":y0,"crop_x0":x0,"crop_h":ch,"crop_w":cw})
  base=np.asarray(Image.open(overlay_base_path).convert('RGB')) if overlay_base_path and overlay_base_path.exists() else np.full((pred.shape[0],pred.shape[1],3),96,dtype=np.uint8)
  if base.shape[:2]!=pred.shape: base=np.full((pred.shape[0],pred.shape[1],3),96,dtype=np.uint8)
  base_crop=base[y0:y0+ch,x0:x0+cw]; rows=[]; regions=[]
  if annotation_metadata_path and annotation_metadata_path.exists():
    payload=json.loads(annotation_metadata_path.read_text())
    for i,p in enumerate(payload.get('polygons',[]) if isinstance(payload,dict) else []):
      v=np.asarray(p.get('vertices',[]),dtype=float)
      if v.ndim!=2 or v.shape[0]<3 or v.shape[1]!=2: continue
      sy,sx=ch/max(1,scribble.shape[0]),cw/max(1,scribble.shape[1])
      v2=np.column_stack([(v[:,0]*sy),(v[:,1]*sx)])
      m=_polygon_mask((ch,cw),v2)
      if np.any(m): regions.append({'mask':m,'class_name':str(p.get('class_name') or 'Unknown'),'annotation_index':i,'source_type':'annotation_polygon','bbox_annotation_yxhw':[int(v[:,0].min()),int(v[:,1].min()),max(1,int(v[:,0].max()-v[:,0].min()+1)),max(1,int(v[:,1].max()-v[:,1].min()+1))]})
  if not regions:
    for cls,lbl in ANNOTATION_LABEL_MAPPING.items():
      if lbl==0: continue
      for comp in _components(ann_crop==lbl):
        m=np.zeros_like(ann_crop,bool); m[comp[:,0],comp[:,1]]=True
        regions.append({'mask':m,'class_name':cls,'annotation_index':None,'source_type':'annotation_component'})
  for i,r in enumerate(regions):
    m=r['mask']; pred_pos=pred_crop; annpx=int(m.sum()); pp=int((pred_pos&m).sum()); cls=r['class_name']
    if annpx==0: continue
    if cls=='Positive_Tumor': cp,ep,sn,sc,iss=pp,annpx-pp,'sensitivity',pp/annpx,'positive_annotation_missed_by_prediction' if annpx-pp>0 else 'positive_annotation_detected'
    elif cls in {'Negative_Tumor','NonTumor'}: cp,ep,sn,sc,iss=int((~pred_pos&m).sum()),pp,'specificity',int((~pred_pos&m).sum())/annpx,'false_positive_in_negative_context' if pp>0 else 'negative_context_clean'
    else: cp,ep,sn,sc,iss=0,pp,'ignored',None,'ignored_annotation_contains_prediction' if pp>0 else 'ignored_annotation_clean'
    ys,xs=np.where(m); by,bx,bh,bw=int(ys.min()),int(xs.min()),int(ys.max()-ys.min()+1),int(xs.max()-xs.min()+1)
    prv=output_dir/'verification_regions'/f'region_{i:04d}_preview.png'; th=output_dir/'verification_regions'/f'region_{i:04d}_thumb.png'; prv.parent.mkdir(parents=True,exist_ok=True); _write_region_preview(base_crop,m,pred_pos,prv,th)
    rows.append({'region_id':f'region_{i:04d}','source_type':r['source_type'],'annotation_index':r.get('annotation_index'),'class_name':cls,'bbox_working_yxhw':[by,bx,bh,bw],'bbox_annotation_yxhw':r.get('bbox_annotation_yxhw'),'center_annotation_yx':None if r.get('bbox_annotation_yxhw') is None else [int(r['bbox_annotation_yxhw'][0]+r['bbox_annotation_yxhw'][2]//2),int(r['bbox_annotation_yxhw'][1]+r['bbox_annotation_yxhw'][3]//2)],'annotated_px':annpx,'pred_positive_px':pp,'correct_px':int(cp),'error_px':int(ep),'score_name':sn,'score':sc,'review_priority':int(ep),'issue':iss,'preview_path':prv.as_posix(),'thumbnail_path':th.as_posix()})
  warn=None if rows else 'No regions generated from polygons or annotation components.'
  regions_json.write_text(json.dumps({'region_count':len(rows),'regions':rows,'warning':warn},indent=2)+'\n')
  summary.update({'verification_regions_available':len(rows)>0,'verification_regions_path':regions_json.as_posix(),'verification_region_count':len(rows),'verification_regions_warning':warn})
 summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 return {**summary,'verification_overlay_summary_path':summary_path.as_posix()}
