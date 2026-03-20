## MICCAI experiments + scripts below 

1) generate noNORM csv for 40x dir of 2.5x and 5x
3) compare btw mags for concept buckets
Flip rate and direction by scale (10× vs 40×)

DONE:
2) run baseline exp on conch deep feats
4) extract feats at different scales
6) run conch simple classifier on uni feats
5) run abmil (did diff type of MIL) baselines on conch feats


“Which buckets explain flips” (mean Δbucket on flip set)

Patch budget curves (cap patches → does flip stability change?)

- extract conch v1.5 feats? try aligning text with uni feats?

PAPER idea
Concept probing + scale-disagreement analysis as the main method, with a lightweight slide-level classifier
scale-aware semantic probing

A unified disagreement-aware framework that handles both
(i) multiscale disagreement within a slide and
(ii) multi-slide disagreement within a patient,


* extract conch feats
python test_text_encoder\extract_CONCH_feats.py --slides_root "C:\Users\Vivian\Documents\CONCH\all_patches\patches_5x\40x" --patch_map_name "patch_map.csv" --out_h5_dir "C:\Users\Vivian\Documents\CONCH\conch_img_feats\5x_feats\40x\h5" --out_pt_dir "C:\Users\Vivian\Documents\CONCH\conch_img_feats\5x_feats\40x\pt" --project_root "C:\Users\Vivian\Documents\CONCH" --checkpoint_path "C:\Users\Vivian\Documents\CONCH\checkpoints\conch\pytorch_model.bin" --model_cfg "conch_ViT-B-16" --device cuda:0 --batch_size 64 --max_patches 0

* extracting unnorm conch feats --> changed line in script
python test_text_encoder\extract_CONCH_feats.py --slides_root "C:\Users\Vivian\Documents\CONCH\patches_tiled\patches_10x\40x" --patch_map_name "patch_map.csv" --out_h5_dir "C:\Users\Vivian\Documents\CONCH\conch_img_feats\10x_unnorm_feats\20x\h5" --out_pt_dir "C:\Users\Vivian\Documents\CONCH\conch_img_feats\10x_unnorm_feats\20x\pt" --project_root "C:\Users\Vivian\Documents\CONCH" --checkpoint_path "C:\Users\Vivian\Documents\CONCH\checkpoints\conch\pytorch_model.bin" --model_cfg "conch_ViT-B-16" --device cuda:0 --batch_size 64 --max_patches 0

* conch v1.5 didn't work
python trident\extract_CONCH_feats.py --slides_root "C:\Users\Vivian\Documents\CONCH\patches_tiled\patches_10x\40x" --patch_map_name "patch_map.csv" --out_h5_dir "C:\Users\Vivian\Documents\CONCH\conch1.5_img_feats\10x_feats\40x\h5" --out_pt_dir "C:\Users\Vivian\Documents\CONCH\conch1.5_img_feats\10x_feats\40x\pt" --project_root "C:\Users\Vivian\Documents\CONCH" --checkpoint_path "C:\Users\Vivian\Downloads\pytorch_model_vision.bin" --model_cfg "conch_ViT-B-16" --device cuda:0 --batch_size 64 --max_patches 0

python test_text_encoder\extract_conch15_feats.py --slides_root "C:\Users\Vivian\Documents\CONCH\patches_tiled\patches_10x\20x" --out_h5_dir "C:\Users\Vivian\Documents\CONCH\conch15_img_feats\h5" --out_pt_dir "C:\Users\Vivian\Documents\CONCH\conch15_img_feats\pt" --project_root "C:\Users\Vivian\Documents\CONCH"--use_hf --hf_repo "hf-hub:MahmoodLab/conch"

python test_text_encoder\extract_conch15_feats.py --slides_root "C:\Users\Vivian\Documents\CONCH\patches_tiled\patches_10x\40x" --out_h5_dir "C:\Users\Vivian\Documents\CONCH\run2_conch15_img_feats\h5" --out_pt_dir "C:\Users\Vivian\Documents\CONCH\run2_conch15_img_feats\pt" --device cuda:0 --batch_size 32 --patch_size_level0 224

* virchow2 feats
python extract_Virchow2_feats.py --slides_root "C:\Users\Vivian\Documents\CONCH\patches_tiled\patches_10x\40x" --patch_map_name "patch_map.csv" --out_h5_dir "C:\Users\Vivian\Documents\CONCH\virchow2_img_feats\10x_feats\40x\h5" --out_pt_dir "C:\Users\Vivian\Documents\CONCH\virchow2_img_feats\10x_feats\40x\pt" --project_root "C:\Users\Vivian\Documents\CONCH" --checkpoint_path "C:\Users\Vivian\Documents\TRIDENT\trident\trident\conchv1.5\pytorch_model_vision.bin" --use_v15 --device cuda:0 --batch_size 64 --max_patches 0

python test_text_encoder\extract_virchow2_feats.py --slides_root "C:\Users\Vivian\Documents\CONCH\patches_tiled\patches_10x\20x" --patch_map_name "patch_map.csv" --out_h5_dir "C:\Users\Vivian\Documents\CONCH\virchow2_img_feats\10x_feats\20x\h5" --out_pt_dir "C:\Users\Vivian\Documents\CONCH\virchow2_img_feats\10x_feats\20x\pt" --device cuda:0 --batch_size 64 --max_patches 0

* uni2 feats
python test_text_encoder\extract_uni2h_feats.py --slides_root "C:\Users\Vivian\Documents\CONCH\patches_tiled\patches_10x\20x" --patch_map_name "patch_map.csv" --out_h5_dir "C:\Users\Vivian\Documents\CONCH\uni2h_img_feats\h5"  --out_pt_dir "C:\Users\Vivian\Documents\CONCH\uni2h_img_feats\pt" --device cuda:0 --batch_size 64

# zeroshot classification
C:\Users\Vivian\Documents\CONCH\test_text_encoder\zeroshot_classification.ipynb

# generate concept prior
C:\Users\Vivian\Documents\CONCH\test_text_encoder\zeroshot_classificationv2.ipynb

# gated mil and other mil baselines
> C:\Users\Vivian\Documents\CONCH\test_text_encoder\vision_concept_v2.ipynb
- used this for results ^^
- revised mil?
> C:\Users\Vivian\Documents\CONCH\test_text_encoder\vision_concept_v3.ipynb

# simple classifier with concept scores


## MICCAI experiments ==================================================================== 
* change back to og block that processses our tiled patches. currently processed slideio slides
> C:\Users\Vivian\Documents\CONCH\test_text_encoder\zeroshot_classificationv2.ipynb. 

> C:\Users\Vivian\Documents\CONCH\test_text_encoder\vision_concept_v2.ipynb

* run ABMIL and CoGA-MIL baseline with other FE deep feats including Virchow2 and UNI
> C:\Users\Vivian\Documents\CONCH\test_text_encoder\virchow2_vision_concept_v2.py 

* create volcano plots from slide level preds npz files. can also save csv summarizing those to plot later on
> C:\Users\Vivian\Documents\CONCH\test_text_encoder\volcano plots\create_volcano_plots.ipynb

* create barplots and prelim concept maps
> C:\Users\Vivian\Documents\CONCH\test_text_encoder\concept_heatmap.ipynb

### documentation of files
* dir of slide predictions (concept and weighted attn scores)
> C:\Users\Vivian\Documents\CONCH\test_text_encoder\slide_concept_scores\slide_preds + slide_preds_2.5x + slide_preds_5x

* concept map used for MICCAI exp
> C:\Users\Vivian\Documents\CONCH\test_text_encoder\slide_concept_scores\V1_concept_map.csv

* concept priors generated from zeroshot_classificationv2.ipynb
> 10x: C:\Users\Vivian\Documents\CONCH\test_text_encoder\slide_concept_scores\noCAP_patch_ptfile_notopk_conf_PATCH.csv
> 5x: C:\Users\Vivian\Documents\CONCH\test_text_encoder\slide_concept_scores\5x\noCAP_patch_ptfile_notopk_conf_PATCH.csv
> 2.5x: C:\Users\Vivian\Documents\CONCH\test_text_encoder\slide_concept_scores\2.5x\noCAP_patch_ptfile_notopk_conf_PATCH.csv

* summarizes top 15 concepts per slide
> C:\Users\Vivian\Documents\CONCH\test_text_encoder\slide_concept_scores\noCAP_patch_ptfile_notopk_conf_topconcepts.csv

* Re-tiled slides for MassVis viz
> C:\Users\Vivian\Documents\CONCH\test_text_encoder\FA_113_B1_massVis_conf_PATCH.csv
> C:\Users\Vivian\Documents\CONCH\test_text_encoder\PT_78_B2_massVis_conf_PATCH.csv

* volcano plots for multi-scale and spearman correlation tests
> C:\Users\Vivian\Documents\CONCH\test_text_encoder\vision_concept_v3.ipynb 