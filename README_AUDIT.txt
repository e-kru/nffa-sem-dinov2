Reconstructed YOLO code for review; not original source scripts.

train_yolo11n_v2.py:
 - Reads already prepared YOLO train/val images and labels and data.yaml
 - Ultralytics YOLO11n train settings recovered from saved args.yaml
 - Writes to runs/v2_reproduction only; will never overwrite trained v2
 - Does NOT reconstruct training-image selection or original annotation process
 - Preprocessing provenance of saved training images must be checked separately

audit_yolo11n_v2.py:
 - Reads V2 manifest (18,614 included images)
 - Opens original SEM image, applies preprocessing_v2.crop_metadata_strip
 - Runs saved best.pt at conf=0.05 and imgsz=640
 - Generates per-box and per-image detections in a NEW output folder
 - Exclusion decision for V3 uses threshold >=0.50, with box precedence
 - No masking or extra cropping from YOLO
 - A later reproduction run may deviate due to software versions or settings

Run from repo root using `python -m scripts.audit_yolo11n_v2`, not `python scripts/...`,
so that `src` can be imported without special PYTHONPATH handling.

The original V3 manifest builder is already verified byte-by-row/field values
against 21,169 original manifest rows. Use the original stored audit_v2/detections.csv
for that verification. Do not overwrite it with this reconstructed script.

evaluate_linear_probe_A.py:
 - Reconstructed from B's original evaluation code with A paths
 - Checks original stored Split A metrics; writes only reproduction JSON
 - Does not re-extract features or overwrite the original test report

audit_dinov2_neighbors_v3.py:
 - Reconstructed GPU/CPU cosine similarity on cached V2 features for V3 images
 - Generates high cosine >=0.995 pair evidence needed for candidate_pairs_v3
 - Inference only, no re-preprocessing or extraction; audit against original pair list
 - New directory required; do not overwrite original neighbor files
