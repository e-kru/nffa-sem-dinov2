# Frozen DINOv2: gemeinsamer Downstream-Runner

Einstieg: `python -m scripts.evaluate_sem_encoder`. Der Runner unterstützt
eingefrorene ViT-S/14-Checkpoints für Klassifikation und Aachen-Segmentierung.
Er ruft kein SSL auf. Auswahl und Auswertung verwenden ausschließlich Validation.

## Auswahl und unveränderte Heads

```bash
python -m scripts.evaluate_sem_encoder --task classification --heads logreg mlp --stage preflight
python -m scripts.evaluate_sem_encoder --task segmentation --heads linear cnn8 --stage preflight
python -m scripts.evaluate_sem_encoder --task both \
  --classification-heads logreg mlp --segmentation-heads linear cnn8 --stage preflight
```

Defaults: Klassifikation `logreg mlp`, Segmentierung `linear cnn8`. `--heads`
gilt für eine einzelne Aufgabe; bei `both` gelten die beiden aufgabenspezifischen
Optionen. Doppelte, widersprüchliche oder für die Aufgabe ungeeignete Namen
werden abgelehnt. Die Head-Auswahl verändert die Feature-Extraktion nicht.

| Head | Bestehende Implementierung | Festgelegte Konfiguration |
| --- | --- | --- |
| `logreg` | sklearn `LogisticRegression`, wie in `train_linear_probe_A.py` und `train_linear_probe_B.py` | Train-only `StandardScaler`; C = 0.001, 0.01, 0.1, 1, 10, 100; lbfgs; max_iter 3000; Validation Macro-F1, erster Kandidat bei Gleichstand |
| `mlp` | `src.downstream.heads.ClassificationHead(kind="mlp")` | 384 → 512 → 256 → 128 → 10; BatchNorm1d/ReLU/Dropout **0,3** nach jedem Hidden Layer; Train-only `StandardScaler`; AdamW, LR 0.001, WD 0.0001; Batch 256; 100 Epochen; Patience 15; Validation Macro-F1, CE als Tie-Break mit 1e-8 Toleranz |
| `linear` | `src.aachen_ablation.engine.make_decoder("linear")` → vorhandener `SegmentationHead` | Pixelweise 1×1-Conv, 384 → 2; AdamW, LR 0.001, WD 0.0001 |
| `cnn8` | `src.aachen_ablation.engine.make_decoder("cnn8")` → vorhandener `DeepCNN(384, 8)` | Unveränderte Projektion auf 128 Kanäle, vier Residualblöcke mit je zwei räumlichen Conv-Layern, vorhandene Normalisierung/Aktivierung/Dropout; AdamW, LR 0.0003, WD 0.0001 |

Der ältere allgemeine `run_downstream.py` hat andere Defaults, insbesondere
Dropout 0,0. Hier gilt die ursprüngliche 3-Hidden-Layer-Konfiguration aus dem
archivierten `train_mlp_head.py`, also Dropout 0,3. Der neue MLP-Trainer verwendet
den vorhandenen Head und bewahrt Batching, Optimizer, train-only Scaling,
Auswahl und Reihenfolge der ursprünglichen Trainings-/Evaluationsschritte.
Es gibt keine neue Head-Architektur und keine erneute Hyperparametersuche für MLP.

Beide Segmentierungs-Heads nutzen unverändert A3 im vorhandenen Aachen-Runner:
**896 × 672 Pixel (W × H)**, **384 × 48 × 64 Features (C × H × W)**, 30 Epochen,
Patience 8, effektiver Batch 8, Microbatch 8, Validation-Batch 8, FP32-Features
und FP32-Decodertraining. Standardseed: 42. Die vorhandenen Geometrie- und
Trainingsfunktionen werden direkt aufgerufen.

Loss: mittlere valide Pixel-CE je Originalbild, danach Mittel über Originalbilder.
Die vorhandene Letterbox-Rückprojektion bringt Logits auf die unveränderte
Original-Maskengeometrie, bevor der Threshold angewendet wird. Ignore-Label: 255.
Beste Epoche: Validation-mIoU bei Threshold 0,5, CE als Tie-Break. Anschließend
werden die vorhandenen Thresholds 0,05 bis 0,95 in 0,05-Schritten auf Validation
geprüft. Auswahl: höchste MA-IoU, bei Gleichstand nächster Threshold zu 0,5.
Reports trennen **feste 0,5 und validation-optimierte Werte**.

Die historische A3-CNN8-Validation von 42,35 % MA-IoU / 67,88 % mIoU gehört zum
validation-optimierten Threshold 0,35. Die historischen Klassifikations-Testwerte
90,87 % / 87,85 % Macro-F1 sind keine Validation-Referenzen. Vollständige
Benchmarks wurden für diesen Auftrag nicht erneut ausgeführt.

## Dateien und Zuständigkeiten

| Datei | Aufgabe |
| --- | --- |
| `scripts/evaluate_sem_encoder.py` | Head-Auswahl, Phasen, Encoder-Identität, Aufgabensteuerung und gemeinsamer Report |
| `scripts/verify_frozen_encoder.py` | Isolierter CPU-Prozess: gepinnter Backbone versus lokal gecachter alter Torch-Hub-Code |
| `src/downstream/benchmark.py` | Manifest-/Split-Prüfung, ausschließlich Train-/Val-Bilder, globale Feature-Caches, Integrität und Resume |
| `src/downstream/probes.py` | Ursprüngliche LogReg-/MLP-Konfigurationen, Fitting des vorhandenen MLP-Heads und Validation |
| `src/dinov2_encoder.py` | Unveränderter historischer Hub-Loader und wiederverwendetes striktes Checkpoint-Laden |
| `src/downstream/pinned_encoder.py` | Zusätzlicher `get_pinned_dinov2()` für striktes lokales Laden ohne Download; hält die historische Factory und deren Cache-Fingerprints unverändert |
| `src/downstream/data.py` | Wiederverwendetes Klassifikationsdataset mit ursprünglicher V2-Vorverarbeitung |
| `src/downstream/heads.py` | Unveränderte Head-Architekturen |
| `scripts/run_aachen_ablation.py` | Bestehende A3-Extraktion, Training, Evaluation und Aachen-Reports; optionaler gepinnter Encoder-Ladepfad |
| `src/aachen_ablation/` | Bestehende Daten-, Geometrie-, Cache-, Trainings-, Threshold- und Metrikimplementierung |
| `tests/test_encoder_evaluation.py` | Synthetische CPU-Tests für Heads, Protokolle, Cache-Isolation, Testausschluss und Phasen |

Historische Extraktions-/Baseline-Skripte und Ausgaben bleiben erhalten. Der neue
Runner verwendet ihre alten Feature-Verzeichnisse nicht.

## Checkpoints, Datenzugriff und Resume

Default ist das vorhandene originale Meta-Release mit dem SHA256 aus
`dinov2-upstream.json`. `--encoder-checkpoint PATH` akzeptiert kompatible rohe
Backbone-States einschließlich `teacher_backbone.pth` sowie die vorhandenen
Wrapper `state_dict` und `encoder_state_dict`. Vollständige SSL-Trainingsstates
mit Student, Teacher, Heads und Optimizer sind keine Backbone-Exports und werden
strikt abgelehnt.

Vor Extraktion/Training/Evaluation ist ein erfolgreicher `preflight` erforderlich.
Zwei frische CPU-Prozesse laden denselben Checkpoint: einmal mit dem alten lokal
gecachten Hub-Code, einmal mit Metas gepinntem Checkout. Das verhindert, dass
Pythons Import-Cache eine falsche Gleichheitsprüfung erzeugt. Ohne Netzwerk
werden Frozen-/Eval-Zustand, State-Fingerprint und synthetische Forward-Ausgaben
geprüft: global (1,384), dense (1,384,48,64). FP32-CPU-Features müssen exakt
gleich sein. Spätere Phasen prüfen Checkpoint, Software und Ladecode gegen
diesen gespeicherten Nachweis.

Pro Encoder entsteht ein Ordner aus dem Checkpoint-SHA256. Caches speichern
Manifest-/Split-/Bild-/Quellcode-Fingerprints, Sample-Reihenfolge, Tensorformen
und Datei-Checksummen. Der gemeinsame Report prüft auch gleiche tatsächliche
Backbone-State-Fingerprints zwischen Aufgaben und Klassifikationssplits.
Konfigurationsabweichungen in bestehenden Runs führen zu Fehlern; ein neues
Protokoll braucht einen neuen `--output-root`.

Für spätere SEM-Exporte zusätzlich den SSL-Laufbericht und dessen
`train_records.json` archivieren: Diese dokumentieren den verwendeten Korpus
und die Train-/Val-/Test-Ausschlüsse. Die rohe Gewichtedatei enthält diese
Informationen nicht; der Runner prüft Ladegleichheit und Evaluationsgrenzen.

A und B erhalten getrennte Klassifikationscaches. Ein A-Testbild wird für A
weder geöffnet noch gecacht, auch wenn es in B zum Training gehört. Der Runner
prüft Split-/Gruppenidentitäten über die komplette CSV, interpretiert Labels
und öffnet Dateien aber nur für Train/Val. Die verwendeten Labels müssen mit
dem kuratierten Manifest übereinstimmen. Gruppen dürfen Partitionsgrenzen nicht
überschreiten. Aachen verwendet die vorhandenen Gruppenregeln; Test-RGBs und
Testmasken bleiben ungeöffnet.

`--resume` überspringt passende vollständige Artefakte und setzt fehlende
Feature-Chunks beziehungsweise unterbrochene Head-Trainings an Epochengrenzen
fort. LogReg beginnt nach Unterbrechung neu; MLP und Aachen sichern Optimizer
und RNG. Ein bereits publizierter, nach einem Absturz aber noch nicht indizierter
globaler Feature-Chunk muss zunächst geprüft werden; er wird nicht überschrieben.
Joblib-Modelle und Resume-States werden aus den eigenen Laufverzeichnissen geladen.

## Manuelle A100-Befehle

Vorbereitet, **nicht gestartet**. Auf `gpu-host` mit genau einer verfügbaren
A100 80 GB ausführen. Die ausgewählte physische GPU heißt im Prozess `cuda:0`.

```bash
cd /path/to/nffa-sem-dinov2
source /path/to/workspace/apps/miniconda3/etc/profile.d/conda.sh
conda activate sem-gpu
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export DINOV2_ROOT=/path/to/workspace/external/dinov2
EVAL_PY=python
EVAL_OUT=outputs/encoder_evaluation/v1
```

1. Originalgewichte: Preflight beider Aufgaben auf CPU:

```bash
"$EVAL_PY" -m scripts.evaluate_sem_encoder --task both --stage preflight \
  --device cpu --workers 8 --seed 42 --output-root "$EVAL_OUT"
```

2. Neue globale Original-DINOv2-Features für A und B:

```bash
"$EVAL_PY" -m scripts.evaluate_sem_encoder --task classification --heads logreg mlp \
  --stage extract --device cuda:0 --workers 8 --seed 42 --output-root "$EVAL_OUT" --resume
```

3. Klassifikations-Heads je Split. Für ausschließlich LogReg `--heads logreg`
verwenden; Features und LogReg-Hyperparameter bleiben gleich:

```bash
"$EVAL_PY" -m scripts.evaluate_sem_encoder --task classification --heads logreg mlp \
  --splits A --stage train --device cuda:0 --workers 8 --seed 42 --output-root "$EVAL_OUT" --resume
"$EVAL_PY" -m scripts.evaluate_sem_encoder --task classification --heads logreg mlp \
  --splits B --stage train --device cuda:0 --workers 8 --seed 42 --output-root "$EVAL_OUT" --resume
```

4. Neue A3-Aachen-Features:

```bash
"$EVAL_PY" -m scripts.evaluate_sem_encoder --task segmentation --heads linear cnn8 \
  --stage extract --device cuda:0 --workers 8 --seed 42 --output-root "$EVAL_OUT" --resume
```

5. Vorhandene Decoder trainieren; für nur CNN8 `--heads cnn8` verwenden:

```bash
"$EVAL_PY" -m scripts.evaluate_sem_encoder --task segmentation --heads linear cnn8 \
  --stage train --device cuda:0 --workers 8 --seed 42 --output-root "$EVAL_OUT" --resume
```

6. Validation beider Aufgaben:

```bash
"$EVAL_PY" -m scripts.evaluate_sem_encoder --task both --stage evaluate \
  --device cuda:0 --workers 8 --seed 42 --output-root "$EVAL_OUT" --resume
```

7. Gemeinsamer Report, ohne GPU-Berechnung. Wurden nur bestimmte Heads trainiert,
diese auch hier wählen; fehlende Ergebnisse werden ausdrücklich gelistet:

```bash
"$EVAL_PY" -m scripts.evaluate_sem_encoder --task both --stage report \
  --workers 8 --seed 42 --output-root "$EVAL_OUT"
```

8. Später denselben Ablauf für einen echten SSL-Export ausführen. Platzhalter
durch den tatsächlichen Export ersetzen. Die Phasen laufen sequenziell; bei
einem Fehler wird abgebrochen:

```bash
EVAL_CKPT=/path/to/nffa-sem-dinov2/outputs/sem_ssl_checks/YOUR_RUN/teacher_backbone.pth
for eval_stage in preflight extract train evaluate report; do
  "$EVAL_PY" -m scripts.evaluate_sem_encoder --task both \
    --classification-heads logreg mlp --segmentation-heads linear cnn8 \
    --encoder-checkpoint "$EVAL_CKPT" --stage "$eval_stage" \
    --device cuda:0 --workers 8 --seed 42 --output-root "$EVAL_OUT" --resume || break
done
```

## Ausgaben und Verifikation

```text
outputs/encoder_evaluation/v1/dinov2_vits14_<checkpoint-sha256-prefix>/
├── provenance.json
├── encoder_verification.json
├── classification/
│   ├── split_A/
│   │   ├── preflight.json
│   │   ├── features/cache_metadata.json + chunk_*.npz
│   │   ├── logreg/seed42/{config.json,model.joblib,validation_results.json,validation_metrics.json,...}
│   │   └── mlp/seed42/{config.json,best_model.pt,last_state.pt,scaler.joblib,history.csv,validation_metrics.json,...}
│   └── split_B/...
├── segmentation/
│   ├── caches/A3_<contract-hash>/...
│   └── runs/A3/
│       ├── linear/seed42_<config-hash>/...
│       └── cnn8/seed42_<config-hash>/...
└── reports/<report-content-hash>/{report.json,report.md}
```

Original-Prefix: `b938bf1bc15cd2ec`. Klassifikation: Validation Macro-F1 und
Accuracy. Segmentierung: MA-IoU, mIoU, MA-Precision und MA-Recall, fest und
validation-selektiert. Detaillierte Aachen-Metriken bleiben im bisherigen Format.
Scores werden als Bruchteile zwischen 0 und 1 gespeichert, nicht als Prozentwerte.

`--smoke-test --device cpu` aktiviert einen **Fake-Encoder**, wenige Samples und
zwei Epochen, aber dieselben vier Head-Architekturen. Artefakte liegen unter
`NONSCIENTIFIC_SMOKE`; dieser Modus belegt keine Baseline-Reproduktion oder
echte Encoder-Ladegleichheit.

Die synthetischen Tests lassen Testdateien bewusst fehlen und verwenden einen
ungültigen Testlabel-String. LogReg wird gegen das vollständige ursprüngliche
Sechs-C-Protokoll geprüft. Wenn der archivierte Original-MLP lokal vorhanden ist,
prüft ein unabhängiger Zwei-Epochen-Vergleich exakte Gewichte und Trainingshistorie.
Beide Aufgaben werden mit allen vier Heads phasenweise getestet. Zusätzlich
wird die echte CPU-Ladegleichheit für Original-Release und vorhandenen
synthetischen A100-SSL-Export geprüft. Das belegt Format-/Forward-Kompatibilität;
es belegt keine verbesserte SEM-Repräsentation.

Ergebnis am 11. November 2026 in `sem-gpu`: **146 Tests bestanden,
2 übersprungen und 3 Subtests bestanden**. Die beiden ausgelassenen Tests
benötigen Unix-IPC-Sockets, die diese Sandbox nicht zulässt. Der Original-MLP-
Vergleich wurde hier ausgeführt und bestanden. Bei beiden echten geprüften
Checkpoints war die maximale globale und räumliche Feature-Abweichung **0,0**.
Die kompakte Evidenz steht in
validation_summary.json;
die vollständigen Logs und Ladeberichte liegen unter
`outputs/encoder_evaluation_checks/heads_final_2026-11-11/`.

Auch der vollständige CPU-Preflight mit den vorhandenen echten Daten wurde
ausgeführt und bestanden:

| Aufgabe | Train | Validation | Geprüfte Originalgeometrie |
| --- | --- | --- | --- |
| NFFA Split A | 12.801 | 2.742 | Ursprüngliche Klassifikationsbilder, Fingerprints aller Train-/Val-Dateien |
| NFFA Split B | 12.799 | 2.743 | Ursprüngliche Klassifikationsbilder, Fingerprints aller Train-/Val-Dateien |
| Aachen-Heerlen | 1.199 | 258 | Alle Originalbilder/-masken 1024 × 768; geplanter A3-Feature-Tensor 1 × 384 × 48 × 64 je Original |

Für diese Prüfung wurden keine Features echter Daten extrahiert und keine Heads
auf echten Daten trainiert. Dateiprüfung und Hashing verwenden bei Klassifikation
acht Threads mit Fortschrittsmeldungen; auf dem gemeinsamen Dateisystem kann
dies mehrere Minuten dauern. Die abschließende Testsuite wurde nach dieser
Parallelisierung erneut ausgeführt und bestanden.

Vollständige Original-Baselines sowie GPU-Laufzeit und Speicherbedarf des neuen
Workflows bleiben manuell zu verifizieren. Die `sem-gpu`-Umgebung wird für
diesen Auftrag nicht verändert.

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q
```
