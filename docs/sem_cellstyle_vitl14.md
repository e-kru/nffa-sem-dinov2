# SEM Cellstyle ViT-L/14 v1: Implementierung und Startanleitung

Status: vorbereitet, kein wissenschaftlicher Lauf gestartet. Der neue SEM-Korpus
ist noch nicht freigegeben. Auch die Original-Large-Gewichte und die physische
Batchgröße sind in der Vorlage bewusst nicht aufgelöst. Es wurden keine Pakete
installiert und keine A100-/GPU-Jobs gestartet. Der bestehende Dataset-Audit und
historische Experimente werden nicht verändert.

Dies ist ein kombiniertes, an Cell-DINO orientiertes Anpassungsrezept, keine
behauptete Reproduktion einer publizierten Cell-DINO+-Konfiguration. Ein Erfolg
würde die praktische Wirkung des gesamten Rezepts zeigen, keine Einzelursache.

## Dateien und Zuständigkeiten

| Datei | Aufgabe |
| --- | --- |
| `configs/sem_ssl/sem_cellstyle_vitl14_v1.yaml` | Vollständige neue Rezeptur; Manifest, Receipt, Originalgewichte/SHA und physische Batch bleiben zunächst `null`. |
| `configs/sem_ssl/templates/sem_corpus_manifest.example.csv` | Nicht ausführbare Beispieltabelle für die spätere kuratierte Bildliste. |
| `configs/sem_ssl/templates/sem_corpus_receipt.example.json` | Nicht freigegebener Receipt mit Bildzahl, Manifestidentität, Quellenregeln und dokumentierten Ausschlüssen. |
| `src/sem_cellstyle/recipe.py` | Strikte Large-Konfiguration, Budgetumrechnung, getrennte absolute LR-Schedules, unveränderlicher Resume-Vertrag. |
| `src/sem_cellstyle/data.py` | Manifestanbindung, quadratische SEM-Crops, Intensität/Orientierung, originaler Meta-Mask-Collate und fortsetzbarer Präsentationsstrom. |
| `src/sem_cellstyle/training.py` | Accumulation, ausreichende Center-Statistiken, getrennte Parametergruppen, Student-Activation-Checkpointing. |
| `src/sem_cellstyle/runner.py` | Initialisierung, Update-Schleife, Signale, Recovery, vollständige Checkpoints und Teacher-Exportprüfung. |
| `src/sem_cellstyle/benchmark.py` | Messung kompletter Large-Updates einschließlich Accumulation, Resume-/Exportprüfung, Batchauswahl. |
| `src/sem_cellstyle/reporting.py` | Gemeinsamer Validation-Vergleich aller vorab festgelegten Large-Meilensteine. |
| `scripts/train_sem_cellstyle.py` | Plan, Preflight, Augmentierungsbilder, Training und Exportprüfung; Standardaktion ist nur `plan`. |
| `scripts/benchmark_sem_cellstyle_a100.py` | Jeder Batchkandidat in eigenem Prozess; CSV/JSON und zwei Diagramme. |
| `scripts/evaluate_sem_cellstyle.py` | Generiert oder startet bestehende Validation-Probes für die festgelegten Meilensteine. |
| `src/ssl/adapter.py` | Vorhandene Crop-/Loss-Routen, jetzt opt-in auch mit originalem Meta-Large-Backbone und aufgeschobenen Center-Statistiken. |
| `src/ssl/checkpoints.py` | Bestehende atomare vollständige Snapshots, ergänzt um optionalen Accumulation-Datencursor. |
| `scripts/evaluate_sem_encoder.py`, `scripts/run_aachen_ablation.py` | Bestehende Evaluation; explizites `--encoder-name dinov2_vitl14`. Small bleibt Standard. |
| `src/downstream/{pinned_encoder,benchmark,probes}.py`, `src/aachen_ablation/{cache,engine}.py` | Striktes Laden und Featuredimension aus Encoder/Cache; unveränderte bestehende Heads. |
| `tests/test_sem_cellstyle.py` | Kleine CPU-Fixtures mit originalen Meta-Komponenten; keine Large-Performancebehauptung. |

## Datenanbindung: bewusst noch kein Datensatz

Der Provider liest ausschließlich eine ausdrücklich konfigurierte CSV. Er führt
keine Verzeichnissuche aus, übernimmt keine Audit-Kandidaten und berechnet keine
neuen Bildinhalts-Hashes. Er prüft SHA256 der kleinen Manifest-/Receipt-Dateien;
die Herkunft und Integrität des tatsächlichen Korpus muss die vorgelagerte
Kuration belegen. Ziel sind200.000 verschiedene Trainingsbilder;100.000 sind
ein möglicher erster Korpus, keine hart erzwungene Mindestzahl. Dieses Interface ersetzt keine Leakage-Prüfung.

CSV-Spalten: `image_id,path,source,original_id,specimen_id,split`.
`path` ist absolut, `split` immer `train`. IDs und Bildpfade dürfen nicht doppelt
auftreten; Originalaufnahme und Probe werden auch bei vorhandenen Tiles erfasst.
Tiles/Augmentierungen dürfen nicht als unabhängige Ursprungsaufnahmen ausgegeben
werden. `verified_train_originals` zählt die freigegebenen Manifestbilder,
niemals die zufälligen Crops. Bei Tiles muss die Kuration ihre Abhängigkeiten und
Zählweise offenlegen.

Der separate JSON-Receipt muss enthalten:

- `schema: verified_sem_manifest_v1`, `status: verified`;
- tatsächliche ganzzahlige `verified_train_originals`, passende `manifest_sha256`;
- `corpus_version`, `content_snapshot`, `curation_record`, `leakage_review_record` als nachvollziehbare Referenzen;
- bestätigte Ausschlüsse `nffa_A_B_val_test_and_groups`, `all_aachen`, `nasa_benchmarks_if_evaluated`;
- `sources` mit Regeln je Quelle. `orientation_invariant` ist ausschließlich ein
  Boolean und bei fehlender Freigabe `false`.

Die Ausschlüsse sind eine explizite Bestätigung der vorgelagerten Prüfung, kein
vom Loader selbst bewiesener Ausschluss. Die bestehenden zulässigen 8.061
NFFA-Bilder können später Teil des größeren Manifests sein. Aachen bleibt ganz
außen vor, NFFA Validation/Test und ihre Gruppen über beide Splits ebenfalls.
Es gibt keine automatische Quellengewichtung. `content_snapshot` muss einen
unveränderlichen, extern verifizierten Korpusstand bezeichnen. Änderungen der
Bildbytes an denselben Pfaden erkennt dieser bewusst hashfreie Bildloader nicht.

## Modell, Originalkomponenten und Update-Semantik

Unveränderter externer Meta-Checkout:
`/path/to/workspace/external/dinov2`, Commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
Der Import prüft weiterhin Commit und sauberen Checkout.

Originaler Meta `dinov2_vitl14`, keine Register, 24 Blöcke, Patch14, Feature1024.
Student lädt einen ausdrücklich angegebenen originalen Backbone mit vertrauenswürdig
hinterlegtem SHA256 und strikter Tensorprüfung. Kein stiller Download und kein
Start aus dem alten Small- oder 5.000-Schritte-Checkpoint. Die frischen originalen
Meta-DINOHeads sind geteilt für DINO/iBOT: 65.536 Prototypen, 3 Schichten,
Hidden2048, Bottleneck256. Teacher erhält eine vollständige Student-Kopie mit
getrennten Parametern, bleibt ohne Gradienten und in `eval()`.

Die Mathematik von `DINOLoss`, `iBOTPatchLoss`, `KoLeoLoss` und `DINOHead` bleibt
im originalen Meta-Code. Gewichte1/1/0,1; Student-Temperatur0,1,
Center-Momentum0,9. Teacher-Temperatur0,04→0,07 über20 Präsentationsdurchläufe;
EMA0,999→1 über den vollen Horizont. Alle Backbone-Blöcke trainierbar,
DropPath0,1 nur beim Student. Teacher bleibt wie bisher in eval.

Pro Optimizer-Update werden4 Microbatches à32 oder8 à16 verarbeitet:

1. Alle Microbatches verwenden dieselben Teacher-Gewichte, Temperatur und alten
   Center. Jeder Microbatch-Loss wird durch die Accumulation-Anzahl geteilt.
2. FP32-Summen und Anzahlen der Teacher-Logits werden gesammelt, ohne große
   Logit-Tensoren über Microbatches hinweg zu behalten. Der iBOT-Center ist über
   alle tatsächlich maskierten Patches gewichtet, nicht über Batchmittelwerte.
3. Gradienten getrennt nach Backbone und Head auf Norm3 begrenzen. NaN/Inf
   beendet den Lauf, bevor ein ungültiges Optimizer-Update gespeichert wird.
4. Ein AdamW-Schritt, eine Center-Aktualisierung durch die originalen Meta-
   `update_center`-/`apply_center_update`-Methoden und ein Teacher-EMA-Schritt.
5. Updatezähler und sämtliche Schedules genau einmal weiterzählen.

Die Gewichte des letzten Head-Ausgabelayers bleiben im ersten Durchlauf über
LR0 eingefroren, wie bei Metas Last-Layer-Schedule. Seine Gradienten bleiben
bei der Head-Normbegrenzung enthalten und AdamW-Momente werden weitergeführt.
Damit sind auch diese Optimizer-Zustände beim Benchmark angelegt. Andere
Head-Schichten lernen bereits.
KoLeo wird unverändert innerhalb der **physischen** Crop-Batches berechnet.
32×4 und16×8 sind deshalb wissenschaftlich unterschiedliche physische Rezepte,
obwohl beide effektiv128 Originalbilder pro Update verwenden.

Die LR-Werte sind absolut, ohne zusätzliche Batchskalierung. Backbone-Peak1e-5,
Minimum1e-7; Head-Peak1e-4, Minimum1e-6. Warmup5 Durchläufe, danach Cosine mit
100-Durchläufe-Horizont. Benachbarte Transformerblöcke: Faktor0,9; tiefster
Block und abschließende Norm erreichen den Backbone-Peak. Metas Gruppenhelfer
zählt die Norm als zusätzliche Ebene: für Transformerblöcke wird der Exponent
um eine Ebene verschoben, ebenso für die Embeddings. Die abschließende Norm
bleibt bei Faktor1; das Patch-Embedding erhält zusätzlich0,2. AdamW-Betas0,9/0,999, WD0,04→0,2;
Bias/Norm-Ausnahmen aus Metas Gruppenhelfer bleiben erhalten.

BF16-Autocast ausschließlich für Backbone/Head. Parameter, AdamW-Zustände,
Loss-Eingänge, Softmax und Center bleiben FP32. Native PyTorch SDPA;
`XFORMERS_DISABLED=1` muss vor Meta-Import gesetzt sein. Nicht-reentrantes
Activation Checkpointing nur der Student-Blöcke mit Erhalt des RNG-Zustands;
keine Umbenennung von State-Dict-Schlüsseln, kein CPU-Offload. Fixer Seed42,
deterministische CUDA-Algorithmen aus, TF32 aus. Keine Behauptung vollständiger
Reproduzierbarkeit über andere Software-/GPU-Versionen.

## SEM-Ansichten und bewusste Abweichungen

Die bestehende Metadatenstreifenentfernung erfolgt vor den Crops.
Neue domänenspezifische SEM-Augmentierung statt Meta `DataAugmentationDINO`:

- Zwei globale518² und acht lokale224² Ansichten.
- Gleichverteilte Cropfläche relativ zu `min(W,H)^2`: global0,5–1,
  lokal0,15–0,5; Seitenlänge gerundet auf Pixel, zufällige gültige Position.
  Quadratische Crops und isotropes bicubisches Resize, keine Rechteckverzerrung.
- Nur nach Quellenfreigabe: H-/V-Flip jeweils0,5 und gleichverteilte Rotation
  0/90/180/270 Grad. Ohne Freigabe alle drei Operationen aus.
- Graustufen: mit Wahrscheinlichkeit0,8 Helligkeit und Kontrast jeweils
  gleichverteilt0,9–1,1, zufällige Reihenfolge; anschließend Kanalreplikation
  und vorhandene ImageNet-Normalisierung.
- Kein Blur, Rauschen, Solarisation, Inversion, Hue/Saturation oder Channel Dropout.
- Originale Meta-Maskierung/Collate: 50% der globalen Student-Ansichten, Ratio0,1–0,5.
  Globale37×37=1.369 und lokale16×16=256 Patch-Tokens.

Es werden niemals zufällige Ansichten vorab fixiert/gecacht. Kein vollständiger
RAM-Cache; initial16 Worker, `spawn`, persistent, Prefetch2, Pin Memory und
nicht blockierende Transfers. Die Prozess-Threads werden auf1 begrenzt.

Abweichungen vom offiziellen verteilten Training: ein GPU-Adapter statt
FSDP/nested xFormers, freigegebene SEM-Augmentierung, getrennte absolute LRs,
effektive Batch128 per Accumulation mit updateweiser Zentrierung, EMA0,999,
DropPath0,1, präsentationsbasierte Schedules, fortsetzbarer Manifeststrom und
zusätzliche Integritäts-/Featurediagnostik. Keine Loss-Neuimplementierung.

## Budget, Datenstrom und Resume

Für die tatsächliche freigegebene Bildzahl `N` gilt:
`step_for_pass(p) = ceil(p*N/128)`.

Bei **nur beispielhaften**200.000 Bildern:

| Parameter | Updates |
| --- | ---: |
| Voller100-Pass-Horizont |156250|
| Erster Stop nach10 Pässen |15625|
| Checkpoints0/1/5/10/25/50/100 |0,1563,7813,15625,39063,78125,156250|
| LR-Warmup5 Pässe |7813|
| Letzter Headlayer eingefroren1 Pass |1563|
| Teacher-Temperatur-Warmup20 Pässe |31250|

Durch ceil wird ein Präsentationsbudget gegebenenfalls um weniger als128 Bilder
überschritten. Der Sampler mischt jede vollständige Manifestpermutation neu und
behält Epochentails; Batches können die Permutationsgrenze überschreiten.
Prefetch verändert den gespeicherten Verbrauchscursor nicht. Draw-abhängige
CPU-RNGs machen Bildansichten und Maskierung von Worker-Timing unabhängig.

Der10-Pass-Checkpoint liegt mitten im100-Pass-Schedule. Fortsetzung verändert
nur `run_until_step`, niemals Horizont, Bildzahl, Manifest, Batch oder Rezept.
Der Vertrag enthält Konfiguration, Datenreceipt, Originalgewichte, Upstream-Pin,
Code und Software. Rezeptänderung = neuer Lauf. Die Workerzahl sollte vor dem
Lauf festgelegt bleiben; der Vertrag ist hier bewusst streng.

Volle Zustände und Teacher-Backbone an jedem Meilenstein, Recovery alle1.000
Optimizer-Updates, letzte2 Recovery-Snapshots behalten. Gespeichert werden
Student/Teacher samt Head, beide Center, AdamW, vollständige Schedules/Zähler,
RNGs, verbrauchter Samplercursor und aufgelöster Vertrag. SIGINT/SIGTERM stoppt
an der nächsten vollständigen Updategrenze. Ein Fehler mitten im Update
verwendet beim Resume ausschließlich den letzten vollständigen Snapshot.

Exports werden mit SHA256, Tensorvergleich und FP32-Features gegen den lebenden
Teacher geprüft. Schritt0 erhält die Originaldatei unverändert und wird auf
Original-Tensorgleichheit geprüft. Bestehende Snapshots werden nie überschrieben.

## Befehle

Jetzt ausführbar, **ohne Datensatz oder GPU**:

```bash
cd /path/to/nffa-sem-dinov2
conda activate sem-gpu
python scripts/train_sem_cellstyle.py --stage plan --example-originals 200000
```

Sobald der Korpus fertig ist: neue Arbeitskopie der YAML anlegen und
`dataset.manifest`, `dataset.receipt`, `initialization.checkpoint` und
`initialization.sha256` eintragen. Keine hypothetische Bildzahl hardcodieren.
Beispielname für die aufgelöste YAML: `configs/sem_ssl/sem_cellstyle_vitl14_ready.yaml`.

Ansichten prüfen; Batch32 ist hier lediglich eine Formkonfiguration:

```bash
python scripts/train_sem_cellstyle.py --config configs/sem_ssl/sem_cellstyle_vitl14_ready.yaml \
  --stage preview --physical-batch 32 --output outputs/sem_ssl_previews/cellstyle_v1
```

Auf `gpu-host`, **erst nach Datensatzfreigabe**, vollständiger neuer Large-Benchmark:

```bash
cd /path/to/nffa-sem-dinov2
conda activate sem-gpu
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/benchmark_sem_cellstyle_a100.py \
  --config configs/sem_ssl/sem_cellstyle_vitl14_ready.yaml \
  --batch-sizes 16 32 --warmup-updates 10 --measured-updates 30 --num-workers 16 \
  --output outputs/sem_ssl_benchmarks/cellstyle_vitl14_v1
```

Jeder Kandidat beginnt mit Originalgewichten und frischen Zuständen in einem
separaten Prozess. Vollständige Multicrop-Daten, DINO/iBOT/KoLeo, Backward,
AdamW-Zustände, Accumulation, Center und EMA sind enthalten. CUDA synchronisiert
an Messgrenzen; die regulären endlichen Loss-/Gradientenprüfungen können intern
zusätzlich Host-Wartezeit verursachen und bleiben Teil der tatsächlichen
Trainingskosten. Erfasst werden End-to-end-Bilder/s, median/mean/p95 Updatezeit,
GPU-Streamzeit, Loader-Wartezeit, Peak allocated/reserved und VRAM-Gesamtgröße.
Es werden echte Optimizer-Updates gemessen, keine Encoder-Inferenz.

Separat nach der Messung: vollständiges Checkpoint-I/O, Exportprüfung und
Wiederholung desselben Updates nach Restore; Modell, Optimizer, Loss und Draws
müssen übereinstimmen. Temporäre Benchmarkzustände werden entfernt, auch bei
Kandidatenfehlern. Benchmark-Gewichte werden nie zum Training übernommen.

Dateien: `benchmark.csv`, `benchmark.json`, `throughput_vs_batch.png`,
`memory_vs_batch.png` und einzelne `candidate_16.json`/`candidate_32.json`.
OOM wird protokolliert; Prozessende gibt GPU-Speicher vor dem nächsten Kandidaten
frei. Empfehlung nur bei endlichen Updates, höchstens90% reserviertem VRAM;
innerhalb5% des besten Durchsatzes wird geringerer Speicherverbrauch bevorzugt.
Wenn beide scheitern, bleibt die Auswahl offen. Keine erfundene Large-Laufzeit:
Schätzung aus gemessenen Originalbildern/s, Checkpoints/Evaluation kommen hinzu.

Nach Prüfung der Resultate die gemessene physische Batchgröße in der YAML
fixieren. Beispiel **nur falls32 erfolgreich gemessen wurde**:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/train_sem_cellstyle.py --config configs/sem_ssl/sem_cellstyle_vitl14_ready.yaml \
  --stage train --physical-batch 32 \
  --benchmark-result outputs/sem_ssl_benchmarks/cellstyle_vitl14_v1/benchmark.json
```

Dies startet später ausdrücklich die erste10-Pass-Stufe; der aktuelle
Implementierungsauftrag startet sie nicht. Für Resume denselben Befehl mit
`--resume`, für eine spätere Stufe zusätzlich `--run-until-step <ceil(p*N/128)>`.
Die Batchwahl ist an Benchmark, Manifest, Initialisierung und Rezept gebunden;
fehlende Resume-/Exportprüfung verhindert den wissenschaftlichen Start.

## Validation und Monitoring

Original-Large ist die Hauptreferenz bei Pass0. Historisches Small bleibt im
alten Bericht separat beschriftet. Keine Wiederverwendung von Small-Features.
Encodername, SHA und Featuredimension gehören zur Cacheidentität. Die
bestehenden Heads werden je Encoder neu aus Seed42 trainiert: NFFA A/B LogReg
und MLP512/256/128 mit ursprünglichem Dropout/Hyperparametern; Aachen A3
896×672 mit64×48 Gitter und ursprünglichem Linear-/CNN8-Decoder. Die einzige
Head-Architekturänderung ist die erforderliche Eingangsdimension1024.

Quick bei1/5 Pässen: LogReg A/B und Aachen Linear. Voll bei0/10/25/50/100:
alle6 Probes. Originalmasken, MA-IoU/mIoU, Precision/Recall, Inselmetriken,
feste0,5- und separat validationgewählte Schwellen bleiben erhalten.
Keine Testauswertung oder Test-Modellauswahl.

Commands zunächst nur anzeigen:

```bash
python scripts/evaluate_sem_cellstyle.py --config configs/sem_ssl/sem_cellstyle_vitl14_ready.yaml \
  --passes 0 1 5 10 --verified-data-root <bereits-verifizierter-Encoder-Datenordner>
```

Mit `--execute` ausdrücklich ausführen; der bestehende Runner verwendet8 NFFA-
und16 Aachen-Worker. Bereits verifizierte Datenmetadaten können übernommen
werden, aber Featurecaches nur für exakt denselben Encoder und Preprocessing.
Ohne `--verified-data-root` bleibt das bisherige strenge Preflight-Verhalten.
`--report-only` erstellt einen gemeinsamen Large-Vergleich ohne Evaluation.
Spätere Meilensteine ohne Artefakte bleiben als fehlend sichtbar; bei1/5 werden
MLP/CNN8 ausdrücklich als nicht vorgesehen markiert.

`metrics.jsonl`: einzelne Lossbeiträge, Teacher-/Student-Entropie,
Featurevarianz, separate Gradientennormen von Backbone und Head, LR jeder Gruppe,
Bilder/s, Peak-VRAM und verbrauchte Originalpräsentationen. Review nach10/25/50/100
Pässen anhand Validation und Featurediagnostik. SSL-Loss allein wählt keinen
Encoder. Zunächst ein Seed; überzeugende Befunde später mit weiteren Seeds prüfen.

## CPU-Prüfung und Grenzen

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XFORMERS_DISABLED=1 \
python -m pytest -q tests/test_sem_cellstyle.py
python -m pytest -q
```

CPU-Fixtures verkleinern ausschließlich für Tests Dimension, Tiefe, Crops und
Batch. Sie verwenden originale Meta-Heads/-Losses und prüfen Teacherkopie,
Center-Aggregation, einmaliges EMA/Optimizer-Update, Activation-Checkpointing-
Gradienten, bitgenauen Resume, Exporte, Budgetberechnung und Large-Head-Dimensionen.
Ein Erfolg dieser Tests ist keine Bestätigung, dass ViT-L mit Batch16/32 und
65.536 Prototypen in80GB passt. Diese offene Frage beantwortet erst der reale
Large-Benchmark. Es liegt noch kein neuer wissenschaftlicher Messwert vor.

Prüfnachweis vom18.11.2026: vollständige CPU-Suite **245 bestanden,3
übersprungen,3 Subtests bestanden**. Protokoll unter
pytest_full.txt.
Der datensatzfreie Plan ist als
200.000-Bilder-Beispiel
gespeichert; diese Zahl ist ausdrücklich keine Freigabe eines existierenden Korpus.

Die19 neuen Rezepturtests wurden danach nochmals separat im finalen Stand
ausgeführt: pytest_cellstyle.txt.
