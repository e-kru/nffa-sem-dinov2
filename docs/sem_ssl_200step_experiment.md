# Erstes SEM-DINOv2-Post-Training: 200 Optimizer-Schritte

Diese Pipeline bereitet Phase A vor: ein eigenständiges, begrenztes Post-Training
auf NFFA-SEM und anschließend explizite Frozen-Encoder-Validierung. Die folgenden
GPU-Befehle sind für den Benutzer auf `gpu-host`; Codex startet sie nicht.
Kein Test-Split wird ausgewertet. Es werden keine Pakete installiert und keine
neue Umgebung erstellt.

Vor der Wahl einer größeren physischen Batchgröße kann der separate
[A100-Batch-/Durchsatzbenchmark](sem_ssl_a100_benchmark.md) ausgeführt werden.
Er misst den vollständigen ursprünglichen SSL-Update-Pfad auf realen NFFA-
Trainingsbildern, schreibt ausschließlich eigene Benchmark-Artefakte und startet
keinen Experiment-Lauf.

## Dateien und Ablauf

| Datei | Aufgabe |
| --- | --- |
| `configs/sem_ssl/nffa_200step.yaml` | Versionierte Phase-A-Rezeptur, Datenquelle, geplanter Schedule und aktueller Stoppunkt. |
| `scripts/train_sem_ssl.py` | CLI: Daten-Preflight oder Training; prüft Initialisierungs-SHA und bei CUDA genau eine A100 80 GB. |
| `src/ssl/upstream.py` / `dinov2-upstream.json` | Aktivieren ausschließlich den sauberen, externen Meta-Checkout beim gepinnten Commit. |
| `src/ssl/adapter.py` | Vorhandener Student/Teacher-Adapter: originale Backbones, Head, drei Losses, Optimizer-Gruppen, per-Modul-Clipping, EMA; optional zusätzliche Diagnosewerte. |
| `src/ssl/data.py` | Vorhandene konservative Train-Auswahl, Metadata-Strip-Crop, RGB-Konvertierung; originale Meta-Augmentation und Mask-Collate. |
| `src/ssl/experiment.py` | Validiert Phase A; erzeugt originale Meta-CosineScheduler mit Schritthorizont; erstellt Daten-Inventar und wissenschaftlichen Resume-Vertrag. |
| `src/ssl/stream.py` | Deterministische Epoch-Permutationen, Image-Presentation-IDs und lokale RNG-Kontexte für Bilder und Masken. |
| `src/ssl/trainer.py` | Führt genau die angeforderte Zahl Updates aus; JSONL-Monitoring, Signalbehandlung, Meilensteine und Recovery. Ruft keine Downstream-Evaluation auf. |
| `src/ssl/checkpoints.py` | Atomare Checkpoint-Verzeichnisse, SHA-Prüfung, Auswahl des letzten gültigen Snapshots, RNG-Restore und begrenzte Recovery-Aufbewahrung. |
| `scripts/verify_sem_ssl.py` | Prüft vollständige Meilensteine und Teacher-Exports, lädt originale Meta-Backbones strikt und vergleicht deren Features. |
| `scripts/evaluate_sem_ssl.py` | Expliziter sequenzieller Aufruf des vorhandenen `evaluate_sem_encoder.py` für die vier Teacher-Encoder. |
| `src/ssl/reporting.py` / `scripts/report_sem_ssl.py` | Lesen vorhandene Validation-Ergebnisse, prüfen gemeinsame Protokolle und schreiben Vergleichstabelle/Plots. |
| `tests/test_ssl_experiment.py` | Synthetische CPU-Integration, Schritte/Resume/RNG/Meilensteine/Exports/Fehlerfälle. |
| `tests/test_ssl_pilot.py` | Bereits vorhandene Prüfung der originalen Losses und aller Student-Gradienten gegen Metas `SSLMetaArch`, einschließlich EMA. |

Der bisherige `scripts/pilot_dinov2.py` bleibt ein separater kurzer Pilot mit
unverändertem 20-Schritt-Limit und bisheriger epochbasierter Schedule-Semantik.
Die neue Pipeline erweitert ihn nicht durch Umgehen seines Limits.

## Exakte wissenschaftliche Konfiguration

Externes Repository: `/path/to/workspace/external/dinov2`,
Commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
Initialisierung:
`/path/to/workspace/cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`,
SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
Der externe Checkout bleibt unverändert. Der CLI-Start prüft den Datei-Hash;
der Adapter lädt jedes Backbone-Tensor strikt und verifiziert dessen Gleichheit.

* Originales Meta DINOv2 ViT-S/14: 12 Blöcke, Dimension 384, Patchgröße 14,
  keine Register, veröffentlichter positional embedding grid 37×37, originale
  Interpolation für globale/lokale Auflösungen, Student DropPath 0.3.
* Originaler gemeinsamer `DINOHead`: 3 Layer, Hidden 2048, Bottleneck 256,
  65.536 Prototypen, Student und Teacher mit zunächst identischen Gewichten.
* Originale `DINOLoss`, `iBOTPatchLoss`, `KoLeoLoss`; Gewichte 1, 1 und 0.1.
  Die mathematischen Losses werden ausschließlich aus Meta importiert.
* Teacher immer `eval()` und ohne Gradienten; Student `train()`.
  AdamW aktualisiert zuerst den Student, danach EMA **alle** Teacher-Parameter
  von Backbone und Head. Keine Teacher-Optimierung durch Backpropagation.
* Batch 64 **Originalbilder**, entsprechend 128 globale und 512 lokale Ansichten.
  Seed 42, standardmäßig 16 Loader-Worker. Genau ein A100-80-GB-Gerät.
  Prefetch 2, Pinning und nicht-blockierende Transfers aktiv; Persistent Workers,
  Bildcache und Komponenten-Profiling aus. Diese Einstellung stammt aus dem
  [vollständigen A100-Durchsatzvergleich](sem_ssl_loader_throughput.md).
* BF16-Autocast für Backbone/Head, FP32 für Losses und Center. Native originale
  PyTorch-Attention; `XFORMERS_DISABLED=1`. Kein GradScaler bei BF16.
* AdamW, Betas (0.9, 0.999), originales Layerwise-Decay 0.9,
  Patch-Embedding-LR-Multiplikator 0.2, originales Per-Modul-Gradient-Clipping 3.
* Base-LR 0.0002, originale Skalierung `sqrt(batch/1024)`: effektive maximale
  LR bei Batch 64 = **5e-5**. Warmup 20 Schritte ab 0, danach Cosine
  Richtung 1e-6. Letzter Head-Layer: LR 0 für die ersten 10 Schritte.
  Optimizer-States werden dabei wie im offiziellen Trainer weitergeführt.
* Weight Decay 0.04 → 0.1, Teacher-Momentum 0.996 → 1.0, jeweils originales
  CosineScheduler. Teacher-Temperatur 0.04 → 0.07 in den ersten 20 Schritten,
  anschließend konstant. Dies sind konservative Startwerte für ein Pilotexperiment,
  keine bereits anhand von Ergebnissen optimierten Hyperparameter.

Die zuvor vorbereitete Rezeptur mit Batch 8 wurde nach dem A100-Benchmark auf
Wunsch des Nutzers angepasst. Der Horizont bleibt bei 200 Updates: damit werden
jetzt 12.800 statt 1.600 Originalbildpräsentationen verarbeitet. Das ist ein neuer
Pilot mit größerem physischem Batch und entsprechend skalierter LR; bestehende
Batch-8-Checkpoints werden nicht unter der geänderten Konfiguration fortgesetzt.

Meta indexiert seinen Cosine-Schedule von 0 bis `total_steps-1`; der letzte
Wert liegt nahe dem Endwert, der Endwert selbst gilt ab Index `total_steps`.
Der erste Optimizer-Aufruf verwendet LR 0 und zählt trotzdem als Update.
Die Checkpoint-Nummer bezeichnet **abgeschlossene** Optimizer- und EMA-Schritte;
Checkpoint 50 enthält somit die Updates mit Schedule-Indizes 0…49.

### Unveränderte Augmentation und Maskierung

Direkter Import von Metas `DataAugmentationDINO`, ohne eigene Augmentation:

| Bestandteil | Einstellung |
| --- | --- |
| Globale Crops | 2 × 224×224, RandomResizedCrop/Bicubic, Scale (0.32, 1.0) |
| Lokale Crops | 8 × 98×98, RandomResizedCrop/Bicubic, Scale (0.05, 0.32) |
| Horizontaler Flip | p=0.5 |
| ColorJitter | brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1; RandomApply p=0.8 |
| RandomGrayscale | p=0.2 |
| GaussianBlur | Global 1: p=1.0; Global 2: p=0.1; lokal: p=0.5 |
| Solarization | Zweiter globaler Crop: Threshold 128, p=0.2 |
| Normalisierung | ImageNet Mean (0.485,0.456,0.406), Std (0.229,0.224,0.225) |
| iBOT | Originaler `MaskingGenerator` und `collate_data_and_cast`, Mask-Sample-Wahrscheinlichkeit 0.5, Ratio (0.1,0.5), globales Grid 16×16 |

Einzige SEM-Vorverarbeitung: vorhandenes `crop_metadata_strip` aus
`src/preprocessing_v2.py` und RGB-Konvertierung. Die bestehende Heuristik entfernt
den erkannten hellen Metadatenstreifen am unteren Rand. Sie ist kein allgemeiner
OCR-Filter für beliebige Beschriftungen an allen Rändern. Es wird kein zusätzlicher
Classification-Center-Crop vor die SSL-Augmentation geschaltet.

## Datensicherheit und Identität

Die Konfiguration liest das existierende V3-Manifest und die unveränderten
Split-Dateien A/B. Primär ist B; `train_records()` behält ausschließlich
eingeschlossene `crop_v2`-Bilder aus B-Train, deren **gesamte primäre Gruppe**
auch in A-Train erlaubt ist. Dies ist konservativer als eine einfache
Schnittmenge einzelner IDs. Klassenlabels werden nicht als SSL-Targets verwendet.

Der Preflight ermittelt die Anzahl neu. Das erwartete Ergebnis ist ungefähr
8.061; maßgeblich sind die tatsächlich ausgewählten IDs. Nur diese Bilddateien
werden geöffnet/gehasht. Split-Zuordnungen werden als Metadaten gelesen, aber
Validation-/Testbilder werden weder gehasht noch geladen. Der Start speichert:

* `selected_image_ids.txt`: exakte Reihenfolge aller erlaubten IDs;
* `run.json`: vollständige aufgelöste Konfiguration, Manifest-/Split-SHA256,
  ausgewählte Datensätze mit Bild-SHA256 und zusammengefasster Datenidentität;
* wissenschaftlicher Vertrag: Daten, initialer Checkpoint, Meta-Commit,
  Python/PyTorch/torchvision, Präzision und SHA256 der Trainingsquellen.

Preflight und Training hashen die ausgewählten Bilddateien jeweils erneut.
Das benötigt auf dem geteilten Dateisystem Zeit, schützt aber gegen stillen
Datentausch. Während eines Laufs muss der Datenbestand unverändert bleiben.
Die Provider-Grenze liefert Dataset, originale Collate-Funktion und Identität;
ein späteres versioniertes Multi-Source-Manifest kann dort ergänzt werden,
ohne den Optimizer-Loop neu zu schreiben. Aktuell ist ausschließlich NFFA V3
implementiert.

## Schedule, Stoppen und Resume

`experiment.schedule_total_steps` ist der ursprüngliche wissenschaftliche
Horizont. `experiment.run_until_step` bzw. `--run-until-step` beendet nur die
aktuelle Ausführung. Phase A setzt beide auf 200.

Für einen späteren 10.000-Schritt-Lauf wird **vor dem ersten Update** eine neue
Konfiguration mit Horizont 10.000 angelegt. Man kann diesen Lauf bei 2.000
stoppen und danach mit `--resume --run-until-step 10000` fortsetzen. Der Schedule
bleibt unverändert. Einen abgeschlossenen 200-Schritt-Cosine nachträglich auf
10.000 zu verlängern wird beim Resume abgewiesen und erfordert ein gesondertes,
explizit dokumentiertes Experiment.

Jeder volle Snapshot enthält Student und Teacher inklusive Head, beide Loss-Center,
AdamW-State, abgeschlossenen Schritt, aufgelöste Konfiguration, **vollständige
ursprüngliche Schedule-Arrays**, wissenschaftlichen Vertrag, CPU-/CUDA-/NumPy-/
Python-RNG und den Daten-Cursor. Meta hält Center-Updates zunächst außerhalb des
State-Dicts bereit. Vor dem Speichern werden sie mit Metas eigener Methode
angewandt; das ergibt denselben Center, den der nächste Forward sonst anwenden
würde. Ein Resume verliert dadurch kein ausstehendes Center-Update.

Ein Snapshot entsteht zunächst in `.pending_*`. Nach Dateisynchronisation und
Checksummen wird das Verzeichnis atomar in seinen finalen Namen umbenannt.
`complete.json` markiert einen vollständigen Snapshot. Beim Resume werden beide
Checkpoint-Bereiche durchsucht, Dateihashes und Payload geprüft und der höchste
gültige Schritt gewählt. Beschädigte/unvollständige neuere Snapshots werden
gemeldet und übersprungen. Ein gültiger, aber inkompatibler Vertrag wird **nicht**
übersprungen: Resume stoppt mit Fehler. Das vermeidet ein unbemerktes Zurücksetzen
auf eine andere Rezeptur. Ein Dateilock verhindert zwei gleichzeitige Schreiber.

SIGINT/SIGTERM fordert Stoppen an der nächsten abgeschlossenen Update-Grenze an.
Bei NaN/Inf, Backward-Fehler, OOM oder harter Terminierung wird kein teilweise
aktualisiertes Modell gespeichert. Dann bleibt der letzte gültige Snapshot
verfügbar. JSONL-Zeilen nach diesem Snapshot werden beim Resume zunächst unter
`metrics_before_resume_*.jsonl` archiviert und aus der aktiven Historie entfernt.
Bei einem Abbruch vor dem ersten vollständigen Schritt-0-Snapshot ist ein neuer
Ausgabeordner notwendig. Atomare Umbenennung/fsync ersetzen keine Garantie gegen
einen kompletten Ausfall des zugrunde liegenden NFS-Speichers.

### Zufallszustände und genaue Grenzen der Reproduzierbarkeit

Die Datenreihenfolge ist eine deterministische Permutation pro Daten-Epoche;
unvollständige Batches werden verworfen. Pro Originalbildpräsentation und pro
Batch-Maskierung werden stabile Seeds aus Seed 42 und dem Präsentations-Cursor
abgeleitet. Python-, NumPy- und Torch-CPU-RNG werden dafür lokal gesetzt und
anschließend wiederhergestellt. Metas Transformationen und ihre Verteilungen
bleiben unverändert; die Zuordnung der Zufallszahlen zu Bildern unterscheidet
sich vom offiziellen Worker-RNG-Stream. DropPath und Modell-RNG sind davon
unabhängig. Der Loader besitzt einen eigenen Generator.

Beim Resume wird aus dem **verbrauchten** Schritt fortgesetzt. Vorgefetchte Batches
können verworfen und identisch neu erzeugt werden. Daher werden weder Worker-
Zufallszustände noch Prefetch-Queues serialisiert. Die CPU-Tests vergleichen
ununterbrochenes und unterbrochenes Training einschließlich Losses, Modell,
Center und AdamW-State bitweise; außerdem prüfen sie die erzeugten Batches mit
unterschiedlichen Worker-Zahlen, soweit die Sandbox Multiprocessing erlaubt.

Dies verspricht keine bitweise identischen CUDA-Updates über Hardware-/Software-
Wechsel hinweg. BF16, parallele CUDA-Reduktionen und insbesondere der Backward
der positional embedding interpolation können nichtdeterministisch sein.
Standard `deterministic: warn` fordert deterministische Torch-Algorithmen an,
warnt bei verbleibenden Fällen und deaktiviert TF32/cuDNN-Benchmarking.
`strict` kann bei einer CUDA-Operation abbrechen; diese Einstellung gehört zum
Resume-Vertrag. Für vergleichbare Experimente Präzision, Software, Seed und
Worker-Zahl beibehalten. Eine GPU-Resume-Messung ist noch auszuführen.

## Checkpoints und Speicher

Beispiel: `outputs/sem_ssl/nffa_200step_seed42/`:

```text
run.json
selected_image_ids.txt
metrics.jsonl
status.json
latest.json
checkpoints/
  step_000000/{training.pth,teacher_backbone.pth,complete.json}
  step_000050/{training.pth,teacher_backbone.pth,complete.json}
  step_000100/{training.pth,teacher_backbone.pth,complete.json}
  step_000200/{training.pth,teacher_backbone.pth,complete.json}
recovery/
  step_…/{training.pth,teacher_backbone.pth,complete.json}
```

Meilensteine werden nie automatisch entfernt. Standardmäßig entsteht alle 25
Schritte ein Recovery-Snapshot; die zwei neuesten gültigen Recovery-Snapshots
bleiben erhalten. Der aktuelle Stoppunkt erhält ebenfalls einen Snapshot.
Bei Meilensteinen wird keine redundante Recovery-Kopie geschrieben. `.pending_*`
aus einem harten Abbruch werden nicht als gültig erkannt und nicht automatisch
gelöscht. Jeder Snapshot enthält einen rohen Teacher-State-Dict als separaten
Export für den vorhandenen Frozen-Encoder-Loader.

Die großen 65.536er Heads und AdamW-Zustände bestimmen einen wesentlichen Teil
des Speicherbedarfs. Vollständige Snapshots benötigen wesentlich mehr Platz als
ein Backbone-Export. Im vollständigen CPU-Komponententest wurden 444 MB bei
Schritt 0 und 800 MB nach Updates gemessen, jeweils inklusive Teacher-Export
(Backbone allein ca. 88 MB). Für vier Meilensteine, zwei Recovery-Snapshots und
einen temporären Save mindestens rund 6 GB frei vorsehen, zusätzlich zu den
wesentlich größeren Downstream-Feature-Caches. Zusätzlich wird für
atomare Saves vorübergehend ein weiterer Snapshot geschrieben. Checkpoint-Zeit
und erste Loader-Starts sind keine reine GPU-Compute-Zeit.

Der Nutzer hat inzwischen vollständige A100-SSL-Benchmarks mit realen NFFA-
Bildern, BF16, 224/98-Crops und vollen Heads ausgeführt. Batch 64 mit 16 Workern
erreichte 316,39 Bilder/s bei etwa 17,42 GiB allocated / 30,62 GiB reserved.
Die [Ergebnisse und Grenzen](sem_ssl_loader_throughput.md) sind dokumentiert;
der Benchmark ersetzt keine Checkpoint-/Resume-Prüfung des eigentlichen Piloten.
Bei OOM den Batch nicht im bestehenden Lauf ändern: separat dokumentierten
Speichertest/Experiment mit neuer Ausgabe und explizitem `--batch-size` anlegen;
dadurch ändert sich auch die effektive LR.

## Monitoring

`metrics.jsonl` erhält pro abgeschlossenem Update Gesamt-Loss, DINO global/lokal,
iBOT, gewichteten KoLeo-Beitrag, LR/letzte-Layer-LR, WD, Teacher-Momentum/-Temperatur,
Gradient-Norm vor Clipping pro Modul und insgesamt, Dauer einschließlich Laden,
Originalbilder/s, Peak CUDA allocated bytes, Schritt-/Präsentationszähler und
die verwendeten Record-/Draw-IDs. Speicherwerte sind allocated, nicht reserved
und nicht der gesamte GPU-Prozessspeicher. Der Diagnose-Transfer synchronisiert
am Schrittende; zusätzliche Synchronisation nur zum Timing wird vermieden.

Zusätzlich werden normierte CLS-Feature-Standardabweichung, mittlere paarweise
CLS-Cosine-Similarity und Teacher-Entropie protokolliert. `collapse_signal` markiert
Feature-Std <0.001 oder Cosine >0.99 als grobe, nicht kalibrierte Beobachtung.
Das ist keine neue Loss und kein automatischer Checkpoint-Selektor. Sinkende SSL-
Losses beweisen keine verbesserte Repräsentation. Qualität wird separat über
identisch trainierte Downstream-Heads auf Validation beurteilt.

## Unveränderte Downstream-Protokolle

`evaluate_sem_ssl.py` startet für jeden Schritt den bestehenden Runner mit
denselben Heads, Seed 42 und separater SHA256-basierter Encoder-Ausgabe:

* NFFA Splits A/B: originales Logistic Regression mit Train-only StandardScaler
  und ursprünglichem C-Grid; ursprüngliches MLP 384→512→256→128→10,
  BatchNorm/ReLU/Dropout 0.3 und unveränderten Trainings-/Selektionsparametern.
* Aachen A3: 896×672, ursprünglicher Linear-Head und CNN8, unveränderte Group-
  Splits und Evaluation in originaler Maskengeometrie. Feste Schwelle 0.5 und
  validationselektierte Schwelle werden getrennt berichtet.

Neue Encoder erhalten neue Feature-Caches und frisch trainierte Heads. Ein
identischer abgeschlossener Head kann mit `--resume` wiederverwendet werden,
ohne einen Head eines anderen Encoders als Initialisierung zu verwenden.
Bei `--stage all` führt der Wrapper zunächst den vorhandenen Preflight aus.
Für Schritt 0 kann danach die existierende vollständige Baseline verwendet
werden: Export-SHA, Daten-/Quellverträge, Head-Konfigurationen, Seed und Software
müssen passen. Der Wrapper verwendet dann den ursprünglichen Release-Pfad,
damit auch dessen pfadgebundener Segmentation-Cache identisch bleibt.

Die tatsächlichen vorhandenen Validation-Artefakte unter
`outputs/encoder_evaluation/v1/dinov2_vits14_b938bf1bc15cd2ec` enthalten:

| Head | NFFA A Macro-F1 | NFFA B Macro-F1 |
| --- | ---: | ---: |
| LogReg | 0.92274524 | 0.91428025 |
| MLP | 0.95501116 | 0.94649681 |

Aachen CNN8: validationselektiert MA-IoU 0.42351602, mIoU 0.67874443,
Schwelle 0.35. Linear: MA-IoU 0.34704426, mIoU 0.62995183, Schwelle 0.35.
Dies sind die vorhandenen reproduzierten Frozen-Baselines, keine hier neu
ausgeführten Benchmarks und keine Ergebnisse eines SEM-adaptierten Encoders.

Der neue Bericht liest diese JSON-Dateien direkt. Er vergleicht Daten-, Quell-
und Head-Protokolle über alle Encoder und stoppt bei Unterschieden. Fehlende
Schritte bleiben `null`/`—`. Enthalten sind beide Segmentation-Heads mit MA-IoU,
mIoU, Precision und Recall bei beiden Schwellenprotokollen. Downstream-Performance-
Plots und SSL-Loss-Plot sind getrennte Dateien.

## Befehle auf A100: Reihenfolge und getrennte Stufen

Die Befehle werden im Projektverzeichnis ausgeführt. `CUDA_VISIBLE_DEVICES=0`
an das zugewiesene physische Gerät anpassen. Kein automatischer SSH-Aufruf.

### 1. Bestehende Umgebung und CPU-Daten-Preflight

```bash
cd /path/to/nffa-sem-dinov2
source /path/to/workspace/apps/miniconda3/etc/profile.d/conda.sh
conda activate sem-gpu
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XFORMERS_DISABLED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/train_sem_ssl.py --stage preflight \
  --output outputs/sem_ssl_checks/nffa_200step_preflight.json
```

Der Preflight allokiert keine GPU. Er prüft noch keine A100-CUDA-Kernel und
schreibt nicht in den späteren Trainingsordner. Ausgabedateien werden nicht
überschrieben; für eine neue Prüfung einen neuen Namen wählen.

### 2. Kurzer CUDA-Test mit realen NFFA-Bildern, Batch 64 und vollem Head

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sem_ssl.py --stage train \
  --output outputs/sem_ssl_checks/nffa_real_cuda_3step_seed42 \
  --run-until-step 3 2>&1 | tee nffa_real_cuda_3step_seed42.log
```

Erwartet: Schritte 1…3 mit endlichen Losses/Gradienten, BF16 auf genau einer
A100 80 GB, Schritt-0-Export und Recovery-Snapshot bei Schritt 3. Dieser Lauf
verwendet den geplanten 200-Schritt-Schedule; er ist kein abgeschlossener
Drei-Schritt-Cosine. Erst nach erfolgreichem Kurztest den eigentlichen Lauf starten.
Der Testordner und der wissenschaftliche 200-Schritt-Lauf bleiben getrennt.

### 3. Wissenschaftlicher 200-Schritt-Lauf

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python scripts/train_sem_ssl.py --stage train \
  --output outputs/sem_ssl/nffa_200step_seed42 \
  2>&1 | tee nffa_200step_seed42.log
```

### 4. Derselbe Lauf in screen

```bash
screen -S sem-dinov2-200
```

Innerhalb von screen Umgebung aus Stufe 1 aktivieren und den Befehl aus Stufe 3
ausführen. Mit `Ctrl-A D` abhängen; mit `screen -r sem-dinov2-200` zurückkehren.
Nicht zusätzlich denselben Lauf außerhalb von screen starten.

### 5. Monitoring / kontrolliertes Stoppen

```bash
tail -f nffa_200step_seed42.log
tail -n 3 outputs/sem_ssl/nffa_200step_seed42/metrics.jsonl
cat outputs/sem_ssl/nffa_200step_seed42/status.json
nvidia-smi
```

`status.json` entsteht beim Abschluss/Stoppen/Fehler; während des ersten Laufs
ist es zunächst noch nicht vorhanden. Im Trainings-Terminal `Ctrl-C` einmal
anfordern und die Checkpoint-Meldung abwarten. Ein SIGKILL erlaubt keine
abschließende Sicherung.

### 6. Resume nach Unterbrechung

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python scripts/train_sem_ssl.py --stage train \
  --output outputs/sem_ssl/nffa_200step_seed42 --resume --run-until-step 200 \
  2>&1 | tee -a nffa_200step_seed42.log
```

Der Start nennt den tatsächlich verwendeten Snapshot. Daten, Initialisierung,
Präzision, ursprünglicher Schedule und Trainingsquellen müssen gleich bleiben.
Kein `--batch-size`-Wechsel in einem begonnenen Experiment.

### 7. Vier Meilensteine einschließlich Exports prüfen (CPU)

```bash
python scripts/verify_sem_ssl.py --run outputs/sem_ssl/nffa_200step_seed42 \
  --steps 0 50 100 200 \
  --output outputs/sem_ssl_checks/nffa_200step_checkpoints_verified.json
```

### 8. NFFA Classification für alle vier Encoder

Zunächst Befehlsplan anzeigen, anschließend bewusst ausführen:

```bash
python scripts/evaluate_sem_ssl.py --run outputs/sem_ssl/nffa_200step_seed42 \
  --task classification --stage all --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_sem_ssl.py \
  --run outputs/sem_ssl/nffa_200step_seed42 --task classification --stage all
```

Der Wrapper unterstützt auch einzelne Stufen `preflight`, `extract`, `train`,
`evaluate` und `report`. Für alle Stufen außer `all` erfolgt kein impliziter
Preflight. Der vorhandene Runner verlangt einen bereits erfolgreichen Preflight
vor GPU-Arbeit.

### 9. Aachen Segmentation für alle vier Encoder

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_sem_ssl.py \
  --run outputs/sem_ssl/nffa_200step_seed42 --task segmentation --stage all
```

Diese GPU-Jobs laufen sequenziell. Classification und Segmentation ebenfalls
nacheinander auf dem zugewiesenen Gerät ausführen.

### 10. Abschließender Vergleichsbericht

```bash
python scripts/report_sem_ssl.py --run outputs/sem_ssl/nffa_200step_seed42 \
  --output outputs/sem_ssl_reports/nffa_200step_seed42/comparison_v1
```

Ausgaben: `comparison.json`, `comparison.md`, `downstream_vs_steps.png` und,
wenn SSL-Metriken vorliegen, getrennt `ssl_losses.png`. Ein vorhandener Report-
Ordner wird nicht überschrieben. Ein Bericht vor dem Training kann die
vorhandene Original-Baseline bereits zeigen; spätere Ergebnisse bleiben leer.

## Abweichungen vom offiziellen verteilten Trainer

1. Ein Gerät ohne FSDP/NCCL/DDP und ohne xFormers-Nested-Tensor-Packaging;
   originale PyTorch-Backbone-Aufrufe für globale und lokale Crops getrennt.
   Der volle offizielle `SSLMetaArch` war im vorhandenen Probe wegen seiner
   xFormers-Anforderung nicht ausführbar. Die Adapter-Parität ist auf CPU gegen
   Metas Forward/Backward einschließlich aller Student-Gradienten geprüft.
2. Fortsetzung vom veröffentlichten Backbone, frische originale SSL-Heads und
   initiale Center; der veröffentlichte Backbone enthält keine trainierten
   SSL-Head-/Center-/Optimizer-Zustände. Teacher übernimmt Student vollständig.
3. Konservative LR/EMA/WD und kurze explizite Schritt-Schedules statt Metas
   großem ImageNet-Pretraining-Budget. Originaler Scheduler und Skalierungsregel.
4. NFFA-Train-Auswahl mit konservativem Schutz beider Benchmark-Splits sowie
   vorhandenes Metadata-Strip-Crop/RGB vor unveränderten Meta-Transformationen.
5. Deterministische Seeds pro Bildpräsentation/Mask-Batch anstelle des laufenden
   verteilten Worker-RNG-Streams; Checkpoints/Resume/JSONL auf einem Gerät.
6. BF16 für Backbone/Head, explizites FP32 für ursprüngliche Losses/Center,
   per-Modul-Clipping entsprechend dem vorhandenen Adapter. Keine Loss-Kopie.
7. `ibot_loss` und `koleo_loss` im JSONL sind die tatsächlichen gewichteten
   Beiträge zur Backpropagation. Metas ursprüngliche Log-Ausgabe halbiert diese
   zwei Beiträge; die Gradienten-Paritätsprüfung berücksichtigt diesen Faktor.
8. Teure Frozen-Encoder-Evaluation ist eine explizite unabhängige Stufe.

## Validierung und noch offene Hardwareprüfung

Abschließender Testlauf in `sem-gpu`: **159 passed, 3 skipped, 3 subtests
passed**, `python -m pytest -q` (224.80 s). Alle drei ausgelassenen Tests betreffen
die hier gesperrten IPC-Sockets für reale Torch-Multiprocessing-Worker.
`tests/test_ssl_experiment.py` ergänzt 14 Testfälle; der neue Worker-Test gehört
zu den drei explizit ausgelassenen Fällen. Die übrigen neuen Tests bestehen,
einschließlich bitweisem CPU-Resume und den 200 kleinen synthetischen Updates.

CPU-Tests, reale Train-Auswahl und die vorhandenen Baseline-JSONs werden bei der
Implementierung geprüft. Die synthetische CPU-Testarchitektur reduziert Tiefe,
Embedding- und Head-Dimension ausschließlich als Testfixture; sie wird niemals
vom Produktions-CLI für Phase A angeboten. Die Tests führen auch 200 kleine
synthetische Updates aus, um die exakte Meilenstein-/Aufbewahrungslogik zu prüfen.

Ein zusätzlicher CPU-Integrationstest verwendet das vollständige Produktionsmodell
mit originalen 65.536er Heads, 224/98-Crops und acht lokalen Ansichten: zwei
synthetische Updates, Batch 2/FP32, originale Release-Initialisierung. Beide
geprüften Exports (Schritt 0 und 2) lieferten nach strengem Reload exakt identische
CLS-/Patch-Features (maximaler absoluter Fehler 0). Die reale Train-Auswahl ergab
8.061 Bilder; Record-Inventar-SHA256
`5b3b07f1ea79a26c426ef6a7d913132c93802399369daf33b12195a2e5931c03`.
Nachweise liegen unter
lokales Prüfprotokoll.
Der vollständige Trainingsbild-Hashbestand liegt in
`outputs/sem_ssl_checks/phase_a_preflight_20261111.json`; der vorbereitete
Baseline-Vergleich in `outputs/sem_ssl_checks/phase_a_baseline_report`.
Die Sandbox verbietet den Unix-Socket für Torch-Tensortransfer zwischen Workern;
der reale Mehrworker-Test ist hier deshalb ausdrücklich ausgelassen und bleibt
für A100 offen. Ein gesonderter Dateisystem-Probe hat Lock, atomare Umbenennung
und Verzeichnis-fsync im gemeinsamen Datenverzeichnis erfolgreich geprüft.
Die optionale Meta-Warnung zu fehlendem `tifffile` betrifft dessen ungenutzten
TIFF-Decoder; dieser Dataset-Pfad lädt Bilder über PIL und benötigt dieses
zusätzliche Paket nicht.

Nicht ausgeführt: reales 200-Schritt-Training, neue Downstream-Head-Trainings,
Remote-Verbindungen oder A100-Tests durch Codex. Der Nutzer hat den separaten
Batch-64-/16-Worker-Durchsatztest inzwischen ausgeführt. Noch auf A100 zu prüfen
sind CUDA-Determinismuswarnungen, lange NFS-Checkpoint-Latenz und GPU-Resume.
Die dokumentierten Befehle bereiten
diese Prüfung vor; sie behaupten keine noch nicht gemessene Qualitätsverbesserung.
