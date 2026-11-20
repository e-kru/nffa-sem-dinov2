# A100: Batchgröße und Durchsatz des vollständigen SEM-SSL-Schritts

Für den anschließenden Worker-/Prefetch-/Cache-Vergleich bei Batch 32 und 64
siehe [DataLoader-Profiling und Bildcache](sem_ssl_loader_throughput.md).

Der eigenständige Benchmark ermittelt die schnellste praktikable physische
Batchgröße für das vorbereitete SEM-DINOv2-Experiment. Er trainiert ausschließlich
kurzlebige Modelle zum Messen. Er startet weder das 200-Schritt-Experiment noch
einen längeren Lauf, speichert keine Modell-Checkpoints und verändert keine
Trainingskonfiguration oder Python-Umgebung.

## Aufruf auf gpu-host

Diese Befehle manuell auf A100 ausführen. `0` durch den zugewiesenen physischen
GPU-Index ersetzen; alternativ einen einzelnen `GPU-…`-UUID angeben.

```bash
cd /path/to/nffa-sem-dinov2
source /path/to/workspace/apps/miniconda3/etc/profile.d/conda.sh
conda activate sem-gpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset CUDA_LAUNCH_BLOCKING

CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_sem_ssl_a100.py \
  --config configs/sem_ssl/nffa_200step.yaml \
  --batch-sizes 8 16 32 64 \
  --warmup-steps 10 --measured-steps 20 --num-workers 8 \
  --output outputs/sem_ssl_benchmarks/a100_batch_v1
```

`--output` muss ein **neuer** Ordner sein. Für Wiederholungen beispielsweise
`a100_batch_v2` verwenden. Bestehende Verzeichnisse werden abgewiesen; ebenso
Ausgaben innerhalb eines vorhandenen SSL-Laufs mit `run.json`.
`CUDA_LAUNCH_BLOCKING=1` würde die Messung verfälschen und wird abgewiesen.
Der Benchmark prüft echte NFFA-Trainingsbilder; es gibt keinen Synthetic-/CPU-
Fallback im produktiven CLI.

Optional denselben Aufruf mit `--try-128` und einem neuen Ausgabeordner ausführen.
128 wird nur getestet, wenn 64 erfolgreich und endlich war, dessen reservierter
VRAM höchstens 70 % des verfügbaren Budgets belegt, p95/Median ≤2 ist und der
Durchsatz mindestens 5 % über jeder kleineren erfolgreichen Batchgröße liegt.
Eine explizite `128` in `--batch-sizes` verwendet dieselbe Prüfung und erfordert
auch `64`. Übersprungene optionale Fälle werden mit Begründung protokolliert.
Andere physische Batchgrößen zwischen 2 und 128 sind konfigurierbar.

Standardwerte: Batches 8/16/32/64, zehn Warmup-Updates, zwanzig Mess-Updates,
acht Worker. Warmup ist auf 5–10 begrenzt; Messschritte und Worker sind über
`--measured-steps` und `--num-workers` einstellbar. Warmup plus Messung darf den
ursprünglichen Schedule-Horizont nicht überschreiten. Kurze Testfenster liefern
weniger belastbare Quantile als die voreingestellten zwanzig Schritte.

## Wiederverwendete Komponenten

| Datei | Verantwortung |
| --- | --- |
| `scripts/benchmark_sem_ssl_a100.py` | CLI, Audit, Provenienz und isolierte Worker-Aufrufe. |
| `src/ssl_benchmark.py` | Fenster-Timing, Subprozess-Steuerung, Fehlerbehandlung, Aggregation, Empfehlung, CSV/JSON/Plots. |
| `src/ssl/adapter.py` | Vorhandener `SingleGPUAdapter` und `train_step`; optional verzögerte Skalar-Rückgabe für den Benchmark. |
| `src/ssl/experiment.py` | Unveränderte Konfigurationsvalidierung, Leakage-sicheres `dataset_plan`, originale Schritt-Schedules. |
| `src/ssl/data.py`, `src/ssl/stream.py` | Vorhandene SEM-Vorverarbeitung, Meta-Augmentation/Maskierung und deterministische Bildpräsentationen. |
| `tests/test_ssl_benchmark.py` | CPU-Tests einschließlich reduzierter originaler Meta-Komponenten und echter CPU-Subprozesse. |

Es gibt keinen zweiten SSL-Algorithmus und keine kopierte mathematische Loss.
Verwendet werden die gleichen originalen Meta-Komponenten wie im
[200-Schritt-Experiment](sem_ssl_200step_experiment.md):

* Gepinnter Meta-Commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
* Originaler ViT-S/14 ohne Register, Dimension 384, kompletter originaler
  gemeinsamer DINOHead mit 65.536 Prototypen.
* Veröffentlichter Backbone mit SHA256
  `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
* Originale DINOLoss, iBOTPatchLoss und KoLeoLoss, Gewichte 1/1/0.1;
  BF16 für Backbone/Head, FP32 für Losses und Center, native PyTorch-Attention.
* Zwei globale 224×224- und acht lokale 98×98-Ansichten; dieselben Crop-Scales,
  Flip-, Jitter-, Grayscale-, Blur-, Solarization- und ImageNet-Einstellungen.
  Originales `DataAugmentationDINO` und originale Patch-Maskierung.
* Vorhandenes Metadata-Strip-Crop und RGB-Konvertierung, reale NFFA-V3-Bilder
  aus konservativ erlaubten Train-Gruppen **beider** Splits A/B.
* AdamW, Per-Modul-Gradient-Clipping, vollständiger Forward und Backward,
  Optimizer-Schritt, anschließend Teacher-EMA auf Backbone und Head.
* Dieselben Diagnosewerte wie der Experiment-Trainer: Loss-Beiträge,
  Gradienten-Normen, CLS-Feature-Std/-Cosine und Teacher-Entropie.

## Identischer Ausgangszustand pro Batchgröße

Der Elternprozess erzeugt keinen CUDA-Kontext. Er prüft den vorhandenen externen
Checkout, lädt dieselbe Konfiguration und verwendet `dataset_plan()` für den
Audit. Dabei werden einmal alle ausgewählten **Trainingsbilder** gehasht und
das exakte Inventar mit Manifest-/Split-SHA256 gespeichert. Der zuletzt geprüfte
Bestand enthielt 8.061 erlaubte Bilder; maßgeblich ist der jeweils neu berechnete
Audit. Validation-/Testbilder werden nicht geladen. Klassenlabels sind keine
SSL-Targets.

Jede Batchgröße läuft danach in einem eigenen, sequenziellen Python-Prozess.
Dieser prüft Konfiguration, Quellen, initialen Checkpoint, Inventar und erneut
die train-sichere Auswahl gegen die Manifest-/Split-Hashes. Die Bild-Hashes des
gesamten Bestands werden nicht für jeden Prozess erneut gelesen; der Bestand
muss während des Benchmarks unverändert bleiben. SHA256 und die exakten IDs
stehen für spätere Prüfung bereit. Das initiale Hashen erwärmt möglicherweise
den Dateisystem-Cache: gemessen wird der laufende Trainingsdurchsatz mit realem
Laden/Decodieren, keine standardisierte Cold-Cache-I/O-Leistung.

Jeder Prozess setzt Seed 42 neu und initialisiert den veröffentlichten Backbone,
dieselben frischen originalen Heads und initialen Loss-Center. Teacher und
Student sind vor dem ersten Update identisch; Teacher bleibt eingefroren und
im Eval-Modus. AdamW beginnt mit leerem State. Ein Tensor-Fingerprint des
vollständigen initialen Modellzustands wird über die Fälle verglichen.

Schedule-Indizes beginnen je Prozess wieder bei 0. Warmup führt echte Updates
aus; die Messung setzt unmittelbar anschließend mit den nächsten Indizes fort.
Der 200-Schritt-Schedule wird dafür nicht verkürzt oder neu definiert.
Die effektive LR folgt bei jeder physischen Batchgröße **derselben** ursprünglichen
Trainer-Regel `base_lr * sqrt(batch/1024)`, was im Ergebnis dokumentiert wird.
Alle übrigen Crop-, Loss-, EMA-, Temperatur- und Optimizer-Einstellungen bleiben
gleich. Ein Batchwechsel verändert das Präsentationsbudget pro Update und die
KoLeo-Nachbarschaft innerhalb des Batches. Die Empfehlung belegt daher weder
die beste Repräsentationsqualität noch die wissenschaftliche Gleichwertigkeit
verschiedener Batchgrößen oder Budgets.

Der Child prüft genau ein sichtbares CUDA-Gerät: A100 mit mindestens 75 GiB
gemeldeter Kapazität, BF16-Unterstützung und ohne MIG. Gerätename, UUID,
Gesamtkapazität und freier VRAM vor Modellinitialisierung werden gespeichert.
Ein Wechsel des GPU-UUID zwischen Fällen wird abgewiesen. Der Benchmark
beansprucht keine weiteren GPUs und führt kein DDP/FSDP aus.

## Messverfahren und Synchronisation

Warmup und Messung sind getrennte Fenster. In jedem Fenster werden CUDA-Events
gesetzt; **erst am Fensterende** erfolgt ein explizites
`torch.cuda.synchronize()`. Erst danach werden Event-Zeiten gelesen und die
gesammelten Skalar-Tensoren zur CPU übertragen. Es gibt kein explizites
Synchronisieren und keine Loss-/Norm-Skalar-Rückgabe zur CPU pro Schritt.
Initialisierung, Audit, Protokollierung und Plot-Erzeugung liegen außerhalb der
Messung. H2D, Forward, originale Losses, Backward, Clipping, AdamW, EMA und die
regulären Diagnoseberechnungen liegen innerhalb des gemessenen Schritts.

| Messwert | Definition |
| --- | --- |
| `originals_per_second` | `physical_batch * measured_steps / window_wall_seconds`. Einschließlich Loader-Warten und abschließendem GPU-Drain; die zehn Ansichten werden nicht als zehn Originalbilder gezählt. |
| `step_seconds` | CUDA-Zeit zwischen aufeinanderfolgenden Abschlussmarkern; enthält unüberlappte GPU-Leerzeiten durch Host/Loader. Bericht mit Mittelwert, Median und p95. |
| `train_step_seconds` | CUDA-Intervall vom Marker direkt nach Batch-Abholung bis nach vollständigem SSL-Update; umfasst auch H2D und nicht verdeckte Host-Dispatch-Lücken. |
| `dataloader_wait_seconds` | CPU-Wanduhrdauer von `next(iterator)` pro Schritt. Loader-Arbeit kann mit vorheriger GPU-Arbeit überlappen: diesen Wert nicht pauschal zu CUDA-Zeiten addieren. |
| `peak_allocated_bytes` | Höchster Torch-Allocated-Peak aus Warmup und Messung, einschließlich Modell/Optimizer. |
| `peak_reserved_bytes` | Höchster Torch-Caching-Allocator-Reserved-Peak aus beiden Fenstern. Kein Gesamtprozesswert aus nvidia-smi. |
| `warmup_peaks`, `measured_peaks` | Zusätzlich getrennte Peaks; Reset erst nach dem vollständig beendeten Warmup. |

Die CUDA-Abschlussintervalle und die Wanduhr des gesamten Fensters beantworten
verwandte, aber nicht identische Fragen. Die Durchsatzempfehlung verwendet die
Wanduhr, damit auch Datenbereitstellung und nicht verdeckte Host-Kosten zählen.
Andere Prozesse auf derselben GPU können die Messung beeinflussen; Gerät,
Software, Datenbestand und sonstige Auslastung für Wiederholungen konstant halten.

### Nichtendliche Werte und OOM

Im normalen Trainer bleibt die sofortige Prüfung von Loss und Gradienten
unverändert aktiv. Nur der Benchmark nutzt `train_step(..., diagnostics=True,
defer_metrics=True)`: Er erhält abgetrennte Skalar-Tensoren auf dem Gerät;
Clipping benutzt dieselbe mathematische Operation ohne synchronisierende
Fehlerabfrage pro Schritt. NaN/Inf in Losses, Gradienten-Normen oder Diagnosewerten
wird an der jeweiligen Fenstergrenze erkannt. Bei einem ungültigen Zustand
können daher bereits weitere Updates innerhalb des Fensters eingereiht sein.
Der gesamte Fall wird als `nonfinite` verworfen, niemals als stabil empfohlen
und niemals als trainiertes Modell gespeichert. Endliche Updates sind gegen
den normalen Pfad bitweise auf CPU geprüft.

CUDA-OOM ergibt `status: oom`, Phase, Fehlermeldung und verfügbare Memory-Peaks.
Andere Fehler werden als `error` protokolliert. Alle Worker werden geschlossen;
der Prozess gibt Referenzen und CUDA-Cache frei und beendet sich, bevor die
nächste Batchgröße startet. Die Prozessgrenze entfernt auch den CUDA-Kontext.
Ein unerwarteter Prozessabbruch ohne Resultat wird ebenfalls als Fehler
gespeichert. Bei SIGINT/SIGTERM beendet der Elternprozess die aktive Child-
Prozessgruppe einschließlich Loader-Workern und startet keine weiteren Fälle.
Vollständige bisherige Resultate stehen in `partial_results.json`.

Es werden keine historischen Checkpoints verändert. Die optionale Erweiterung
ändert den SHA256 der Adapter-Quelldatei; bestehende strikte Resume-Verträge
prüfen weiterhin ihren aufgezeichneten Code-Stand und werden nicht automatisch
migriert oder umgangen.

## Ergebnisformat

```text
outputs/sem_ssl_benchmarks/a100_batch_v1/
  config.yaml                 # aufgelöste Referenz-Konfiguration
  plan.json                   # Protokoll, Seed/Software/Quellen, Argumente
  dataset_inventory.json      # Auswahl und Trainingsbild-/Manifest-/Split-Hashes
  selected_image_ids.txt
  partial_results.json        # nach jedem vollständig beendeten Child aktualisiert
  batch_0008/
    worker.log
    result.json               # vollständige Fenster-/Schrittwerte und GPU-Metadaten
    steps.jsonl               # gemessene Loss-/Gradient-/Schedule-Werte
  batch_0016/…
  batch_0032/…
  batch_0064/…
  benchmark.csv
  benchmark.json
  throughput_vs_batch.png
  memory_vs_batch.png
```

`benchmark.json` enthält `plan`, `results`, `optional_128` und `recommendation`.
Je Fall stehen Status (`ok`, `oom`, `nonfinite`, `error`), Batchgröße,
Fensterdauer, Durchsatz, aggregierte Zeiten, Memory-Peaks, GPU-Informationen,
initialer Fingerprint und vollständige Schrittwerte bereit. In CSV sind Zeiten
als `step_mean_seconds`, `step_median_seconds`, `step_p95_seconds`,
`train_step_mean_seconds` und Loader-Mean/p95 abgeflacht. Speicher ist in Bytes;
der Plot verwendet GiB. Fehlende Messwerte gescheiterter Fälle bleiben leer
beziehungsweise fehlen im JSON; sie werden nicht als Null-Durchsatz erfunden.

Falls kein Fall erfolgreich war, entstehen dennoch Report und Plots mit
entsprechendem Hinweis. Es wird dann keine Batchgröße empfohlen und der CLI
endet mit Exit-Code 1. Ein erfolgreicher Benchmark mit tragfähiger Empfehlung
endet mit 0, ein kontrollierter Abbruch mit 130.

## Empfehlungsregel

Eine Konfiguration ist zunächst nur geeignet, wenn sie das gesamte Fenster mit
endlichen Losses/Gradienten beendet, p95/Median ≤2 ist und der Peak-Reserved-VRAM
höchstens 85 % von `min(total_vram, free_vram_before_model)` beträgt.
Die Speichergrenze ist über `--max-memory-fraction` konfigurierbar.

Unter diesen Fällen wird die **kleinste** Batchgröße innerhalb von 3 % des
besten gemessenen Originalbilder/s-Durchsatzes empfohlen. So entscheidet ein
geringer oder verrauschter Geschwindigkeitsgewinn nicht automatisch zugunsten
der größten Batchgröße. Ausschlüsse und deren Gründe stehen im Report.
Wenn kein Fall die Kriterien erfüllt, ist die Empfehlung leer. Die Reserve ist
für einen kurzen Benchmark bewusst konservativ; sie garantiert keine OOM-
Freiheit über Tausende zufällige Maskierungen und Bildpräsentationen.

Die tatsächliche Experiment-Konfiguration wird durch diese Empfehlung nicht
verändert. Der längere SSL-Lauf und seine LR-/Schedule-/Präsentationsbudgets
werden anschließend bewusst separat festgelegt.

## CPU-Validierung und offene A100-Messung

Die Tests prüfen Timing-Aggregation, Originalbildzählung, zusammenhängende
Warmup-/Mess-Schritte, Fenster-Synchronisation, NaN/Inf-Erkennung, OOM-Behandlung,
separate CPU-Prozesse, Fortsetzen nach fehlgeschlagenen Fällen, Prozessgruppen-
Abbruch, Memory-/Stabilitäts-Empfehlung, optionale 128er-Gates und CSV/JSON/Plots.
Ein kleiner CPU-Integrationstest benutzt originale Meta-Backbones, Heads und
Losses mit reduzierten Testdimensionen sowie synthetische PNGs. Er prüft den
vollständigen Update-Pfad und dessen Gleichheit zum normalen `train_step`.
Diese CPU-Zahlen sind keine A100-Benchmark-Ergebnisse.

Prüfstand vom 11.11.2026 in `sem-gpu`: **22 Benchmark-Testfälle bestanden**;
Gesamtsuite **181 passed, 3 skipped, 3 subtests passed** (200.36 s). Die drei
Skips stammen aus den bestehenden IPC-Socket-abhängigen Mehrworker-Tests.
Nachweise: lokales Prüfprotokoll.

```bash
python -m pytest -q tests/test_ssl_benchmark.py
python -m pytest -q
```

Codex hat den A100-Benchmark nicht ausgeführt, keine Remote-Verbindung aufgebaut
und keine Pakete installiert. CUDA-Timing, BF16-Verhalten, reale Acht-Worker-
Ausführung und die beste Batchgröße müssen mit dem obigen A100-Aufruf gemessen
werden. Bis dahin gibt es keine belastbare GPU-Durchsatzempfehlung.
