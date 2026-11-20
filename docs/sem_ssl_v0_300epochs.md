# SEM Post-Training V0 mit ViT-S/14 und Aug-A

Der Einstieg prüft zuerst das eingefrorene Trainingsmanifest und seine Herkunft. Nach bestandener Prüfung folgen ein kurzer A100-Test mit den echten Daten, ein gemeinsamer Trainingsplan über 300 Epochen und die Downstream-Validierung an jedem Meilenstein. SSL und GPU-Validierung laufen nacheinander. Die nach einem Dekodierabbruch erstellte Revision verwendet `outputs/sem_ssl/sem_v0_vits14_aug_a_300epochs_decode_verified_seed42/`.

## Trainingsmanifest und Freigabe vom 20.11.2026

Die erste Revision `outputs/sem_corpus/sem_posttraining_v0_decode_trim20/` enthält **172.301** aktive Dateien aus 15 Quellen. Im vollständigen Manifest sind 20 zuvor aktive Zeilen auf `included=0` gesetzt und mit einem Ausschlussgrund versehen. Dazu gehören 3 BIL-Si-Graphite-TIFFs und 14 Drill-Cuttings-TIFFs über dem harten Pillow-Pixellimit sowie die 3 geprüften JLU-Stichproben, die beim bisherigen Laden vollständig weiß wurden. Die Originaldaten und der frühere V0-Export bleiben verfügbar. Die Auswahl ist mit `scripts/select_sem_v0_decode_revision.py` reproduzierbar. [Dateiliste und Ausschlussgründe](../outputs/sem_corpus/sem_posttraining_v0_decode_trim20/curation_decode_trim20_v0.json)

Die übrigen **4.109 JLU-Bilder** werden beim Laden auf die feste Skala 0–255 abgebildet (`round(uint16 / 257)`), bevor die RGB-Konvertierung, die Erkennung des Metadatenstreifens und die Aug-A-Crops stattfinden. Die Skala ist für alle Dateien gleich. Der Preflight prüft den Zugriff auf jede aktive Datei und dekodiert bis zu drei aktive Bilder je Quelle mit dem Trainingsloader.

- Alle 12.801 Bilder aus NFFA Split A Train sind enthalten. Split A hat null direkte oder gruppierte Überschneidungen nach der exakten A-Gruppierung. Nach den vorhandenen Ähnlichkeitsgruppen aus B betreffen mögliche Überschneidungen 342 A-Val- und 332 A-Testbilder.
- Split B hat 1.919 direkte Val- und 1.946 direkte Testüberschneidungen. Die vorhandenen B-Gruppen betreffen 2.013 Val- und 2.043 Testbilder.
- Aachen steuert 1.199 Train-Bilder bei. Die bekannten Aachen-Val-/Testgruppen überschneiden sich dabei nicht. Die darauf folgende Aachen-Validierung misst den Transfer innerhalb derselben Datendomäne auf zurückgehaltenen Gruppen.
- Der frühere SHA-Audit wurde abgebrochen. Der datensatzübergreifende Leakage-Abgleich ist unvollständig. Das V0-Inventar enthält 1.000 Bilder mit KIT-Val-Hinweis und 3.866 mit Drill-Cuttings-Test-Hinweis. Beide Quellen werden ausschließlich für SSL genutzt.

Die Freigabe gilt für diese Manifestrevision mit der dokumentierten Einschränkung des Leakage-Audits. Sie ist an die Export-Prüfsumme, die drei Split-Dateien und die bekannten Überschneidungszahlen gebunden. Dekodierfehler, direkte oder exakte A-Überschneidungen und Überschneidungen mit Aachen-Holdouts sperren weiterhin den Lauf. Nur NFFA A und Aachen werden auf der Validierung ausgewertet. [Preflight](../outputs/sem_ssl/sem_v0_vits14_aug_a_300epochs_seed42/preflight.json)

## Vollständige Pixelprüfung nach dem ersten Trainingsabbruch

Update 357 scheiterte an `jlu_cathodes/raw/BM10_SEM_BSE/BM10_SEM_BSE/slice-In-Beam BSE-00700.png`. Pillow liest dort den Header, aber der PNG-Datenstrom enthält einen fehlerhaften IDAT-CRC und scheitert beim Entpacken. Der alte Lauf bleibt mit seinem gültigen Checkpoint bei Schritt 0 erhalten. Die neue Standardrevision wird mit diesem CPU-Aufruf erzeugt.

```bash
bash scripts/prepare_sem_ssl_v0_decode_verified.sh 8
```

Der Audit dekodiert alle aktiven Bilder mit Pillow und wiederholt fehlerhafte Leseversuche. Bestätigte Bilddefekte erhalten im neuen Manifest den Grund `excluded_confirmed_pixel_decode_failure`. Ein wieder erfolgreicher Leseversuch oder ein Defekt in NFFA beziehungsweise Aachen stoppt die Publikation zur Prüfung. Umfang, Pfade und Fingerprint des Audits werden gespeichert. Der Einstieg berechnet daraus den neuen Epochencursor und verlangt einen zur Revision passenden A100-Kurztest. Der CPU-Aufruf selbst beendet sich nach dem Preflight.

## Festgelegte Trainingseinstellungen

ViT-S/14 startet vom gepinnten originalen Meta-Checkpoint. Student und EMA-Teacher erhalten wie im Aug-A-Pilot frische SSL-Heads mit Seed 42. Die Originalverluste DINO, iBOT und KoLeo sind aktiv. BF16, native PyTorch-Attention, AdamW und die bestehende EMA-/Center-Implementierung kommen aus dem vorhandenen Adapter.

- Aug-A liefert zwei globale 224×224-Crops mit Fläche 50–100 % und acht lokale 98×98-Crops mit Fläche 15–50 % des größten eingeschriebenen Quadrats. Es gelten Crop nach Metadatenstreifen, Graustufen, Helligkeit/Kontrast um 0,9–1,1 mit gemeinsamer Wahrscheinlichkeit 0,8, RGB-Replikation, ImageNet-Normalisierung und originales iBOT-Masking. Diese Rezeptur gilt auch für kleine Originale.
- Ein voller Update verarbeitet 32 + 32 Bilder. Jede Epoche ist eine neue deterministische Permutation aller Originale. Der Runner berechnet die letzte Batchgröße und die Zahl der Updates aus der aktiven Manifestrevision. Die beiden Microbatch-Losses werden wie im Piloten gleich gewichtet.
- Aus 5 %, 1 % und 20 % der Aug-A-Pilotschedule folgen **15 Epochen LR-Warmup**, **3 Epochen Freeze der letzten SSL-Head-Schicht** und **60 Epochen Temperatur-Warmup**. Der Preflight gibt Updates und Bildpräsentationen für den tatsächlich geprüften Export aus.
- Backbone-LR 1e-5 → 1e-7 und Head-LR 1e-4 → 1e-6 nach dem linearen Warmup ab null. Cosine Weight Decay 0,04 → 0,20 und EMA-Momentum 0,999 → 1,0 über alle 300 Epochen. Teacher-Temperatur 0,04 → 0,07. AdamW-Betas 0,9/0,999, Layer-Decay 0,9, Patch-LR-Multiplikator 0,2 und Gradient Clipping 3,0.

Meilensteine liegen bei Epochen 0, 1, 5, 10, 25, 50, 100, 200 und 300. Der Einstieg speichert bis zu zwei zusätzliche Recovery-Snapshots im Abstand von 1.000 Updates und beim kontrollierten Abbruch. Er stellt Student, Teacher, Optimizer, Center, RNG, Schedule und die exakte Position wieder her. Die Downstream-Artefakte abgeschlossener Meilensteine werden wiederverwendet. GPU-Wiederherstellung wird im kurzen Benchmark geprüft. Eine bitgleiche CPU-Reproduktion wurde im Test mit einem unvollständigen letzten Update verifiziert.

## Start, Kosten und Auswertung

```bash
CUDA_VISIBLE_DEVICES=2 bash /path/to/nffa-sem-dinov2/scripts/run_sem_ssl_v0_300epochs.sh
```

Der gleiche Aufruf setzt einen unterbrochenen Lauf fort. `--stage preflight` führt nur die Datensatz- und Ressourcenprüfung aus. `--stage benchmark` ergänzt den kurzen A100-Test. Für einen anderen Export akzeptiert der Einstieg `--manifest-dir PFAD` und ein separates `--leakage-review DATEI.json`. Das Review muss `manifest_sha256`, `cross_source_leakage_resolved: true`, die tatsächlich geprüften `reviewed_benchmarks` (mindestens `nffa_A` und `aachen_heerlen`) und eine beschreibende `evidence`-Referenz enthalten.

Die Speicherplanung reserviert 74,4 GiB für Meilensteine, Recovery, Caches, Metriken und temporäre Dateien. Der A100-Test misst nach 10 Warmup-Updates 30 Updates mit mehreren V0-Quellen und protokolliert Durchsatz, VRAM und Restore-Prüfung. Die Gesamtschätzung berücksichtigt Checkpoint-I/O und eine aus dem bisherigen ViT-S-Lauf abgeleitete Validierungsdauer. Die 253 Originale/s des früheren NFFA-Aug-A-Laufs ergäben als reine Rechenreferenz rund 57 Stunden Training für 51,7 Millionen Präsentationen. Der Durchsatz des größeren V0-Korpus wird beim A100-Test ermittelt.

NFFA verwendet Split A mit LogReg und Residual-MLP (8 Worker). Aachen A3 verwendet 896×672 mit Linear und CNN8 (16 Worker). Der Bericht trennt feste 0,5 und auf der Validierung bestimmte Schwellen und enthält MA-IoU, mIoU, Precision und Recall. Original ViT-S und ViT-L werden aus kompatiblen vorhandenen Resultaten übernommen. `report/comparison.csv`, `comparison.json` und `validation_by_epoch.png` werden nach jedem Meilenstein aktualisiert.
