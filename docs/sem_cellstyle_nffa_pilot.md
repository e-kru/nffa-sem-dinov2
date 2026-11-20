# NFFA-Pilot der Large-Regularisierungsstudie

Stand: 18.11.2026. Vorbereitung auf dem gemeinsamen Dateisystem; GPU-Ausführung
erfolgt manuell auf `gpu-host`. Keine Änderung der Conda-Umgebung oder des
externen Meta-Checkouts.

## Ziel und Daten

Die vier Varianten A (SSL), B (SSL + L2-SP), C (SSL + Feature-Distillation) und
D (beides) können bereits auf dem freigegebenen NFFA-Trainingsanteil laufen.
Dieser Versuch ist ein eigener NFFA-Pilot, keine Durchführung der geplanten
Studie auf dem größeren gemischten SEM-Korpus.

**Ab 18.11.2026 ist Split A der Standard für neue NFFA-SSL-Läufe:**
**12.801 Trainingsbilder**, 2.742 Validierungsbilder und 2.743 Testbilder.
Die SSL-Auswahl enthält ausschließlich den Trainingsanteil von
`outputs/dataset_v3/splits/split_A_exact.csv`, ohne Schnittmenge mit Split B.
Die vorhandenen exakten Bildgruppen und das bereinigte `crop_v2`-Manifest bleiben
maßgeblich. Die A-Validierungs-/Testbilder sind vom SSL-Training ausgeschlossen.

Das neue Inventar `outputs/sem_ssl_inventories/nffa_split_a_v1.json` übernimmt die
bereits geprüften Bildhashes aus
`outputs/encoder_evaluation/nffa_5000step_workers_v1/dinov2_vits14_b938bf1bc15cd2ec/classification/split_A/preflight.json`.
`nffa_training_inventory` prüft Manifest, Split, Root und Trainingsgruppen gegen
diesen Nachweis; anschließend prüft der bestehende Inventar-Loader die Auswahl.
Es werden keine neuen Bildhashes berechnet. `selection_protocol=split_A_train_v1`
und `holdout_splits=[]` halten die neue Auswahl ausdrücklich fest.

Die automatische NFFA-Evaluation verwendet ebenfalls **nur Split A**. Split B
hat andere Partitionen; seine Holdouts überlappen nun teilweise das SSL-Training
und sind keine unabhängige Evaluation dieses Encoders. Aachen bleibt unverändert.
Der historische 8.061-Bilder-Lauf und seine A/B-Ergebnisse bleiben erhalten.

Es gibt keine neue SHA-256-Prüfung der Bilddateien und keine automatische Auswahl
anderer Datensätze. Bildinhalte müssen gegenüber dem ursprünglichen Audit
unverändert sein. Validierungs-/Testbilder werden bei der Auswahl nicht geöffnet.
Für die spätere Evaluation werden die bereits geprüften Downstream-Metadaten
weiterverwendet; Small-Features werden nicht als Large-Features wiederverwendet.

Die vorhandene `crop_metadata_strip`-Entfernung erfolgt vor den zufälligen
SEM-Crops. Der Pilot verwendet die neue Large-Rezeptur mit global 518×518 und
lokal 224×224, nicht die früheren Small-Crops 224/98. Die NFFA-Orientierung ist
nicht freigegeben: keine zusätzlichen Rotationen oder Spiegelungen. Es werden
keine festen zufälligen Ansichten gespeichert.

## Initialisierung und unverändertes Rezept

Alle Arme starten unabhängig vom **originalen Meta DINOv2 ViT-L/14 ohne
Register**, nicht vom alten Small-SSL-Checkpoint. Die offiziellen Gewichte liegen
unter `outputs/pretrained/dinov2_vitl14_pretrain.pth`. Downloadquelle, Bytezahl,
lokaler SHA-256 und gepinnter Meta-Commit stehen in
`outputs/pretrained/dinov2_vitl14_provenance.json`; der SHA ist außerdem in der
Pilotkonfiguration festgelegt. Der lokale Hash fixiert den offiziellen
HTTPS-Download; er ist keine zusätzliche, unabhängig signierte Herstellerprüfsumme.

Fixierter SHA-256:
`d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`.

Konfiguration: `configs/sem_ssl/sem_cellstyle_nffa_vitl14_pilot.yaml`.
Der aktuelle Pilot verwendet `evaluation.head_profile: residual_v1`: ein
Residual-MLP512/256/128 mit Dropout0.1 und den bereits residualen CNN8 mit
Dropout2d0.1. Details und Abgrenzung zum historischen Head-Rezept stehen im
[Residual-Head-Guide](residual_downstream_heads.md). Original-Large und alle
vier Arme erhalten dieselben neuen Downstream-Heads.
Abgesehen von Daten, Originalgewichtspfad und Ausgabe-/Metadatenpfaden bleibt die
Large-Rezeptur unverändert. Seed 42, effektiver Batch 128, BF16, native SDPA,
originale Meta-Losses, Teacher-EMA, Lernraten und Regularisierung sind in allen
Armen identisch gemäß dem [Studienprotokoll](sem_cellstyle_regularization.md).
Die physische Batchgröße 16 oder 32 wird vorab auf derselben A100 festgelegt und
dann zwischen den Armen nicht verändert.

## Budget aus der tatsächlichen Bildzahl

`step(p) = ceil(p * 12801 / 128)`:

| Originalbild-Pässe | Optimizer-Updates |
| --- | ---: |
| 0 | 0 |
| 1 | 101 |
| 5 | 501 |
| 10, erster Stop | 1.001 |
| 25 | 2.501 |
| 50 | 5.001 |
| 100, Schedule-Horizont | 10.001 |

LR-Warmup: 501 Updates; letzte Head-Schicht: LR null für die ersten 101 Updates;
Teacher-Temperatur-Warmup: 2.001 Updates. Der Zehn-Pass-Checkpoint liegt im
100-Pass-Schedule. Pro Arm werden durch vollständige Batches 128.128 Originalbilder
präsentiert, geringfügig mehr als 10×12.801 aufgrund der Aufrundung.
Benchmark- und Kalibrierungsupdates sind zusätzliche, verworfene Diagnostik und
werden nicht als wissenschaftliche Trainingsschritte gezählt.

## Start auf A100

Der folgende Befehl ist ein tatsächlicher Startauftrag. Er führt zunächst die
vollständigen A100-Speicher-/Numerik-/Resume-/Exportchecks und die gemeinsame
Trainingsdaten-Kalibrierung aus. Nur nach Erfolg folgen nacheinander A–D bis
1.001 Updates, Checkpoint-Verifikation, ausschließlich Validation auf NFFA A
(LogReg + MLP) und Aachen A3 (Linear + CNN8) sowie Vergleichsbericht.
Fehlende Checks oder eine fehlgeschlagene Kalibrierung stoppen den Ablauf.
Die genehmigte Ausnahme für numerische CUDA-Abweichungen nach dem
Resume-Update ist in der NFFA-Pilotkonfiguration aktiviert.

```bash
cd /path/to/nffa-sem-dinov2
CUDA_VISIBLE_DEVICES=2 bash scripts/run_sem_cellstyle_nffa_pilot.sh
```

Der Wrapper verwendet direkt das Python aus `sem-gpu`, setzt die CPU-Threads
auf eins und startet genau eine sichtbare A100 80 GB. Training: 16 Loader-Worker;
Downstream: NFFA 8, Aachen 16. Es erfolgt kein SSH-Aufruf und keine Installation.

Nur GPU-Checks und Kalibrierung, ohne wissenschaftliches Training:

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_sem_cellstyle_nffa_pilot.sh --stage checks
```

Plan ohne GPU-Zugriff:

```bash
bash scripts/run_sem_cellstyle_nffa_pilot.sh --stage plan
```

Unterbrechung: denselben Startbefehl erneut ausführen. Der bestehende Einstieg
prüft die unveränderten Verträge, verwendet abgeschlossene Checks weiter und
setzt eigene Trainingszustände fort. Historische Ausgaben werden nicht ersetzt.
Nach Änderungen an Daten, Rezept, Koeffizienten oder gebundenem Quellcode ist ein
neuer Studienordner erforderlich.

### Korrektur nach dem ersten A100-Check am 18.11.2026

Im ersten realen Lauf scheiterten B16 und B32 am strengen Resume-Selbsttest,
nachdem die Warmup-/Messupdates und die Teacher-Exportprüfung durchgelaufen waren.
Es wurde kein wissenschaftliches Training gestartet. Die gespeicherten Fehler
sind keine CUDA-OOM-Meldungen; `finite: false` wurde vom allgemeinen Fehlerhandler
gesetzt und belegt hier keine NaN-Verluste.

Im Benchmark wurde ein geladenes Optimizer-Dictionary zweimal verwendet.
PyTorch übernimmt dabei CPU-AdamW-Schrittzähler per Referenz, auch für
CUDA-Parameter. Der erste Probe-Update veränderte damit den Ausgangszustand des
zweiten. Der korrigierte Prüfcode lädt für jeden Restore eine unabhängige Kopie
des Optimizer-Zustands. Losses, EMA, Optimizer-Rezept und strenge
Gleichheitsprüfungen bleiben unverändert. CPU-Reproduktion und Regression stehen
im Resume-Prüfnachweis.
**Ergänzung nach der Wiederholung:** Auch
`nffa_split_a_vitl14_resumecheck_v2_seed42` scheiterte auf A100 bei B16 und B32
am exakten Update-Vergleich. Der nachgewiesene Alias-Fehler erklärt diesen
zweiten Befund nicht allein. Beide historischen Ordner bleiben erhalten.

**Stand 19.11.2026.** Die v3-Diagnose auf GPU 2 bestätigt exakten
registrierten Restore-Zustand, identische Batches und Zufallszustände sowie
identischen Loss. Die Backbone-Gradientennormen und nachfolgenden Updates
weichen ab. Der deterministische Gegencheck v4 brach im ersten
Warmup-Backward bei `upsample_bicubic2d_backward_out_cuda` ab. Damit ist
der Operator identifiziert, der strikten Determinismus mit dem bestehenden
Rezept verhindert. Auf Nutzerfreigabe darf der NFFA-Pilot Abweichungen in
endlichen Student-/Teacher-Gleitkommatensoren und AdamW-Momenten **nach dem
Update** akzeptieren. Checkpoint-Zustand, Masken, Eingaben, Zufallszustände,
Schrittzähler, Loss, Center und Optimizer-Gruppen werden weiterhin exakt
verglichen. Der Benchmark protokolliert die Abweichungen und kennzeichnet
das Ergebnis ausdrücklich als akzeptierte Abweichung, falls es nicht
bitgenau ist. Der A100-Vorlauf mit der Änderung steht aus.
[Ergebnisse und Freigaberegel](sem_cellstyle_resume_diagnostics.md)

## Ausgaben und Prüfstatus

Der freigegebene Pilot schreibt in
`outputs/sem_ssl_regularization/nffa_split_a_vitl14_residual_heads_v1_resume_approved_seed42/`.
Unterordner: `configs/`, `checks/`, `runs/A` bis `runs/D`, `logs/`,
`evaluation/`, `evaluation_reports/`, `report/`; gemeinsame `calibration.json`.
Der bestehende Grid-Reporter legt Tabellen und Plots in seinem Report-Verzeichnis
an. Die Original-Large-Referenz bei Schritt null wird gemeinsam ausgewertet;
alle angepassten Encoder sind eingefrorene EMA-Teacher-Exporte.

Vorbereitungsnachweise liegen separat unter
`outputs/sem_ssl_regularization/nffa_split_a_vitl14_pilot_preparation/`, damit der neue
Studienordner beim ersten Start leer ist. CPU-Prüfnachweise:
das lokale Prüfarchiv.
Die frühere Vorbereitung mit 8.061 Bildern ist unter
das lokale Prüfarchiv erhalten. Dort ist auch der
strikt geladene Original-Large-Backbone geprüft. Ein CPU-Forward
mit 224×224 liefert endliche CLS-Features `(1,1024)` und Patchfeatures
`(1,256,1024)`; 24 Blöcke und keine Register bestätigt. Das ist keine
GPU-Trainingsmessung. Der Meta-Checkout bleibt unverändert am gepinnten Commit.
Der ursprüngliche Teststand mit 8.061 Bildern war 275 bestanden, 3 übersprungen,
3 Subtests bestanden. Nach der Umstellung: **54 gezielte Tests bestanden;
Gesamtsuite 280 bestanden, 3 übersprungen, 3 Subtests bestanden**. Tatsächliche
Auswahl von12.801 Bildern, identische Small-/Large-Defaults und zwei echte
Large-Crop-Prüfungen bestätigt. Der separate Nachweis steht im neuen
Prüfverzeichnis. Keine Pakete installiert oder verändert.
Der erste A100-Vorlauf scheiterte wie oben dokumentiert am Resume-Selbsttest.
Für diesen NFFA-Large-Pilot liegen weiterhin weder gültige Benchmark-Empfehlungen
noch wissenschaftliche Trainings- oder Downstream-Ergebnisse vor.

Dateizuständigkeiten:

- `src/sem_cellstyle/data.py`: kleiner NFFA-Adapter mit bestehenden Auswahlprüfungen
  und neuer Large-Transformation; die Losses und Update-Schritte bleiben unverändert.
- `configs/sem_ssl/sem_cellstyle_nffa_vitl14_pilot.yaml`: NFFA-Pfade und festgelegte
  Originalgewichte, ansonsten das vorhandene Large-Rezept.
- `scripts/run_sem_cellstyle_nffa_pilot.sh`: ein A100-Startbefehl für den vorhandenen
  Studien-Einstieg.
- `tests/test_sem_cellstyle_nffa.py`: sichere Auswahl, Metadatenbindung, fehlende
  Bild-Neuhashes, Textstreifen-Entfernung, Crop-/Maskenformen und unverändertes Rezept.

## Standard für die bisherigen Small-Skripte

`scripts/train_sem_ssl.py` und `scripts/benchmark_sem_ssl_a100.py` verwenden für
neue Aufrufe ohne `--config` jetzt `configs/sem_ssl/nffa_default.yaml`: Split A,
12.801 Trainingsbilder und dasselbe wiederverwendete Inventar. Das Small-Rezept
und sein 200-Update-Budget bleiben ansonsten unverändert. Historische Konfigurationen
`nffa_200step.yaml`, `nffa_5000step.yaml` und die Low-LR-/Freeze-Konfigurationen
bleiben ausdrücklich reproduzierbare ältere A/B-Protokolle. Bei manueller
Downstream-Evaluation eines neuen Split-A-Encoders `--splits A` verwenden.
