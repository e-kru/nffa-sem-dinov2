# SEM SSL: Low LR und Freeze 8

Stand: 12.11.2026. Die Studie ist vorbereitet. Die drei neuen GPU-Läufe wurden
nicht gestartet; ihre gemessenen Ergebnisse und die daraus abgeleitete Empfehlung
entstehen erst nach dem manuellen Start.

## Fragestellung und Referenz

Beim bestehenden 5.000-Schritt-Lauf verschlechtern sich NFFA und der lineare
Aachen-Head. CNN8 zeigt einen kleinen Gewinn bei Schritt 500. Wir prüfen, ob
kleinere Updates bzw. feste frühe Transformer-Blöcke die Verschlechterung verringern.

Referenz: [gespeicherter Lauf](../outputs/sem_ssl/nffa_5000step_seed42/run.json),
[Rezept](../configs/sem_ssl/nffa_5000step.yaml),
[vorhandene Validation-Ergebnisse](../outputs/sem_ssl_reports/nffa_5000step_workers_v1/comparison.json).
Schritte 0/500/1.000 werden daraus übernommen. Es gibt kein erneutes Baseline-Training.

## Teacher-Audit

Geprüft wurden der bestehende Adapter, Metas Quellen am gepinnten Commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`, der echte historische
Schritt-0-Checkpoint und die CPU-Integrationstests.

| Prüfpunkt | Tatsächliches Verhalten / Nachweis |
| --- | --- |
| Student-Backbone | Originaler Meta-ViT-S/14; Release-State wird mit `strict=True` geladen und tensorweise verglichen. Die Initialisierungsdatei muss den gepinnten SHA256 haben. |
| Student-Head | Frischer originaler `DINOHead`, 65.536 Prototypen; DINO/iBOT teilen den Head. Das veröffentlichte Backbone-Artefakt enthält keinen fortsetzbaren SSL-Head. |
| Teacher-Initialisierung | Backbone wird kopiert und ein separater Head konstruiert. Anschließend lädt der **vollständige Teacher** den gesamten Student-State strikt, einschließlich Head-MLP und WeightNorm-Parametern. |
| Historischer Schritt 0 | Auf CPU read-only geprüft: vollständiger Student-/Teacher-State gleich; Teacher frozen/eval; Export gleich Teacher; Exportdatei entspricht dem Originalrelease. |
| EMA | Nach `optimizer.step()`: Parameter-EMA von Backbone und Head ohne Teacher-Gradienten. Momentum aus demselben 5.000-Schritt-Schedule, 0,996 → 1,0. |
| Teacher-Modus | `requires_grad_(False)`; bei jedem `model.train()` erneut `teacher.eval()`. |
| Temperatur | 200 Warmup-Schritte von 0,04 auf 0,07, danach 0,07. Der Stopp bei 1.000 verkürzt den Schedule nicht. |
| Zentrierung | Originale DINO-/iBOT-Center in FP32, Momentum 0,9; verzögertes Update vor dem nächsten Teacher-Softmax. Vor dem Checkpoint wird das ausstehende Update geflusht. Resume-Parität wird getestet. |
| DINO/iBOT/KoLeo | Originale Meta-Klassen mit Gewichten 1/1/0,1 und ursprünglicher Masken-/Crop-Zuordnung. Der bestehende Test vergleicht Loss-Beiträge und Student-Gradienten mit Metas `SSLMetaArch.forward_backward` auf CPU, mit Test-Stellvertretern für CUDA/FSDP/Tensor-Packing. |
| Loss-Logging | Der Adapter zeigt tatsächliche gewichtete iBOT-/KoLeo-Beiträge. Meta protokolliert für diese beiden Werte die Hälfte; die Backprop-Beiträge stimmen im Paritätstest überein. |
| Downstream-Export | Ausschließlich `teacher.backbone.state_dict()`. Schritt 0 behält die Release-Dateibytes. Verifikation prüft Export gegen Teacher und normalerweise auch den Forward nach Reload. |

### Bestätigte Fehler

Im auditierten Teacher-/Loss-/Exportpfad wurde **kein bestehender
Implementierungsfehler nachgewiesen**. Der zufällig initialisierte Head wird
korrekt vom Student in den Teacher kopiert. Der zusätzliche historische
Schritt-0-Audit nutzt keinen neuen Forward; bestehende Forward-Nachweise und
neue CPU-Tests sind separat verlinkt.

Die neue Freeze-Option benötigt eine zusätzliche Regel: feste Parameter werden
auch von der EMA ausgelassen. Wiederholte FP32-Arithmetik kann selbst bei
gleichen Eingabetensoren Rundungsänderungen verursachen. Diese Regel betrifft
die neu eingefrorenen Parameter; der bisherige unbeschränkte Pfad bleibt gleich.

### Methodische Bedenken

- Ein vortrainierter Backbone mit frischem Head ist ein anderer Ausgangszustand
  als ein vollständiger SSL-Checkpoint mit vortrainierten Heads, Centern und
  Optimizer. Frühe Teacher-Targets werden durch den frischen Projektionsraum
  bestimmt. Das ist eine Hypothese zur Anpassungsdynamik, kein nachgewiesener Fehler.
- EMA verhindert nicht grundsätzlich eine ungünstige Student-/Teacher-Drift.
  Der negative Downstream-Befund benennt noch keine Ursache.
- Low LR reduziert auch die Head-LR. Die Ablation isoliert deshalb nicht
  ausschließlich die Backbone-Lernrate.
- Ein Seed und Validation-Auswahl unter mehreren Checkpoints belegen keine
  robuste Verbesserung. Bitidentische GPU-Verläufe werden nicht behauptet.

Nachweise: echter Schritt-0-Audit,
[ursprüngliche Export-Prüfung](../outputs/sem_ssl_checks/nffa_5000step_verified.json),
[Loss-/Gradienten-/EMA-Tests](../tests/test_ssl_pilot.py),
[neue Freeze-/Resume-Tests](../tests/test_ssl_ablations.py).
Die bisherigen Abweichungen vom vollständigen Meta-Trainer stehen weiterhin
im [Pilot-Guide](sem_ssl_pilot.md).

## Kontrollierte Konfigurationen

| Experiment | Config | Nominale maximale LR | Minimum-LR | Feste Blöcke |
| --- | --- | ---: | ---: | --- |
| Referenz | `nffa_5000step.yaml` | 5e-5 | 1e-6 | keine |
| A – Low LR | [nffa_ablation_low_lr.yaml](../configs/sem_ssl/nffa_ablation_low_lr.yaml) | 5e-6 | 1e-7 | keine |
| B – Freeze 8 | [nffa_ablation_freeze8.yaml](../configs/sem_ssl/nffa_ablation_freeze8.yaml) | 5e-5 | 1e-6 | 0–7 |
| C – Low LR + Freeze 8 | [nffa_ablation_low_lr_freeze8.yaml](../configs/sem_ssl/nffa_ablation_low_lr_freeze8.yaml) | 5e-6 | 1e-7 | 0–7 |

`optim.base_lr` sinkt für A/C von 0,0002 auf 0,00002. Bei B64 ergibt die
unveränderte Skalierung `base_lr * sqrt(64/1024)` die angegebenen nominalen
LR-Werte. Layerwise-/Patch-Embedding-Multiplikatoren bleiben erhalten. Auch
die Minimum-LR wird mit 0,1 skaliert, damit der ganze LR-Verlauf exakt ein
Zehntel beträgt. Die Last-Layer-Freeze-Dauer bleibt 100 Schritte.

Alle drei Läufe starten unabhängig mit Seed 42 aus dem **originalen Meta-Release**,
mit derselben Auswahl von 8.061 NFFA-V3-Bildern: Split-B-Train, zusätzlich durch
Split-A-Train erlaubte ganze Gruppen, eingeschlossene `crop_v2`-Bilder. Es gelten
B64, 16 Trainings-Worker, BF16, native Attention, kein Bildcache und die
Original-Multicrops mit zwei globalen 224er- und acht lokalen 98er-Ansichten.
Teacher-Temperatur, Momentum, Weight Decay, Losses, Maskierung, Vorverarbeitung,
Augmentation und Clipping bleiben erhalten.

Jede Konfiguration setzt `schedule_total_steps=5000`, `run_until_step=1000`
und Milestones `[0, 500, 1000]`. Vorbereitung und Tests vergleichen sämtliche
Schedule-Arrays mit der Referenz. Die aufgelöste YAML wird gegen die tatsächlich
gespeicherte Baseline geprüft; zusätzliche Abweichungen führen zum Abbruch.

### Genaue Bedeutung von Freeze 8

`student.freeze_first_blocks=8` setzt `requires_grad=False` für alle Parameter
der Blöcke `backbone.blocks.0` bis `.7`. Metas originale Parametergruppen-Funktion
schließt sie damit aus AdamW aus; sie erhalten keinen Weight Decay. Teacher-EMA
überspringt genau diese Parameter. Die Checkpoint-Prüfung vergleicht alle
State-Tensoren dieser Blöcke in Student **und** Teacher bitgenau mit dem Release.

Blöcke 8–11, Patch-Embedding, Positions-/CLS-/Mask-Token, finale Norm und
Projektionskopf bleiben trainierbar. Student-Trainingsmodus und DropPath bleiben
unverändert. Durch die festen Blöcke können weiterhin Gradienten zu trainierbaren
Eingangsparametern fließen. Ein proportionaler Zeit-/Speichergewinn wird nicht
vorausgesetzt.

## Ein Befehl auf A100

Auf `gpu-host`, mit einer zugewiesenen A100 80 GB (hier physische GPU 0):

```bash
CUDA_VISIBLE_DEVICES=0 bash /path/to/nffa-sem-dinov2/scripts/run_sem_ssl_ablations.sh
```

Der Wrapper wechselt ins Projekt und verwendet direkt den vorhandenen
`sem-gpu`-Interpreter. Er installiert nichts. Genau eine sichtbare
physische A100 und BF16-Unterstützung werden geprüft. Serieller Ablauf:

1. Konfigurationen, Inventar, Referenzberichte und Protokolle prüfen.
2. A trainieren und 0/500/1.000 verifizieren; danach B und C ebenso.
3. Pro Experiment Schritt 500 und 1.000 auswerten: NFFA A/B mit LogReg/MLP
   und acht Workern; Aachen A3 mit Linear/CNN8 und 16 Workern.
4. Gemeinsamen Bericht einschließlich Referenz 0/500/1.000 erzeugen.

Nur Vorbereitung und Befehle, ohne GPU-Job oder Studienausgaben:

```bash
bash scripts/run_sem_ssl_ablations.sh --dry-run
```

Neuer Studienordner: `--output outputs/sem_ssl_ablations/<neuer_name>`.
`--report-only` sammelt ausschließlich bereits abgeschlossene Ergebnisse.

## Daten- und Evaluationswiederverwendung

Das Training verwendet `train_sem_ssl.py --inventory` mit dem gespeicherten
Referenzinventar. Die bestehende `dataset_from_inventory` prüft Auswahl und kleine
Manifest-/Split-Dateien erneut. **Bildbytes werden nicht erneut zum Hashen gelesen.**

Der Evaluationsrunner erhält optional `--verified-data-root` mit dem geprüften
Schritt-0-Encoderordner. NFFA übernimmt dessen Train-/Val-Records und gespeicherte
Bild-Hashes. Aachen übernimmt Records, Masken-Hashes und Originalgrößen aus der
vorhandenen A3-Konfiguration. Manifestidentität und Gruppenzuordnung werden geprüft.
Der Rohdatenbestand muss unverändert bleiben; eine neue Pixelintegritätsprüfung
ist damit ausdrücklich nicht verbunden. Ohne die Optionen bleibt der bisherige
vollständige Auditpfad Standard.

Jeder neue Encoder braucht neue Features. Sie entstehen getrennt je Encoder-Hash
und Experiment; innerhalb eines Encoders teilen sich die Heads den Feature-Cache.
Normale Checks für Checkpoints, Code, kleine Manifeste und Feature-Dateien bleiben
aktiv. Es gibt keine neue Datenbereinigung, Hashing-Pipeline oder Vorverarbeitung.
Head-Konfigurationen, Original-Maskengeometrie und Auswahlregeln bleiben gleich.
Testdaten werden nicht zur Modellauswahl verwendet.

Für die gemeinsame Protokollprüfung wird nur der Dateihash von
`scripts/evaluate_sem_encoder.py` als dokumentierte Änderung an der CLI zur
Inventarwiederverwendung ausgeklammert. Alte und neue Hashes bleiben in den
historischen Preflights bzw. `study.json`. Alle übrigen Transform-/Head-/Trainings-/
Metrikquellen sowie Daten und Hyperparameter müssen übereinstimmen. Historische
Quellenangaben werden nicht umgeschrieben.

## Ausgaben und Wiederaufnahme

```text
outputs/sem_ssl_ablations/low_lr_freeze8_seed42_v1/
  study.json                     # Konfigurationen, Quellen, Software-/Run-Verträge
  dataset_inventory.json         # bereits geprüftes Referenzinventar
  A_low_lr/                      # entsprechend B_freeze8/, C_low_lr_freeze8/
    train/
      run.json, status.json, metrics.jsonl, selected_image_ids.txt
      checkpoints/step_{000000,000500,001000}/
      recovery/
    checkpoints_verified.json    # inkl. Freeze-/Export-Nachweis
    evaluation/dinov2_vits14_<hash>/
    logs/train.log, verify.log, evaluation_{0,1,2,3}.log
  report/
    comparison.md
    comparison.json
    comparison.csv
```

Nach Unterbrechung denselben Startbefehl wiederholen. Locks verhindern mehrere
Writer. Abgeschlossene Trainingsstufen werden übersprungen, vorhandene
Prüfbelege gegen die Checkpoint-Dateien geprüft. Unvollständige Läufe verwenden
`--resume` mit Optimizer, Teacher, Centern, RNG und Daten-Cursor. Extraktion und
Head-Training verwenden ebenfalls das vorhandene Resume-Verfahren; Logs werden
angehängt. SIGINT/SIGTERM wird an den aktiven Unterprozess weitergegeben. Ein
unterbrochenes Training startet keine spätere Stufe. Ein harter Kill kann bis
zum letzten vollständigen Checkpoint zurücksetzen.

Änderungen an Config, Datenidentität, Software oder relevanten Quellen werden
bei Resume abgelehnt. Der Report wird nach vollständiger Sammlung atomar
publiziert; identische vollständige Reports sind wiederverwendbar. Historische
Läufe und Checkpoints werden nicht verändert.

## Auswertung und Empfehlung

Neun Vergleichszeilen: Referenz 0/500/1.000 plus je zwei neue Stände pro Ablation.
Vier NFFA-Macro-F1-Werte; für beide Segmentierungs-Heads MA-IoU/mIoU bei 0,50 und
ausgewählter Val-Schwelle separat. Die CSV ergänzt Differenzen gegenüber Original
und gleich lang trainierter Referenz in Prozentpunkten.

Primäre Punktwerte für die Empfehlung sind vier NFFA-Macro-F1 und zwei Aachen-
MA-IoU bei 0,50. Gewinner werden je Aufgabe/Head benannt. Ein allgemeiner Kandidat
für weitere Seeds muss alle sechs Originalwerte mindestens erreichen und einen
verbessern. Verbesserungen gegenüber der gleich lang trainierten Referenz werden
zusätzlich ausdrücklich genannt. Bei gemischten Ergebnissen bleibt das Original
die allgemeine Referenz; aufgabenspezifische Kandidaten bleiben sichtbar. Das
ist eine beschreibende Auswahlregel ohne Signifikanzbehauptung. Low LR verändert
auch die Head-LR; eine kausale Aussage nur über den Backbone ist damit begrenzt.
Die Pipeline startet keine längere Trainingsfortsetzung.

## Nachweise der Vorbereitung

Abschließende CPU-Suite in `sem-gpu`: **213 passed, 3 skipped,
3 subtests passed** in 54,93 s. Die Sandbox beschränkt die drei zusätzlichen
Mehrprozess-/IPC-Prüfungen. Bash-Syntax und vollständiger Metadaten-/Konfigurations-/
Schedule-/Protokoll-Dry-Run sind geprüft. Es wurden keine Pakete installiert.

Die [CPU-Tests](../tests/test_ssl_ablations.py) prüfen Initialisierung/Head-Kopie,
LR-Skalierung, andere Schedules, Gradienten-/Optimizer-/EMA-Ausschluss, feste
Blockwerte, Resume, Export, Datenwiederverwendung, CLI-Weitergabe, Protokollvergleich,
Bericht und Abbruch der Stufenkette. Die bisherigen Loss-/Gradienten-Paritätstests
bleiben aktiv; mathematische Losses wurden nicht verändert.

Vollständige Ausgaben: CPU-Audit,
abschließende CPU-Suite,
geprüfter Ablaufplan.
Die neuen A100-Ergebnisse sind vor dem manuellen Start noch offen.
