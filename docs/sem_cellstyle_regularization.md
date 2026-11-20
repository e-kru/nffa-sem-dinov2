# ViT-L/14: kontrollierte L2-SP- und Feature-Distillation-Studie

Stand18.11.2026: implementiert und mit kleinen CPU-Fixtures geprüft. Der große
SEM-Korpus ist weiterhin nicht freigegeben/configuriert. Keine Pakete wurden
installiert, keine GPU-Jobs und keine vier wissenschaftlichen Läufe gestartet.
Koeffizienten sind bewusst noch nicht numerisch festgelegt: erst reale,
ausschließlich trainingsbasierte Gradientenkalibrierung, dann unveränderlich einfrieren.

## Vier Arme, zwei Referenzen

| Arm | L2-SP | Feature Distillation | Config |
| --- | --- | --- | --- |
| A | aus | aus | `configs/sem_ssl/sem_cellstyle_regularization_A.yaml` |
| B | an | aus | `configs/sem_ssl/sem_cellstyle_regularization_B.yaml` |
| C | aus | an | `configs/sem_ssl/sem_cellstyle_regularization_C.yaml` |
| D | an | an | `configs/sem_ssl/sem_cellstyle_regularization_D.yaml` |

Abgesehen von den zwei Schaltern und getrennten Ausgabeordnern enthalten die
Vorlagen dasselbe Rezept. Der Einstieg nimmt eine gemeinsame fertig angebundene
Basis-YAML, übernimmt deren verifiziertes Manifest, Originalgewichte und Rezept
für alle vier Arme und schreibt separate eingefrorene Arbeitskonfigurationen.
Keine Initialisierung aus einem bereits adaptierten Encoder; jeder Arm startet
frisch mit denselben Original-Large-Gewichten und derselben Head-Initialisierung.

Unverändert: originaler Meta ViT-L/14 ohne Register, originale DINO/iBOT/KoLeo-
Losses, SEM-Augmentierungen, Backbone-LR1e-5/Head-LR1e-4, Seed42,
AdamW/WD/Layer-Decay, EMA/Center/Temperaturen,100-Pass-Horizont und erster Stop
bei10 Pässen. Effektiv128 Originalbilder pro Update. Physisch16×8 oder32×4 wird
vor der Kalibrierung gewählt und bleibt in allen Armen gleich. Keine automatische
Verringerung der effektiven Batch oder der Bildauflösung bei Speicherproblemen.

Die Hauptreferenz ist **originales Meta-Large ohne Post-Training**. Die zweite
Referenz ist **A am gleichen Meilenstein**. Alte Small-Ergebnisse werden weder
überschrieben noch als Large-Baseline verwendet.

## Losses und genaue Ausführung

Das Trainingsziel ist:

`SSL + lambda_sp * sum_j ||theta_j-theta0_j||²/(2*Ntheta)`
`    + lambda_feat * (lambda_cls * (1-cos_cls) + lambda_patch * (1-cos_patch))`.

`Ntheta` ist die Anzahl der **skalaren** vortrainierten Backbone-Parameter,
nicht die Anzahl der Parametertensoren. Neue SSL-Heads sind ausgeschlossen.
L2-SP wird genau einmal nach den SSL-/Feature-Microbatches und vor dem
Gradienten-Clipping addiert. Seine exakte analytische Ableitung
`lambda_sp*(theta-theta0)/Ntheta` wird pro Tensor eingetragen; dadurch entsteht
kein zusätzlicher vollständiger Autograd-Graph für ViT-L. CPU-Tests vergleichen
diese Ableitung mit direktem Autograd der angegebenen Formel. Kein zusätzliches
Dividieren durch die Accumulation-Anzahl.

Feature Distillation nutzt **zusätzliche unmaskierte globale Student-Ansichten**.
Der Referenzencoder bekommt genau dieselben bereits augmentierten Tensoren,
gleiche Reihenfolge und gleiche Patchkoordinaten. Maskierte iBOT-Features werden
hier nicht wiederverwendet. CLS- und Patch-Cosinus werden getrennt in FP32
berechnet und protokolliert. Jeder Feature-Loss wird durch die Microbatchanzahl
geteilt; Gradienten fließen ausschließlich zum Student-Backbone.

**Explizite neue methodische Entscheidung:** Für diese zusätzliche Feature-
Verzweigung werden Referenz und Student-Backbone im eval-Modus ausgeführt,
während Gradienten beim Student eingeschaltet bleiben. Das vermeidet ein
Distillationssignal allein aus zufälligem DropPath. Auch der Backward inklusive
Activation-Checkpoint-Recompute bleibt in diesem Modus. Anschließend wird der
ursprüngliche Student-Trainingsmodus wiederhergestellt. Die normalen SSL-
Forwards behalten DropPath0,1 und alle bisherigen Einstellungen; ihr RNG-Strom
wird von der zusätzlichen eval-Verzweigung nicht weitergeschaltet.

Der originale Referenzencoder bleibt eingefroren, in eval und unter `no_grad`;
er ist **nicht** der DINO/iBOT-EMA-Teacher. EMA und Center werden weiterhin genau
einmal pro vollständigem Optimizer-Update aktualisiert. Bei D teilen sich L2-SP
und Distillation dieselben unveränderlichen Originalgewichte. B hält nur
Gewichtsanker und führt keinen Referenzforward aus. A erzeugt keine Anker,
keinen zusätzlichen Encoder und keine zusätzlichen Trainingsforwards.

Die SSL-Autograd-Aktivierungen werden vor dem zusätzlichen Feature-Forward
freigegeben. Nur der Student nutzt nicht-reentrantes Activation Checkpointing.
Original-Losses und der externe Meta-Checkout bleiben unverändert.

## Gemeinsame trainingsbasierte Kalibrierung

`lambda_sp` und `lambda_feat` beginnen als `null`. `lambda_cls=lambda_patch=1`
sind die vorab festgelegten relativen Tokengewichte, nicht pro Arm optimiert.
Beide sind über `regularization.lambda_cls`/`lambda_patch` in der gemeinsamen
Basis-YAML konfigurierbar und werden vor der Kalibrierung festgelegt.

An der Initialisierung ist L2-SP exakt null. Sehr kleine anfängliche Warmup-
Updates können auch im BF16-Featurepfad unsichtbar sein. Die Kalibrierung nutzt
deshalb ein **eigenes verworfenes Diagnoseexperiment**: Originalinitialisierung,
Schedule-Index am Ende des bestehenden LR-Warmups,20 Updates zur Erzeugung einer
messbaren Verschiebung, danach10 Gradientensonden. Es werden nicht fiktiv fünf
Datendurchläufe behauptet: `schedule_index`, tatsächliche Kalibrierungsupdates,
verbrauchte Trainingsbilder und Draw-IDs sind getrennt gespeichert. Die
wissenschaftlichen Arme starten anschließend alle bei Schedule-Index0 mit dem
ursprünglichen Warmup; keine Kalibrierungsgewichte werden übernommen.

Die erste Diagnose optimiert ausschließlich SSL. Zusätzliche `autograd.grad`-
Sonden messen den vollständigen, über Microbatches akkumulierten rohen Feature-
Gradienten. L2-SP-Gradienten werden analytisch gemessen. Normen beziehen sich auf
den Backbone vor Clipping; die zufälligen Head-Gradienten gehen nicht in den
Nenner ein. Der Eintrag des SP-/Featuregradienten wird dabei verhindert.

Für jeden Regularisierer wird ein gemeinsamer Koeffizient gewählt, sodass sein
maximal beobachtetes Verhältnis `||lambda*g_reg||/||g_SSL||` höchstens0,10 ist.
Der Median muss mindestens0,01 betragen. Rohgradienten müssen messbar, endlich
und ungleich null sein. Der Featuregradient muss zudem mindestens10× größer
als das direkt an der unveränderten Originalinitialisierung gemessene numerische
Gradientenrauschen sein; andernfalls wird kein Koeffizient freigegeben. Das ist eine explizite technische Kalibrierungsregel,
kein behaupteter literaturbasierter optimaler Wert.

Ein zweiter, ebenfalls verworfener D-Diagnoselauf startet unabhängig vom Original
mit **denselben festgelegten Koeffizienten**. Er prüft Nichttrivialität und einen
konservativen kombinierten Gradientendreiecksbound von höchstens0,25. Erfüllt er
die Bedingungen nicht, stoppt die Vorbereitung. Es wird weder pro Arm
nachgestimmt noch anhand von Downstream-Validation ein Koeffizient ausgewählt.
Änderung des gemeinsamen Kalibrierungsprotokolls erfordert einen neuen Studienordner.

`calibration.json` enthält Werte, Normen, Verhältnisse, alle Probezeilen,
Datensatz-/Gewichts-/Batch-/Quellcodeidentität und die eingefrorene Freigabe.
Alle vier finalen YAMLs verweisen auf exakt diese Datei und deren SHA256.
Die Werte sind in B/D bzw. C/D identisch und vor jeder Downstream-Evaluation
fixiert. Die frühe Gradientenkalibrierung garantiert keine Nichtdominanz über
den gesamten späteren Trainingsverlauf; sie ersetzt keine Ergebnisanalyse.

## Ein Einstieg, sichere Fortsetzung

Jetzt ausführbar, ohne Korpus oder GPU:

```bash
cd /path/to/nffa-sem-dinov2
bash scripts/run_sem_cellstyle_regularization_grid.sh --stage plan
```

Die Basis-YAML muss später ein freigegebenes Trainingsmanifest samt bestehendem
Receipt und die Original-Large-Gewichte samt vertrauenswürdigem SHA enthalten.
Details der Schnittstelle: [Large-Rezeptur](sem_cellstyle_vitl14.md). Der Provider
liest keine Audit-Kandidaten automatisch ein, berechnet keine neuen Bildinhalts-
Hashes und prüft die dokumentierten Ausschlüsse. Die vorgelagerte Kuration muss
NFFA A/B Validation/Test und Gruppen, ganz Aachen und ggf. NASA-Benchmarks
nachweislich ausgeschlossen haben. `--example-originals` ist nie ein Datensatz.

**Zuerst nur vollständige GPU-Checks und Kalibrierung**, manuell auf A100:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sem_cellstyle_regularization_grid.sh \
  --stage checks --config configs/sem_ssl/sem_cellstyle_vitl14_ready.yaml \
  --output outputs/sem_ssl_regularization/v1_seed42
```

Der Bash-Einstieg verwendet direkt `sem-gpu` und setzt OMP/MKL/OpenBLAS auf1.
Er verändert die Umgebung nicht. Exakt eine sichtbare A10080GB und BF16 werden
gefordert. Zuerst läuft ein separater D-Speicherprobelauf mit Diagnosekoeffizienten
für physisch16/32; nach der Kalibrierung werden **alle vier tatsächlichen Arme**
mit den eingefrorenen Koeffizienten vollständig geprüft:10 Warmup- und30 gemessene
Optimizer-Updates, echte Multicrop-Daten, Accumulation, alle Losses, Backward,
AdamW-Momente, EMA, Center, Checkpoint/Export und Restore. Maximal90% reserviertes
VRAM; größte Batch ist nicht automatisch die Empfehlung.

Passt irgendein tatsächlicher Arm nicht oder schlägt Numerik/Resume/Export fehl,
startet kein wissenschaftlicher Lauf. Kein stiller Fallback nach der Kalibrierung.
Die Sonden/Kandidaten laufen in separaten Prozessen; ihre Gewichte werden verworfen.
Kalibrierungsgradienten benötigen temporär zusätzlichen Backbone-Gradientenspeicher;
auch dort führt OOM zum Abbruch, nicht zu einer unangekündigten Rezeptänderung.

**Ein einzelner Befehl für die komplette Studie**, sobald die Voraussetzungen
erfüllt und die Checks geprüft sind:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sem_cellstyle_regularization_grid.sh \
  --stage run --config configs/sem_ssl/sem_cellstyle_vitl14_ready.yaml \
  --output outputs/sem_ssl_regularization/v1_seed42
```

`run` führt fehlende Checks/Kalibrierung zuerst aus und trainiert dann A/B/C/D
nacheinander. Derselbe Befehl setzt eine unterbrochene Studie fort. Fertige
Trainingsarme werden nach Vertragsprüfung übersprungen, unterbrochene über den
letzten vollständigen Snapshot fortgesetzt. Bestehende gültige Benchmark-
Kandidaten werden wiederverwendet; fehlende Kandidaten starten frisch.
Downstream-Runner verwenden ihren vorhandenen `--resume`-Pfad. Alte Ausgabeordner
werden nicht überschrieben. Originalreferenzen sind Teil des vollständigen
Modellzustands und werden beim Resume gegen die ursprünglichen Gewichte geprüft.

## Auswertung und Dateien

Alle wissenschaftlichen Meilensteine bleiben0/1/5/10/25/50/100. In der ersten
Studienstufe werden0/1/5/10 **mit allen sechs vorhandenen Probes** ausgewertet;
das erweitert ausschließlich den Umfang der Zwischen-Validation gegenüber den
bisherigen Quick-Checks. Training, Preprocessing, Splits, Heads und Metriken
bleiben gleich. NFFA A/B LogReg+MLP mit8 Workern, Aachen A3 Linear+CNN8 mit16.
Original-Large-Pass0 wird nur einmal ausgewertet; alle adaptierten Arme erhalten
getrennte Evaluationsordner. `--verified-data-root <vorhandener Encoderordner>`
kann vorhandene verifizierte Downstream-Datenmetadaten wiederverwenden.

Nach Training erfolgt für alle Arme eine **separate** Diagnose auf einem
gemeinsamen temporären Batch von Trainingsansichten: exportierter eingefrorener
Teacher gegen Original-Large, CLS und Patch separat. Diese Diagnose ist keine
zusätzliche Verzweigung des deaktivierten Trainings und kein Cache augmentierter
Trainingsbilder. Sie wird nicht zur Koeffizientenwahl verwendet.

- `configs/{A,B,C,D}.yaml`: tatsächliche feste Arbeitskonfigurationen.
- `checks/{reference_D,A,B,C,D}/`: Benchmark-CSV/JSON, Durchsatz-/Speicherplots.
- `calibration.json`: gemeinsame vor der Evaluation eingefrorene Koeffizienten.
- `runs/{A,B,C,D}/`: vollständige unabhängige Trainingsläufe, Checkpoints und Logs.
- `verified/`: Integritätsnachweise der Meilensteine.
- `evaluation/original_large`, `evaluation/{A,B,C,D}`: getrennte Validation-Artefakte.
- `diagnostics.json`: gemeinsame Trainingsansichten, Teacher-CLS-/Patchähnlichkeit.
- `report/<Identität>/comparison.{md,json,csv}`: absolute Metriken und Differenzen
  gegen Original-Large sowie A am gleichen Pass. MA-IoU, mIoU, Precision und Recall
  für fixe0,5- und validationgewählte Schwellen getrennt.
- `validation_vs_pass.png`, `drift_and_feature_similarity.png`: gemeinsame Plots.
- `logs/`: separate Prozessprotokolle für Checks, Kalibrierung, Training und Evaluation.

Der Bericht liest Student-Parameterdrift blockweise aus den vollständigen
Snapshots, auch für A, ohne A einen zusätzlichen Trainingsencoder aufzubürden.
`torch.load(..., mmap=True)` lädt dabei nur angesprochene Backbone-Tensoren,
nicht die AdamW-Arrays. Trainingslogs unterscheiden SSL, rohe/gewichtete
Regularisierung, Student-CLS-/Patchähnlichkeit und post-update Blockdrift.
Runtime aus Updatezeiten schließt Checkpoint-/Evaluations-I/O ausdrücklich aus;
Peak-VRAM und Benchmark-Reserved-VRAM werden separat berichtet. Noch nicht
existierende spätere Meilensteine bleiben als fehlend sichtbar.

Bericht ohne Training/Evaluation neu erzeugen:

```bash
bash scripts/run_sem_cellstyle_regularization_grid.sh \
  --stage report --output outputs/sem_ssl_regularization/v1_seed42
```

Kein automatisches Modellranking aus SSL-Loss und keine Vermischung mit alten
Small-Ergebnissen. Qualitätsaussagen beziehen sich auf Validation und zunächst
einen Seed. Vielversprechende Verbesserungen benötigen spätere Wiederholungen.

## Geänderte Implementierung und Tests

- `src/sem_cellstyle/regularization.py`: Anker, exaktes gestreamtes SP-Gradient,
  unmaskierter Featurezweig, unveränderliche Referenz und Gradientensonden.
- `training.py`, `runner.py`: kleine optionale Hooks im vorhandenen Updatepfad,
  Initialisierung und Resume; unveränderte Meta-Loss-Aufrufe.
- `regularization_study.py`: gemeinsame Rezeptidentität, feste vier Schalter,
  eine trainingsbasierte Kalibrierung und deren Freigabeprüfung.
- `regularization_report.py`: zwei Vergleichsreferenzen, Blockdrift, separate
  Trainingsbilddiagnostik und gemeinsame Berichte/Plots.
- `scripts/run_sem_cellstyle_regularization_grid.{sh,py}`: ein manueller Einstieg;
  nutzt den bestehenden Prozesslogger und bestehende Training/Evaluation.
- Bestehender Large-Benchmark: optionale Fortsetzung einzelner fertiger Kandidaten.
- Bestehende Large-Evaluation: opt-in `--full` für alle Heads auch bei1/5 Pässen.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XFORMERS_DISABLED=1 \
python -m pytest -q
```

CPU-Fixtures prüfen Formel/Normierung/Gradienten, Head-Ausschluss, unveränderliche
Referenzen, korrekte unmaskierte Zuordnung, Student-Gradienten und Modi,
Accumulation, einmaliges EMA/Center/Optimizer, bitgenaue OFF-Parität,
Checkpoint/Resume jeder aktivierten Kombination, gemeinsame Kalibrierung,
Batch-/Koeffizientenbindung, Ablaufsteuerung und beide Vergleichsdifferenzen.
Echte Large-Kalibrierungswerte und A100-Speicher-/Stabilitätswerte sind weiterhin
offen, solange Korpus und Originalgewichte fehlen. Die technischen Tests sind
keine wissenschaftlichen GPU-Ergebnisse.

Prüfnachweis18.11.2026: vollständige CPU-Suite **266 bestanden,3 übersprungen,
3 Subtests bestanden**. Siehe pytest_full.txt
und den datensatzfreien Plan.
Der zusätzliche Berichtstest verwendet ausschließlich ausdrücklich synthetische
Metriken, keine behaupteten Large-Experimentergebnisse.

Die21 Regularisierungstests wurden anschließend im finalen Stand erneut
ausgeführt, einschließlich des zusätzlichen CSV-/Plot-Berichtstests:
pytest_regularization.txt.
