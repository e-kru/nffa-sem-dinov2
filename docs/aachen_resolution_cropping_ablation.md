# Aachen-Heerlen: Resolution und Cropping mit eingefrorenem DINOv2

## Fragestellung und Versuchsmatrix

Steigert eine feinere lokale Abtastung den MA-Recall, besonders bei kleinen
Inseln? Ist die Verarbeitung des ganzen Bildfeldes bei ähnlicher lokaler
Abtastung günstiger als eine Verarbeitung einzelner Quadranten?

| Variante | Inhalt pro Original | Encoder-Eingabe W×H | Gitter H×W pro Ausschnitt | Tokens pro Original | Heads |
|---|---|---|---|---|---|
| A1 | vollständiges Bild | 448×336 | 24×32 | 768 | Linear, CNN8 |
| A2 | vollständiges Bild | 672×504 | 36×48 | 1728 | Linear, CNN8 |
| A3 | vollständiges Bild | 896×672 | 48×64 | 3072 | Linear, CNN8 |
| B1 | vier Quadranten | je 448×336 | je 24×32 | 3072 | Linear, CNN8 |

Das ergibt acht Konfigurationen je Seed, mit 42/43/44 insgesamt 24 Trainingsläufe.
Features werden einmal pro Geometrie extrahiert und zwischen Heads und Seeds geteilt.
Hypothesen sind vor den Messungen festzulegen; eine Verbesserung wird nicht vorausgesetzt.

## Kontrollierte Bedingungen

- DINOv2 ViT-S/14, 384 Kanäle; finaler normalisierter Patch-Layer über
  `get_intermediate_layers(n=1, reshape=True, norm=True)[0]`.
- Encoder eingefroren, `eval()` und `inference_mode()` bei der Extraktion;
  volle Feature-Gitter bleiben erhalten. Identische Gewichte werden über einen
  SHA-256 des tatsächlichen State-Dicts überprüft. Wissenschaftliche Caches
  speichern zusätzlich einen gemeinsamen Backbone-Snapshot unter `backbones/`.
- Linear: vorhandener 1×1-Conv-Head, LR 1e-3. CNN8: unveränderte `DeepCNN(384, 8)`
  mit 128 Kanälen, vier Residualblöcken, GroupNorm, GELU und Dropout2d 0.10;
  LR 3e-4. `SpatialResidual` und `DeepCNN` wurden aus dem alten Vergleichsskript
  in `src/downstream/heads.py` verschoben und dort wieder importiert.
  Parameter-Namen, Initialisierung und Architektur bleiben kompatibel.
- AdamW, Weight Decay 1e-4, kein Scheduler, FP32-Training, maximal 30 Epochen,
  Patience 8. Diese Werte entsprechen den bisherigen Linear-/CNN8-Skripten.
- Checkpoint-Auswahl nach Validation-mIoU bei 0.5, bei Gleichstand niedrigere CE,
  jeweils Toleranz 1e-8 wie im bisherigen Trainer.
- Effektive Batchgröße 8 **Originalbilder**. Default Microbatch 8 (bei CUDA-OOM schrittweise reduziert); Gradienten
  werden für alle Originale der effektiven Batch gesammelt. Die letzte Batch
  erhält ihre tatsächliche Anzahl als Nenner. Schritte je Epoche:
  `ceil(N_train / batch_size)`, unabhängig von der Anzahl Quadranten.
- B1: ein einziger Decoder; alle vier Quadranten werden gemeinsam vorwärts
  gerechnet. Die Summe der gültigen Pixelverluste aller Quadranten wird durch
  die Anzahl gültiger Pixel des Originals geteilt. Danach Mittel über Originale.
  Auch ungleich große/teilweise ignorierte Quadranten erhalten somit korrekte Gewichte.
- Seeds initialisieren Decoder, Shuffle und Dropout; die Originalbild-Reihenfolge
  bleibt epochenweise deterministisch und entspricht dem vorherigen Sampler.
  Standard ist jetzt `--determinism off`: CUDA-Determinismus und deterministische
  cuDNN-Ersatzalgorithmen sind deaktiviert, Seeds bleiben gesetzt. Dadurch wird
  der langsame deterministische bilineare CUDA-Upsampling-Pfad vermieden.
  `warn` und `strict` bleiben explizite Optionen. Bitidentische CUDA-Läufe werden
  mit `off` nicht garantiert; größere Microbatches können außerdem die Zuordnung
  stochastischer Dropout-Masken und die Gleitkomma-Summationsreihenfolge ändern.
  Architektur, Dropout-Wahrscheinlichkeit, Loss, effektive Batchgröße und
  Optimierungsschritte ändern sich dadurch nicht. Flags werden nach dem Lauf
  auch im Fehlerfall wiederhergestellt. Policy und Ausführungsversion stehen in
  `config.json`; Reports trennen alte und optimierte Trainingsprotokolle.
- Gleiches unverändertes Group-Manifest. Identitäten, Bild-/Maskenpfade und
  Gruppen werden über **alle Manifestzeilen** auf Duplikate/Leakage geprüft.
  Nur Train-/Val-Dateien werden geöffnet. Es gibt keinen Test-CLI-Schalter.

## Originalgeometrie und neue Referenz

Bild und Maske kommen direkt aus den im Manifest angegebenen Originaldateien.
Der Preflight erfasst tatsächliche Originalgrößen, prüft Maskenausrichtung,
Klassen 0/1/255 und Seitenverhältnisse und hasht die Bild- und Maskeninhalte.
Eine Datenherkunft aus dem originalen PNG-Archiv muss bei einem fremden Manifest
zusätzlich organisatorisch sichergestellt werden; ein Hash beweist keine Herkunft.

Bei 4:3 wird das vollständige Bild ohne Cropping direkt bilinear verkleinert.
Bei abweichendem Seitenverhältnis wird der gesamte Inhalt proportional eingepasst
und schwarz gepolstert (Letterbox), statt das Bild zu strecken. Ganzzahlige
Rundung kann höchstens einen Pixel Größenabweichung erzeugen. Anzahl gepolsterter
Regionen und Originalgrößen stehen im Preflight. Es gibt keinen SEM-Metadaten-
oder Center-Crop und keinen Umweg über bereits verkleinerte Bilder.

B1 trennt bei `x=W//2`, `y=H//2` in links oben, rechts oben, links unten,
rechts unten. Rechte und untere Regionen erhalten die Restpixel. Koordinaten
sind halb offen, ohne Überlappung und vollständig. Masken werden niemals
interpoliert oder verkleinert. Für Training und Validation werden Gitter-Logits
bilinear (`align_corners=False`) direkt auf das Originalraster projiziert.
Bei Letterbox wird zuerst auf das Eingaberaster projiziert, das Padding entfernt
und dann die Originalregion rekonstruiert. B1 setzt diese Originalregionen zum
vollständigen Bild zusammen. Erst danach folgen Sigmoid der Logit-Differenz
und Thresholding. Ignore 255 zählt weder zu CE noch Metriken.

**A1 ist eine neue Referenz.** Die bisherigen Caches/Trainer verwenden auf
448×336 verkleinerte Trainings- und Validationsmasken. Kleine Inseln können dabei
verschwinden. Die historischen 39,78 % MA-IoU und 66,21 % mIoU sind deshalb nicht
exakt vergleichbar und wurden mit dieser Implementierung nicht reproduziert.
Der neue Loss gewichtet Originalbilder gleich; der alte Trainer mittelt über
Pixel der verkleinerten Batch. Die alten Skripte und Caches bleiben erhalten.

## Interpretation von A3 gegenüber B1

Beide verarbeiten 3072 Patch-Tokens pro Original und haben bei geraden 4:3-
Originalen die gleiche lokale Abtastung. B1 begrenzt Attention auf Quadranten;
Kachelgrenzen, Positionsembeddings und Encoder-Rechenkosten unterscheiden sich.
Auch die effektive räumliche Reichweite des CNN8 kann sich zwischen Geometrien
unterscheiden. Bei ungeraden Größen/Letterboxing ist die lokale Abtastung nur
annähernd gleich. Der Vergleich misst gesamte Verarbeitungsstrategien,
keinen isolierten kausalen Kontexteffekt. Gleiche Tokenzahl bedeutet wegen
Attention auch keine gleiche Laufzeit oder gleiche Encoder-Speichernutzung.

## Metriken, Komponenten und Reports

Primär: globale Validation-MA-IoU (nicht Mittel der Bild-IoUs). Zusätzlich
Background-IoU, mIoU, MA-Precision/-Recall, Confusion Matrix (Wahrheit in Zeilen),
Best Epoch, Schwelle, Tokens, Laufzeit und GPU-Speicher. Leere Klassen erhalten
wie bisher 0 statt NaN. Schwellen 0.05 bis 0.95, Schritt 0.05; Auswahl nach
Validation-MA-IoU, Gleichstand zur 0.5 nächsten Schwelle, verbleibender
Gleichstand zur niedrigeren Schwelle. Das ist eine Validationsoptimierung;
ihre Ergebnisse sind keine unabhängigen Generalisierungsschätzungen.

Alle Metriken und Inselanalysen werden für 0.5 und die ausgewählte Schwelle
aufgeschrieben. Inseln sind 8-zusammenhängende Komponenten der originalen
Train-/Val-Masken. Größenklassen stammen ausschließlich aus den 1/3- und
2/3-Quantilen der **Train-Komponentengrößen in Pixeln**: klein ≤ Grenze 1,
mittel ≤ Grenze 2, sonst groß. Gleiche Quantile können leere Klassen erzeugen;
diese erhalten Anzahl 0 und Metriken null, ohne Grenzen auf Val anzupassen.
Gezählt werden Inselanzahl, mittlerer Insel-Pixel-Recall, pixelgewichteter Recall
und Anteil mit mindestens 25 % erkanntem Inselinhalt. Inseln werden vor dem
Cropping auf der gesamten Maske bestimmt; Grenzen teilen keine GT-Inselzählung.

`component_metrics.csv` und `per_image_metrics.csv` enthalten Detailwerte.
Reports speichern Tabellen, Seed-Mittel und Stichprobenstandardabweichung
(bei einem Seed 0), Auflösungs-/Strategie-/Insel-/Laufzeit-Plots als PNG und PDF.
Es werden feste und ausgewählte Schwellen getrennt dargestellt. Beispiele
verwenden gemeinsame Val-IDs, RNG 42, pro Modell den kleinsten verfügbaren Seed
und immer Threshold 0.5. Reports kennzeichnen fehlende Konfigurationen.
Unterschiedliche Daten-/Trainingsprotokolle werden in getrennten Kohorten
aggregiert; doppelte Variante/Head/Seed innerhalb einer Kohorte wird abgelehnt.
Lernratenversuche sollten eigene Output-Roots erhalten.

Die epochale Validation arbeitet standardmäßig mit acht Originalen pro Batch.
GPU-Konfusionen und Loss-Summen bleiben bis zum Phasenende auf dem Gerät;
Komponenten-/Per-Image-Details übertragen erst bei der separaten Auswertung
batchweise Wahrscheinlichkeiten auf die CPU. CUDA-Events messen den Decoder,
Loss und Metriken werden dabei ausgeschlossen. `cached_decoder_seconds_per_original`
bezieht sich jetzt auf gebatchte Validation (`cached_decoder_batch_size`), ohne
Encoder oder Plattenzugriff. Nur die separate Ende-zu-Ende-Inferenzmessung nutzt
weiterhin ein Original und einen ausgeschlossenen Warm-up.
`end_to_end_seconds_per_original` misst Preprocessing, Transfer, eingefrorenen
Encoder, Decoder und Rekonstruktion für standardmäßig die ersten fünf Val-
Originale; Original-RGB liegt bereits im RAM. Die Zeit-/Qualitäts-Plots verwenden
die zweite Messung. Peak-Memory ist `torch.cuda.max_memory_allocated`, also
zugewiesener Torch-Speicher, nicht der gesamte reservierte Geräte-/Prozessspeicher.
Train-, Cache-, Evaluations- und Timing-Peaks werden getrennt protokolliert.

## Caches, Artefakte und Wiederaufnahme

Standardroot: `outputs/aachen_heerlen/resolution_cropping_v1`.

- `preflight_<Datenhash>.json`: überprüfte Train-/Val-Daten und Git-Zustand.
- `caches/<Variante>_<Konfigurationshash>/cache_metadata.json` und eine NPY-Datei
  je Original, Form `[Q,384,Hgrid,Wgrid]`, Q=1 oder 4. Keine Kopie pro Head/Seed.
- `backbones/`: ein unveränderlicher Encoder-State-Snapshot je Gewichtehash.
- `runs/<Variante>/<Head>/seed<Seed>_<Konfigurationshash>/`: `config.json`,
  `cache_metadata.json`, `history.csv`, `best_model.pt`, `last_state.pt`,
  `training_complete.json`, `validation_metrics.json`, `threshold_metrics.csv`,
  `component_size_bins.json`, `component_metrics.csv`, `per_image_metrics.csv`,
  `run.log`.
- `reports/<Kohorte>_<Ergebnishash>_examples<N>/`: unveränderliche Report-Snapshots.

Metadaten enthalten Schema/Preprocessing-Version, Geometrie, Original-IDs,
Pfade/Hashes, Crop- und Input-Inhaltskoordinaten, Manifest-Hash, Encoder und
Checkpoint-Hash, tatsächlichen Encodergewichtehash, Feature-Form/-Dtype,
Seed, Torch-Version, Quellcode-Hashes und Git-Commit/Status. Run-Metadaten
referenzieren vollständige Cache-Metadaten. Cache-Laden prüft Vertrag,
Identität/Koordinaten, Tensorform, Dtype, endliche Werte und Datei-Prüfsummen.
Historische 448er-Baseline-Caches haben ein anderes Schema und werden abgelehnt.
Neue Cache-Verträge trennen vollständige Provenienz (`sources`) von für Features
relevanten Quellcode-Hashes (`feature_sources`: Geometrie, Backbone, Bildladen,
Extraktionsfunktionen). Änderungen allein am Trainer/Reporting invalidieren die
Features nicht mehr. Für vorhandene A/B-Caches aus dem geprüften Commit
`f3a90dfb13a03267d10fc458299bb23b1d9bd8e2` gibt es einen eng begrenzten Adapter:
sowohl der vollständige alte Quellcode-Hash-Satz als auch die geprüfte neue
Extraktionspipeline müssen übereinstimmen. Unbekannte alte Quellen, veränderte
Daten/Masken, Layer, Checkpoints oder Geometrien bleiben inkompatibel.
`--cache-root` sucht vorhandene kompatible Caches; sie werden weder umbenannt
noch migriert. Alle Tensor-/Datei-Prüfsummen werden weiterhin geprüft, jetzt nur
einmal je Variante und Prozess, statt nochmals für jeden Dataset-Adapter.
Trainingsparameter und die aktuelle Trainer-Provenienz erzeugen neue Run-IDs.


Abgeschlossene passende Stufen werden übersprungen. `--resume` setzt
unterbrochene Extraktionen nach atomar publizierten Originaldateien und Training
am letzten vollständigen Epochencheckpoint fort (Optimizer und RNG inklusive).
Eine angebrochene Epoche wird wiederholt. Der atomare letzte Checkpoint enthält
auch den konsistenten Best-Checkpoint; History/Best werden daraus wiederhergestellt.
Unbekannte oder inkompatible Ausgaben werden abgelehnt. Keine parallelen Writer
für denselben Run/Cache starten. Verschiedene Seeds dürfen mit vorbereiteten
Caches und getrennten Prozessen/GPU-Geräten laufen. Der Runner selbst läuft seriell.

## Hardware und Speicher

Standard ist ein CUDA-Gerät (`cuda:0`, empfohlen NVIDIA A100 80 GB) und acht
CPU-Worker (`--workers 8`) für Extraktion, Training und Validation. Worker
verwenden `spawn` und greifen nicht auf CUDA zu. CPU-Smoke-Tests setzen die
Workerzahl auf 0; ein zusätzlicher synthetischer Unit-Test prüft paralleles
Laden mit zwei Workern, sofern lokale IPC-Sockets erlaubt sind. Die hiesige
Sandbox verbietet diese Sockets; dieser zusätzliche Test wird deshalb explizit
übersprungen. Loader haben ein Timeout von 120 Sekunden, damit IPC-/Workerfehler
keine unbegrenzten Wartezeiten verursachen. Der Mehrprozesspfad muss auf dem
KIT-Server noch geprüft werden. Keine Annahme, dass acht
GPUs frei sind. Cache-Extraktion ist FP32; `--cache-dtype float16` komprimiert nur
die gespeicherten Features. Für einen fairen Vergleich dieselbe Cache-Dtype für
alle Varianten verwenden. `--micro-batch-size` begrenzt Decoder-Aktivierungen,
`--extraction-batch-size` die Crop-Batch des Encoders. Kein AMP oder DDP nötig.

Der vorhandene Split hat 1199 Train- und 258 Val-Originale. Für zusammen 1457
Originale ergeben sich ohne NPY-/Metadaten-Overhead:

| Cache | FP32 GiB | FP16 GiB |
|---|---:|---:|
| A1 | 1.601 | 0.800 |
| A2 | 3.602 | 1.801 |
| A3 | 6.403 | 3.201 |
| B1 | 6.403 | 3.201 |
| Gesamt | **18.008** | **9.004** |

Hinzu kommen etwa 84 MiB für einen ViT-S-Encoder-Snapshot sowie Run-Checkpoints,
Optimizerzustände und Reports. Die Caches wachsen nicht mit der Anzahl Seeds.
`--dry-run` prüft Daten und zeigt konkrete Cachepfade, Runstruktur, Cachebedarf
und grobe Tensor-/Aktivierungsschätzungen; es schreibt nichts und lädt keinen
Encoder. Die Aktivierungsschätzung ist kein GPU-Spitzenwert; Attention,
Autograd, temporäre Tensoren, Allocator und Framework kommen hinzu.

## Direkt ausführbare Kommandos auf dem KIT-Server

Die folgenden Kommandos werden vom Benutzer manuell auf dem KIT-Server
ausgeführt. Die Implementierung verbindet sich nicht per SSH und führt keine
Remote-Kommandos aus. Im Repository ausführen; `python` muss auf das
PyTorch-CUDA-Environment zeigen:

```bash
cd /path/to/nffa-sem-dinov2
conda activate sem-gpu
python -m pip install -r requirements-dev.txt
export MPLCONFIGDIR=/tmp/aachen-ablation-matplotlib
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

1. Preflight (keine GPU) und optional Vorschau aller Läufe:

```bash
python -m scripts.run_aachen_ablation --experiment both --stage preflight
python -m scripts.run_aachen_ablation --experiment both --seeds 42 43 44 --dry-run
```

2. Alle vier Feature-Caches auf einer zugewiesenen CUDA-GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_aachen_ablation --experiment both --stage extract --device cuda:0 --workers 8 --resume
```

3. A1–A3 mit beiden Heads und Seed 42 trainieren:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_aachen_exp_a --stage train --device cuda:0 --workers 8 --seeds 42 --resume
```

4. B1 mit beiden Heads und Seed 42 trainieren:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_aachen_exp_b --stage train --device cuda:0 --workers 8 --seeds 42 --resume
```

5. Alle acht Konfigurationen mit allen drei Seeds, einschließlich Validation/Report:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_aachen_ablation --experiment both --stage all --seeds 42 43 44 --device cuda:0 --workers 8 --resume
```

6. Validation aller vorbereiteten Runs, danach Tabellen/Plots separat:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_aachen_ablation --experiment both --stage evaluate --seeds 42 43 44 --device cuda:0 --workers 8
python -m scripts.report_aachen_ablation --output-root outputs/aachen_heerlen/resolution_cropping_v1 --example-count 3
```

Für getrenntes Training/Evaluieren müssen dieselben Trainings-/Cacheparameter
angegeben werden, damit der identische Runhash gefunden wird (z.B. `--epochs`,
`--batch-size`, Lernraten, Dtype, Seeds, `--device`). Nach den Schritten 3/4
zunächst nur `--seeds 42` evaluieren; 43/44 sind erst nach Schritt 5 vorhanden.
`--variants A3`, `--heads cnn8`, `--encoder-checkpoint /path/backbone.pt`,
`--output-root /path/new-study`, `--workers 8` sind zusätzlich verfügbar.
Eine lokal archivierte Backbone-Datei ist für langfristig geplante Studien
empfohlen. Bei einem zugewiesenen Gerät stets dessen CUDA-Sichtbarkeit verwenden.

## CPU-Verifikation ohne DINOv2-Download

```bash
python -m pytest -q
python -m flake8 src/aachen_ablation scripts/run_aachen_ablation.py scripts/run_aachen_exp_a.py scripts/run_aachen_exp_b.py scripts/report_aachen_ablation.py tests/test_aachen_ablation.py --select E9,F63,F7,F82
python -m pyflakes src/aachen_ablation scripts/run_aachen_ablation.py scripts/run_aachen_exp_a.py scripts/run_aachen_exp_b.py scripts/report_aachen_ablation.py tests/test_aachen_ablation.py
python -m scripts.run_aachen_ablation --smoke-test --device cpu --epochs 1 --stage all --resume
```

Flake8 (inklusive Pyflakes) ist in `requirements-dev.txt` enthalten.
Unit-Tests erzeugen eigene synthetische Originalbilder/Masken/Features;
der komplette Acht-Konfigurationen-Test blockiert `torch.hub.load` ausdrücklich.
Der CLI-Smoke-Test nutzt höchstens zwei Train- und zwei Val-Bilder des angegebenen
Manifests mit einem künstlichen Encoder, maximal zwei Epochen, und schreibt nur
unter `NONSCIENTIFIC_SMOKE/`. Bei fehlenden Originaldaten laufen weiterhin die
synthetischen Unit-/Integrationstests. Smoke-Reports sind sichtbar markiert und
liefern keine wissenschaftlichen Resultate.

Die beiden alten Tests mit 224er-Segmentierungserwartungen wurden an das bereits
vorhandene 448×336-Preprocessing angepasst; das Klassifikations-Preprocessing
wird weiterhin gegen die historische V2-Transformation geprüft.

## Tatsächlich ausgeführte lokale Verifikation

- Gesamte Testsuite: **37 bestanden, 1 übersprungen** (`python -m pytest -q -rs`).
  Der übersprungene Test betrifft echte Mehrprozess-Loader: die lokale Sandbox
  verweigert die dafür benötigten Unix-Sockets. Ein gezielter lokaler
  Minimalversuch bestätigte `PermissionError` beim IPC-Socket-Aufbau.
- Vollständiger synthetischer CPU-Integrationstest aller acht Konfigurationen:
  Feature-Caches, Training, Threshold-Auswertung, Komponenten, PNG/PDF-Reports,
  Beispielbilder und erneutes Starten mit Überspringen abgeschlossener Stufen.
  Epochengenaues Resume wurde gegen einen ununterbrochenen Lauf verglichen.
- Flake8 (E9/F63/F7/F82), Pyflakes für die neuen Pipeline-Dateien, Compileall
  für `src`, `scripts`, `tests` und `git diff --check`: erfolgreich.
- Lokaler Originaldaten-Preflight: 1199 Train-/258 Val-Bilder, alle 1024×768,
  passende Masken, keine Letterbox nötig. Testdateien wurden nicht geöffnet.
- Vor der späteren Einschränkung auf ausschließlich synthetische Smoke-Tests
  wurde zusätzlich ein lokaler Mini-Smoke-Test mit zwei Train-/zwei Val-
  Originalbildern, künstlichem Encoder und einer Epoche je Konfiguration
  gestartet und abgeschlossen. Die Ergebnisse sind unter `NONSCIENTIFIC_SMOKE`
  markiert und haben keine wissenschaftliche Aussagekraft.
- Keine SSH-Verbindung und keine Remote-Kommandos. Keine vollständige
  Feature-Extraktion und kein vollständiges Experimenttraining gestartet.
- Nicht verifiziert: echter DINOv2-Forward mit Pretrained-Gewichten, CUDA/A100,
  tatsächliche GPU-Spitzenwerte und Laufzeiten, Mehrprozess-Loader auf dem
  KIT-Server und wissenschaftliche Ergebnisse der 24 geplanten Trainingsläufe.


## Korrektur des Determinismus-Fehlers aus dem KIT-Testlauf

Der gemeldete Fehler in `DownstreamTest.test_cli_cached_and_segmentation`
entstand durch einen globalen Zustand: ein vorheriger Ablationstest hatte
`torch.use_deterministic_algorithms(True)` aktiviert und nicht zurückgesetzt.
Der spätere Legacy-Test lief automatisch auf CUDA, dessen NLL2d-Reduktion im
verwendeten Torch-Build keine strikt deterministische Implementierung hat.
Dies ist kein pytest-Installationsfehler. Die Ablation stellt jetzt alle
veränderten Torch-/cuDNN-Flags bei normaler Rückkehr und nach Ausnahmen wieder
her. Zusätzlich ist die oben beschriebene Policy explizit konfigurierbar.
Es gibt keinen stillen Retry mit einer schwächeren Policy.

Lokaler CPU-Test ohne Netzwerk, Modell-Download oder GPU-Zugriff:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q -rs
```

Historisch verwendeten die KIT-Kommandos `--determinism warn`; seit der
Performance-Korrektur ist der Standard `off`.
für eine strikte Prüfung kann dieser Parameter explizit auf `strict` gesetzt
werden. Die Policy muss für Training und separate Evaluation übereinstimmen,
weil sie zur Run-Konfiguration gehört. Aufgrund der konservativen Quellcode-
Hashes erzeugt diese Codekorrektur neue Cache-/Run-Identitäten; bestehende
Artefakte werden nicht überschrieben oder automatisch migriert.

## CUDA-Geräte und fehlertolerante Speicherstatistik

Die gemeinsame Implementierung liegt in `src/aachen_ablation/devices.py`.
Geräte werden vor Transfers/Modellaufbau auf `torch.device` normalisiert:
`cpu` (auch `cpu:0`), `cuda:N`, `torch.device` und ganzzahlige CUDA-Indizes.
Ein unindiziertes `cuda` wird auf das aktuelle logische Gerät aufgelöst.
Der Runner prüft CUDA-Verfügbarkeit und sichtbare Geräteanzahl und wählt das
Gerät vor Modellaufbau und Statistikzugriff explizit mit `set_device(index)`.
Auch die Run-Konfiguration speichert das kanonische Gerät, z.B. `cuda:0`.
Unter `CUDA_VISIBLE_DEVICES=7` ist dieses logische Gerät 0 die physische GPU 7;
`--device cuda:7` wird bei nur einem sichtbaren Gerät abgelehnt.

Der künstliche Smoke-Encoder hat keine Parameter/Puffer. Sein `.to("cuda:0")`
muss CUDA deshalb noch nicht initialisieren. Im installierten Torch-Code kann
`reset_peak_memory_stats` vor der CUDA-/Allocator-Initialisierung mit
`RuntimeError: Invalid device argument` scheitern. Die Pipeline initialisiert
CUDA nun ausdrücklich und übergibt Statistik- und Synchronisationsfunktionen
kanonische ganzzahlige Geräteindizes. Checkpoint-RNG-Zustände werden nur für das
gewählte Gerät erfasst und wiederhergestellt, ohne andere GPUs zu initialisieren.
Ältere Checkpoints mit einer Liste von CUDA-RNG-Zuständen bleiben lesbar; daraus
wird beim Wiederherstellen nur der Zustand des gewählten Geräts verwendet.

Extraktion, Training, Validation und Inferenz-Timing verwenden denselben
optionalen Speicherprofiler. Fehler beim Initialisieren/Zurücksetzen oder Lesen
**der Speicherstatistik** werden geloggt und führen zu `null` in den vorhandenen
Peak-Feldern. Die zusätzlichen Felder `extraction_memory_profiling`,
`train_memory_profiling`, `eval_memory_profiling` und `timing_memory_profiling`
enthalten `device`, `available` und `errors`. Nach einem fehlgeschlagenen Reset
wird kein alter prozessweiter Peak als neuer Messwert ausgegeben. CPU-Messungen
bleiben `null`. Echte Geräte-, Transfer-, Modell- und Synchronisationsfehler
werden weiterhin weitergereicht; es gibt keinen stillen CPU-Fallback.
`--num-workers` ist zusätzlich als Alias für `--workers` verfügbar.

Lokale Verifikation dieser Korrektur: **87 Tests bestanden, 1 übersprungen**
(blockierte IPC-Sockets); der vorhandene rechteckige Smoke-Test lief zusätzlich
mit synthetischem Bild, FakeDINO und CPU erfolgreich. Flake8-Fehlerprüfungen,
Pyflakes, Compileall und `git diff --check` wurden ausgeführt. Die Regressionen
prüfen Normalisierung, GPU-Sichtbarkeit, Initialisierung vor Statistik-Reset,
Reset-/Lesefehler, fehlende CUDA-Zugriffe bei CPU, gerätespezifische RNG-/Sync-
Aufrufe und erfolgreiche komplette Stufen trotz injizierter Profilingfehler.
CUDA-Aufrufe in diesen Tests sind gemockt; echte A100-Ausführung bleibt manuell
auf dem Server zu prüfen. Es gab keine SSH-/Remote-Ausführung und keine
vollständigen Experimentläufe.

Nach Push/Pull kann der GPU-Extraktions-Smoke-Test manuell erneut gestartet werden:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_aachen_ablation --experiment A --stage extract --smoke-test --device cuda:0 --workers 8 --resume
```

Smoke-Tests verwenden weiterhin null CPU-Worker; reguläre Läufe standardmäßig
acht. Neue Quellcode-Hashes erzeugen neue Cache-Identitäten; alte unvollständige
Caches werden nicht überschrieben. Keine Cache-Dateien manuell umbenennen, um
die Konsistenzprüfung zu umgehen.

## Performance-Korrektur: batched-v2

### Gemessener Vergleich und Grenzen

Der Benchmark `scripts/benchmark_aachen_ablation.py` liest die vorherigen
Engine-/Dataset-/Bildlade-Implementierungen direkt aus Git-Commit
`f3a90dfb13a03267d10fc458299bb23b1d9bd8e2`. Mit `--reference-repository` lässt sich
ein lokaler Checkout mit diesem Commit angeben. Der Benchmark erzeugt synthetische
RGB-Bilder, Originalmasken und künstliche Features; DINOv2 und echte Daten werden
nicht geladen. Gemessen wurden 16 Train- und 8 Val-Originale mit 1024×768-Masken,
A1-Features 384×24×32, FP32, AdamW, effektive Batchgröße 8, jeweils eine Warm-up-
Epoche und danach zwei gemessene Epochen. Eine CPU-Rechen-Thread, Workerzahl 0
wegen der lokalen IPC-Sperre, Torch 2.13.0+cu126, **CUDA nicht verfügbar**.
Beide Messungen enthalten den cProfile-Overhead. Mittelwerte für Training plus
Validation, ohne Cache-Verifikation und Checkpoint-Dateischreiben:

| Head | Train vorher/nachher (s) | Val vorher/nachher (s) | Epoche vorher/nachher (s) | Beschleunigung |
|---|---|---|---|---|
| Linear | 0.607 / 0.437 | 0.266 / 0.205 | 0.873 / 0.642 | 1.36× |
| CNN8 | 1.447 / 1.244 | 0.438 / 0.343 | 1.884 / 1.586 | 1.19× |

Im Linear-Profil kosteten 48 Dataset-Zugriffe über zwei gemessene Epochen
vorher 0.870 s, nachher 0.247 s. Die vorherige Funktion `load_original` belegte
0.850 s von insgesamt 1.746 s Profilzeit; RGB-Konvertieren/-Dekodieren allein
0.593 s. Der optimierte Trainingspfad verwendet nur `load_mask`. Die Anzahl
Backward-Aufrufe sank von 32 auf 4, bei unverändert vier Optimizer-Schritten
über die beiden Epochen. Das quantifiziert CPU-/I/O-Kosten und Batch-Overhead;
es ist **keine Messung der A100-Beschleunigung**. Weder der langsamere
CUDA-Determinismuspfad noch echter persistenter Mehrprozessbetrieb konnten
hier gemessen werden. GPU-Zeiten werden auf CPU als `null` gespeichert.

Rohdaten und vier cProfile-Berichte:
benchmark.json,
Linear vorher,
Linear nachher,
CNN8 vorher,
CNN8 nachher.

### Implementierung und numerische Kontrolle

- `--determinism off` ist Standard. Seeds und Shuffle-Reihenfolge bleiben fest.
  Im Unterschied zu `warn` werden deterministische CUDA-Ersatzimplementierungen
  tatsächlich abgeschaltet. `cudnn.benchmark` bleibt aus; FP32 bleibt erhalten.
  CUDA-Atomics, andere Kernelreihenfolgen und anders gruppierte Dropout-Zufalls-
  zahlen verhindern eine Zusicherung bitidentischer Trainingsverläufe.
- Tatsächliche Microbatchgröße und effektive Batchgröße sind standardmäßig 8.
  CUDA-OOM während Forward/Backward halbiert die Microbatch bis minimal 1.
  Teilgradienten werden gelöscht, der RNG-Zustand vor der Batch wiederhergestellt
  und dieselbe Originalbatch erneut verarbeitet. Es wird erst nach einer
  vollständig erfolgreichen Batch ein Optimizer-Schritt ausgeführt. Das reduzierte
  Maß bleibt für spätere Epochen erhalten und steht in `history.csv`.
- Train-/Val-Loader werden einmal je Run aufgebaut, mit je acht persistenten
  CPU-Workern, gepinnten Batches und Transfers mit `non_blocking=True` auf CUDA.
  Es gibt zwei residente Worker-Pools; im jeweiligen Datenpfad arbeiten acht
  Worker, der andere Pool wartet. Worker setzen Torch-Threads auf 1. Die erste
  Epoche umfasst den einmaligen Spawn-Aufwand; spätere Epochen verwenden dieselben
  Prozesse. Bei Workerzahl 0 wird kein Multiprocessing verwendet.
- Der epochenadressierbare Sampler bildet die bisherige Originalbild-Reihenfolge
  exakt nach, einschließlich des früheren DataLoader-Base-Seed-Verbrauchs. Er
  funktioniert auch nach epochengenauer Wiederaufnahme mit persistenten Workern.
- Masken werden beim gecachten Training direkt in Originalauflösung geladen;
  RGB wird nur beim Preflight zur Geometrie-/Dateiprüfung und bei ausdrücklich
  angeforderter Bildvisualisierung/Ende-zu-Ende-Messung geöffnet.
- Decoder-Forwards, Originalraster-Projektion und CE sind für gleiche Original-
  geometrien gebatcht. Unterschiedliche/ungerade Größen und Letterboxing bleiben
  unterstützt. B1 rekonstruiert dieselben vier Regionen. Die gültigen Pixelverluste
  werden je Original normalisiert, danach wird über die effektive Originalbatch
  gemittelt. Ignore 255, Architektur, LR/AdamW-Regeln, Splits, Thresholds und die
  Checkpoint-Regel (Val-mIoU bei 0.5, CE-Tie-Break mit 1e-8) bleiben erhalten.
- Die Validation nutzt standardmäßig `--validation-batch-size 8`. Die Projektion
  wird für Loss und Metriken geteilt. Pro Epoche gibt es einen Abschluss-Wait je
  Trainings-/Validationsphase statt Synchronisationen pro Bild. Die gesonderte
  Ende-zu-Ende-Inferenzmessung behält ihre Batchgröße 1 und ihren Warm-up.

Kontrollierte Vergleiche mit identischen Gewichten und Inputs deaktivieren
Dropout nur für den Vergleich, lassen aber Gradienten zu. Im gespeicherten
1024×768-Benchmark betrug die maximale Loss-Abweichung 0 (Linear) bzw.
1.79e-7 (CNN8), die maximale Parametergradient-Abweichung 3.73e-9 bzw. 1.01e-7.
Die Confusion Matrices stimmen exakt überein. Weitere Tests prüfen alle vier
Varianten, beide Heads, verschiedene/ungerade Geometrien, vollständig ignorierte
Quadranten, Microbatch-Akkumulation, OOM-Retry, feste Sampler-Reihenfolge, Resume,
Cache-Kompatibilität und Timings. Numerische Toleranzen sind ausdrücklich
FP32-Toleranzen, keine Behauptung bitidentischer Dropout-/CUDA-Trainingsverläufe.

### Neue Instrumentierung

`history.csv` und die Epochenzeilen in `run.log` berichten:

- `epoch_seconds`: Training, Validation und Checkpoint-Publikation; ohne den
  abschließenden History-CSV-/Log-Schreibvorgang.
- `train_seconds`, `validation_seconds`: jeweilige Phasen-Wallzeit, einschließlich
  einmaligem Worker-Start in der ersten Epoche und Geräteabschluss am Phasenende.
- `data_loading_seconds`, `validation_data_loading_seconds`: im Hauptprozess
  tatsächlich auf den Loader gewartete Zeit. Mit Prefetching ist dies nicht die
  gesamte CPU-Arbeit der Worker und kann mit GPU-Arbeit überlappen.
- `gpu_computation_seconds`, `validation_gpu_computation_seconds`: CUDA-Event-Zeit
  für Transfers und Gerätearbeit auf dem gewählten Stream; CPU-Launch-Lücken
  innerhalb der Event-Paare können enthalten sein. Diese Werte sind keine
  GPU-Auslastungsprozente und nicht additiv zu sämtlichen Wallzeit-Komponenten.
  CPU oder nicht unterstützte Event-Messung ergibt `null`/leeres CSV-Feld.
- `host_compute_seconds`: Hostzeit für das Einreichen/Ausführen der Trainingsarbeit;
  auf CUDA kein Ersatz für die asynchrone Event-Zeit.
- `checkpoint_seconds`, `actual_micro_batch_size`, `oom_retries` ergänzen die
  Einordnung. Checkpoint-/Datenverifikation vor der ersten Epoche gehört nicht
  zur epochalen Laufzeit.

### Experiment A sicher mit vorhandenen Caches neu starten

Einen noch laufenden alten Trainingsprozess zuerst selbst regulär mit Ctrl+C
beenden. Die alten Run-Verzeichnisse und deren letzten atomaren Checkpoints
bleiben erhalten. Nach Push/Pull die optimierte Ausführung in einem **neuen**
Output-Root starten; ausschließlich das vorhandene Cache-Root referenzieren:

```bash
cd /path/to/nffa-sem-dinov2
conda activate sem-gpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_aachen_ablation \
  --experiment A --stage train --device cuda:0 --workers 8 \
  --batch-size 8 --micro-batch-size 8 --validation-batch-size 8 \
  --determinism off --seeds 42 \
  --cache-root outputs/aachen_heerlen/resolution_cropping_v1/caches \
  --output-root outputs/aachen_heerlen/resolution_cropping_perf_v2 \
  --resume
```

`--stage train` extrahiert keine Features. Fehlen passende Caches oder stimmen
Daten/Encoder/Geometrie/Prüfsummen nicht überein, wird abgebrochen. Die vier
vorhandenen Cache-Metadatensätze wurden lokal auf Kompatibilität und korrekte
Discovery geprüft, ohne Featuredateien/Metadaten zu ändern. Die vollständigen
Tensor-Prüfsummen werden beim tatsächlichen Trainingsstart geprüft. Der neue
Lauf beginnt mit Seed 42 von vorne; alte Gewichte werden nicht stillschweigend
mit dem geänderten Ausführungsprotokoll forttrainiert. `--resume` betrifft nur
passende neue Läufe, die danach unterbrochen wurden. Für drei Seeds
`--seeds 42 43 44` verwenden. Für die separate Auswertung denselben Befehl mit
`--stage evaluate` ausführen; Output-/Cache-Roots und Trainingsparameter beibehalten.

Der kleine Before-/After-Benchmark kann manuell auf der A100 wiederholt werden,
ohne Backbone-Download oder Originaldaten und ohne vollständiges Experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_aachen_ablation \
  --reference-repository /path/to/reference-checkout \
  --device cuda:0 --workers 8 --mode both --variant A1 \
  --output outputs/aachen_heerlen/benchmarks/a100_batched_v2
```

Der Benchmark-Zielordner muss neu sein. Lokal lief ausschließlich die CPU-
Variante mit synthetischen Daten; der Server wurde nicht kontaktiert.

Abschließende lokale Verifikation von batched-v2: **124 Tests bestanden,
2 übersprungen** (beide benötigen durch die Sandbox blockierte IPC-Sockets).
Der synthetische rechteckige CPU-Smoke-Test bestand ebenfalls. Flake8-
Fehlerprüfungen, Pyflakes, Compileall und `git diff --check` waren erfolgreich.
Das ist keine Verifikation von A100-Laufzeit, tatsächlichen GPU-Peaks oder
realem Worker-PID-Reuse auf dem KIT-Server. Die CUDA-Event-/Fehlerpfade werden
lokal über Mocks geprüft; es wurden keine vollständigen Experimente gestartet.
