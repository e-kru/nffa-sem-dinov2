# SEM SSL: DataLoader-Profiling und verlustfreier Bildcache

Die Loader-Optimierungen sind konfigurierbar. Nach dem A100-Vergleich nutzt
`configs/sem_ssl/nffa_200step.yaml` auf Wunsch des Nutzers Batch 64 und 16 Worker:
direkter PIL-Lesepfad, Prefetch 2, `persistent_workers=False`, gepinnter
CUDA-Hostspeicher und nicht-blockierende Transfers. Ohne `loader`-Block gelten
weiterhin diese Loader-Defaults; Batch und Worker-Zahl kommen aus `train`.
Weder Meta-Code noch Losses, Modell, Augmentationsverteilungen oder Conda-Pakete
werden geändert.

## Befund aus dem vorhandenen A100-Benchmark

Quelle: `outputs/sem_ssl_benchmarks/a100_batch_v1/benchmark.json`; acht Worker,
zehn Warmup-Updates und zwanzig Mess-Updates mit dem vollständigen SSL-Schritt.

| Batch | Originalbilder/s | Loader-Warten, Mittel | GPU-Trainingsintervall, Mittel | Gesamtschritt, Mittel |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 143.69 | 0.119 ms | 111.20 ms | 111.35 ms |
| 32 | 212.00 | 33.43 ms | 118.63 ms | 150.94 ms |
| 64 | 179.21 | 156.68 ms | 201.31 ms | 357.13 ms |

Bei Batch 64 ist das p95 des Loader-Wartens sogar etwa 1.516 s, während der
Median nur etwa 0.154 ms beträgt. Die Wartezeit ist sehr ungleich verteilt.
Allein aus diesen Zahlen lässt sich nicht zwischen Decode, Metadaten-Crop und
Augmentation unterscheiden. Gepinnter Speicher und nicht-blockierende Transfers
waren schon aktiv; sie sind keine neu entdeckte Beschleunigung.

Ohne Daten-Leerzeiten wären anhand dieser GPU-Intervalle grob 270 Bilder/s bei
Batch 32 beziehungsweise 318 bei Batch 64 möglich. Das sind rechnerische
Obergrenzen aus dem alten kurzen Lauf, keine gemessenen neuen Ergebnisse.

## Dateien

| Datei | Aufgabe |
| --- | --- |
| `src/ssl/loading.py` | Loader-Optionen, unveränderte RGB/Crop-Vorverarbeitung, optionale Originaltransform-Timer, Cache-Builder und Read-only-mmap-Reader, CPU-/Speicherabfragen. |
| `src/ssl/data.py` | Nutzt wahlweise Originalbilder oder gecachte deterministische RGB-Pixel und führt anschließend immer die Meta-Augmentation aus. |
| `src/ssl/stream.py` | Erhält deterministische Seeds pro Präsentation/Mask-Batch; optionale Stage-Timings werden nach dem originalen Collate aggregiert. |
| `scripts/cache_sem_ssl_images.py` | CPU-only: Cachebedarf abschätzen, Cache bauen und vollständigen Checksum prüfen. |
| `scripts/profile_sem_ssl_data.py` | CPU-only: echte ausgewählte Train-Bilder, Komponentenprofil, Worker-Sweep und gruppierter RSS/PSS. |
| `scripts/benchmark_sem_ssl_a100.py` / `src/ssl_benchmark.py` | Vorhandener vollständiger SSL-GPU-Benchmark mit neuen Loader-Argumenten, optionalen H2D-Events und Hostspeicherwerten. |
| `scripts/benchmark_sem_ssl_loader.py` | Sequenzielle lange A100-Vergleiche bei B32/B64 mit Ressourcenprüfung und gemeinsamem einmaligen Dataset-Audit. |
| `scripts/train_sem_ssl.py` / `src/ssl/trainer.py` | Dieselben Loader-Optionen für einen später bewusst gestarteten Experiment-Lauf. |
| `tests/test_ssl_loading.py` | Pixel-/Ansichts-/Masken-Parität, Cache-Identität, Budgets, Spawn-Serialisierung, Optionen und Vergleichsaggregation. |

## Konfigurierbare Optionen

Die CLI-Optionen gelten sowohl für `benchmark_sem_ssl_a100.py` als auch für
`train_sem_ssl.py`:

```text
--decoded-cache /node/local/path       # weglassen oder 'none': Originalbilder
--persistent-workers / --no-persistent-workers
--prefetch-factor 2                   # positiver Integer; z.B. 4 zum Vergleich
--pin-memory / --no-pin-memory
--non-blocking / --no-non-blocking
--profile-loader / --no-profile-loader
```

Entsprechender optionaler YAML-Block in einer Kopie der Experiment-Konfiguration:

```yaml
loader:
  decoded_cache: /dev/shm/sem_ssl_nffa_rgb_v1
  persistent_workers: true
  prefetch_factor: 2
  pin_memory: true
  non_blocking: true
  profile: false
```

Worker 0 bekommt keine Multiprocessing-/Prefetch-/Persistent-Argumente. Auf CPU
ist `pin_memory` effektiv aus; seinen CUDA-Effekt misst der A100-Vergleich.
Jeder Worker nutzt einen Torch-Thread; OMP/MKL/OpenBLAS sollten ebenfalls 1 sein.

`persistent_workers` amortisiert insbesondere erneute Iterator-Erzeugung. Der
aktuelle Trainer hält einen Cursor-Iterator über alle Optimizer-Schritte; dort
wird durch dieses Flag allein kein großer Geschwindigkeitsgewinn erwartet.
Prefetch kann ungleich teure Batches abfangen, erhöht aber RAM und eventuell
I/O-Druck. Das ist zu messen, keine pauschale Empfehlung für den größten Wert.

## Was gecacht wird

Genau `crop_metadata_strip(Image.open(original).convert('RGB'))`, als verlustfreie
uint8-Pixel in ursprünglicher gecroppter Auflösung. Keine JPEG-Neukompression,
kein Resize, keine Normalisierung und **keine zufällig augmentierten Ansichten**.
Danach laufen bei jeder Präsentation das ursprüngliche `DataAugmentationDINO`
und die ursprüngliche Maskierung mit denselben Seeds wie im direkten Pfad.
Globale/lokale Crop-Verteilungen, Photometrie und Teacher-/Student-Semantik
bleiben gleich. Die Cache-Tests vergleichen auch die erzeugten Tensoren und
Masken bitweise.

Ein Cache besteht aus `pixels.bin` plus `complete.json`. Das Indexfile bindet
die geordnete Train-Auswahl inklusive ursprünglicher Bild-SHA256 und
Manifest-/Split-Identität sowie den Code-SHA256 der kanonischen Vorverarbeitung.
Andere Daten oder andere Vorverarbeitung werden abgewiesen. Der Builder prüft
die Quellbild-Hashes erneut und schreibt erst nach einem vollständigen Build
atomar den fertigen Ordner. Eine vorhandene Cache-Version wird nie überschrieben.
Teilweise fehlgeschlagene Builds entfernen nur ihr eigenes temporäres Verzeichnis.
`--verify` prüft alle gepackten Pixelbytes gegen den Gesamt-SHA256.

Der Reader öffnet `pixels.bin` read-only als Memory Map. Beim Spawn werden nur
Pfad und kleiner Index übertragen; niemals das gesamte RGB-Korpus pro Worker.
Die zugrunde liegenden Seiten werden vom Betriebssystem geteilt. Pro aktuell
verarbeitetem Bild entsteht weiterhin eine temporäre PIL-RGB-Kopie.

Auf tmpfs (`/dev/shm`, gegebenenfalls auch `/tmp`) belegt das komplette File RAM
einmal. Auf einer lokalen SSD dient derselbe gepackte Cache als mmap-Datei mit
geteiltem, reclaimbarem OS-Page-Cache. Der Builder verweigert NFS/CIFS/SMB-Pfade,
damit nicht tausende Decodes durch zusätzliche geteilte Dateisystem-I/O ersetzt
werden. Dateisystemtyp und freier Speicher werden geprüft. `/tmp` ist nicht auf
jedem Rechner eine SSD; maßgeblich ist der gemeldete tatsächliche Typ.

Der Cache ist **knotenlokal**: Auf A100 bauen, wenn er dort benutzt werden soll.
Das gemeinsame Datenverzeichnis macht `/dev/shm` oder `/tmp` der beiden Rechner nicht
identisch. RAM-Caches müssen nach einem Neustart gegebenenfalls neu gebaut werden.
Ein späterer Resume benötigt weiterhin passende Daten-, Code- und Konfigurations-
Fingerprints; diese Schutzprüfungen werden nicht umgangen.

## Speicherbedarf

Für den geprüften Bestand mit 8.061 Bildern beträgt die Header-basierte
RGB-Obergrenze vor Metadata-Crop **19.980.754.251 Bytes ≈18,61 GiB**.
Die tatsächliche Größe nach Crop wird beim Build gemessen. Der 128-Bilder-
Diagnosecache belegt **266.339.328 Bytes ≈254 MiB**.

Der Builder benötigt eine explizite Grenze (`--max-cache-gb`, Einheit GiB),
genügend Dateisystemplatz und bei tmpfs zusätzlich eine konservative RAM-Reserve:
Der obere Cachebedarf darf höchstens 60 % des verfügbaren RAMs belegen.
Damit wird ein voller RAM-Cache auf der aktuellen 32-GB-Workstation mit nur
ungefähr 15 GB freiem RAM abgewiesen. Auf A100 erst die Abschätzung prüfen;
für den vollständigen RAM-Cache plus Loader-Queues ungefähr 40 GB freien
Host-RAM oder mehr einplanen. Andernfalls eine geeignete lokale SSD nutzen.

Unabhängig vom Bildcache benötigt ein augmentierter B64-Batch ungefähr 130 MiB
für die float32-Ansichten. `workers * prefetch_factor` kann viele solcher Batches
vorhalten. Acht Worker/Prefetch 2 ergeben grob 2 GiB reine Batch-Tensoren;
16 Worker/Prefetch 4 etwa 8 GiB. Übergangsweise gepinnte und pageable Kopien,
Interpreter, temporäre PIL-Bilder und weitere Tensoren kommen hinzu. Der
Sweep prüft konservativ diese Queue-Kapazität und die verfügbaren CPUs.

RSS summiert geteilte Seiten mehrfach. Für Vorher/Nachher wird deshalb auch
die Summe des PSS der Eltern-/Worker-Prozesse nach dem Messfenster erfasst.
Das ist ein Snapshot, kein exakt gemessener maximaler Hostspeicherverbrauch.
Cache-Filegröße, PSS und MemAvailable sind unterschiedliche Größen; sie dürfen
nicht blind addiert werden, weil mmap-Seiten teilweise schon im PSS enthalten sind.

## Profiling-Methode

Optional werden Dateilesen, Decode/RGB, kanonischer Metadata-Crop, globale/lokale
Geometrie, Farbe, Blur/Solarization, Tensor-/ImageNet-Normalisierung und Collate
getrennt zeitlich erfasst. Die Stage-Wrapper rufen die originalen Meta-Objekte
in derselben Reihenfolge auf und verbrauchen keine Zufallszahlen.

Zum Trennen von Lesen und lazy PIL-Decodieren verwendet ausschließlich der
diagnostische Rohbildpfad `read_bytes` + `BytesIO`. Pixelidentität zum normalen
PIL-Pfad ist getestet. Zeiten sind Worker-Wanduhr-Servicekosten; parallel
ausgeführte Worker-Zeiten sind nicht identisch mit globaler Batch-Wartezeit.
CPU-Servicekosten der Komponenten sind deshalb keine direkte Addition zum
GPU-Schritt. Der originale Collate wird inklusive Seed-Kontext und kleiner
Statistikaggregation gemessen.

Mit `--profile-loader` erfasst der A100-Benchmark außerdem CUDA-Events um den
Transferabschnitt im vorhandenen Adapter. Die Events werden erst nach dem
Fenster-Sync gelesen, ohne zusätzliche Synchronisation pro Update. Das Intervall
kann auch nicht verdeckte Host-Dispatch-Lücken enthalten. Absichtlich blockierende
Transfers (`--no-non-blocking`) können natürlich intrinsisch synchronisieren.

Für die maßgebliche Durchsatzbewertung bleibt Instrumentierung aus. Der
Vergleichs-Sweep erzwingt `--no-profile-loader`; ein eigener Diagnose-Aufruf
liefert die Komponenten-/H2D-Zeiten. Alle GPU-Fälle messen weiterhin Forward,
originale Losses, Backward, Clipping, AdamW und Teacher-EMA einschließlich
DataLoader-Warten. Es wird kein Encoder-Inference-Benchmark daraus gemacht.

## Exakte A100-Befehle

```bash
cd /path/to/nffa-sem-dinov2
source /path/to/workspace/apps/miniconda3/etc/profile.d/conda.sh
conda activate sem-gpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset CUDA_LAUNCH_BLOCKING
export SEM_SSL_INVENTORY=outputs/sem_ssl_benchmarks/a100_batch_v1/dataset_inventory.json
export SEM_SSL_CACHE=/dev/shm/sem_ssl_nffa_rgb_v1
```

Inventar aus dem bestehenden Benchmark nur bei unverändertem Korpus verwenden.
Manifest, Split-Hashes und konservative Train-Auswahl werden erneut geprüft;
es wird nicht bei jeder Loader-Ablation erneut der gesamte Bildbestand gehasht.
Ohne `--inventory` entsteht mit dem vorhandenen `dataset_plan` ein neuer Audit.

### 1. Abschätzung, Cache-Aufbau und vollständiger Checksum (CPU auf A100)

```bash
python scripts/cache_sem_ssl_images.py --inventory "$SEM_SSL_INVENTORY" \
  --cache-dir "$SEM_SSL_CACHE" --estimate-only

python scripts/cache_sem_ssl_images.py --inventory "$SEM_SSL_INVENTORY" \
  --cache-dir "$SEM_SSL_CACHE" --max-cache-gb 24 --workers 8

python scripts/cache_sem_ssl_images.py --inventory "$SEM_SSL_INVENTORY" \
  --cache-dir "$SEM_SSL_CACHE" --verify
```

Wenn tmpfs/RAM nicht genügt, `SEM_SSL_CACHE` vor dem Build auf eine ausreichend
große lokale SSD setzen und die Abschätzung dort wiederholen. Keine feste
Annahme über freie lokale Laufwerke oder Schreibrechte auf A100.

### 2. Langer, uninstrumentierter GPU-Vergleich

Erst den ressourcengeprüften Plan anzeigen:

```bash
python scripts/benchmark_sem_ssl_loader.py --inventory "$SEM_SSL_INVENTORY" \
  --decoded-cache "$SEM_SSL_CACHE" --worker-counts 8 12 16 \
  --batch-sizes 32 64 --warmup-steps 10 --measured-steps 100 \
  --full-ablation --dry-run \
  --output outputs/sem_ssl_benchmarks/loader_compare_v1
```

Anschließend bewusst auf genau einer zugewiesenen A100 starten:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_sem_ssl_loader.py \
  --inventory "$SEM_SSL_INVENTORY" --decoded-cache "$SEM_SSL_CACHE" \
  --worker-counts 8 12 16 --batch-sizes 32 64 \
  --warmup-steps 10 --measured-steps 100 --full-ablation \
  --output outputs/sem_ssl_benchmarks/loader_compare_v1
```

Ohne `--full-ablation` werden nur Worker-Zahlen mit und ohne Cache verglichen.
Mit diesem Flag kommen Persistent-only, Persistent+Prefetch 4, ungepinnter
Speicher, blockierende Transfers und Cache+Prefetch-4 dazu. Alle Fälle laufen
sequenziell in den vorhandenen isolierten GPU-Benchmark-Prozessen; CPU- und
RAM-Limits können Fälle mit protokollierter Begründung überspringen.

`comparison.json`/`comparison.csv` enthalten für B32 und B64 Durchsatz, Änderung
gegenüber **dem neuen langen `raw_w8`-Lauf**, mittlere/p95 Loader-Wartezeit,
p95/Median-Schrittzeit, PSS und zusätzlichen PSS sowie Cache- und GPU-Speicher.
Pro Fall bleiben außerdem die vollständigen `benchmark.json`/CSV/Plots erhalten.
Der historische kurze Lauf mit 20 Messschritten ist Kontext, nicht der Nenner
einer ungeprüften neuen Verbesserungszahl.

### 3. Separater Komponenten-/H2D-Diagnoselauf

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_sem_ssl_a100.py \
  --inventory "$SEM_SSL_INVENTORY" --batch-sizes 32 64 --num-workers 12 \
  --decoded-cache "$SEM_SSL_CACHE" --prefetch-factor 2 --profile-loader \
  --warmup-steps 10 --measured-steps 100 \
  --output outputs/sem_ssl_benchmarks/loader_profile_v1
```

Nur bei mindestens zwölf verfügbaren CPUs verwenden. Die per-Batch-CPU-Timings
liegen in `result.json -> measured.timing.loader_profiles`, Transferwerte in
`h2d_seconds`. Für den ursprünglichen Pfad `--decoded-cache none` verwenden.

### 4. CPU-Diagnose ohne GPU

```bash
python scripts/profile_sem_ssl_data.py --inventory "$SEM_SSL_INVENTORY" \
  --images 128 --worker-counts 8 12 16 --batch-sizes 32 64 \
  --warmup-batches 4 --steps 128 --max-cache-gb 2 \
  --cache-dir /tmp/sem_ssl_cpu_sample_rgb_v1 \
  --output outputs/sem_ssl_checks/loader_cpu_profile_v1
```

Dieser kleine Sample-Cache ist ausschließlich für das Komponentenprofil,
nicht als vollständiger Trainingscache verwendbar. Für wiederholte Profile
mit derselben Auswahl ist `--reuse-cache` möglich. CPU-Benchmarks mit acht oder
mehr Workern benötigen funktionierende Unix-IPC-Sockets für Torch; eine Sandbox
kann sie sperren. Der Profiler erkennt diese Sperre vor dem Start der Worker.

## Gemessener CPU-Vergleich vom 11.11.2026

Workstation: zwölf verfügbare CPUs, 31,0 GiB RAM, ein Torch-/BLAS-Thread pro
Worker. Identische 128 echte Train-Bilder, unveränderte Meta-224/98-Multicrops,
vier Warmup-Batches und **128 gemessene Batches je Fall**. Die Auswahl wird
deterministisch wiederholt; dies ist kein Durchlauf über das gesamte Korpus.
Alle folgenden Zahlen stammen aus dem instrumentierten CPU-Profil, ohne Modell,
CUDA oder gepinnten Speicher. Keine A100-End-to-End-Beschleunigung daraus ableiten.

| Batch | Worker | Direkt, Bilder/s | RGB-Cache, Bilder/s | Änderung | PSS direkt → Cache |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 8 | 143,0 | 225,4 | +57,6 % | 4,41 → 4,56 GiB |
| 64 | 8 | 137,4 | 215,1 | +56,5 % | 4,78 → 5,06 GiB |
| 32 | 12 | 143,3 | 225,8 | +57,5 % | 6,17 → 6,38 GiB |
| 64 | 12 | 161,0 | 230,3 | +43,0 % | 6,84 → 7,18 GiB |

16 Worker wurden wegen des CPU-Limits ausgelassen. Acht Worker sind auf dieser
Workstation ein sinnvoller Ausgangspunkt: zwölf Worker bringen bei gecachtem
B32 praktisch keinen Mehrdurchsatz und bei B64 etwa 7 %, benötigen aber gut
2 GiB mehr Prozessgruppen-PSS. Das ersetzt den A100-Vergleich nicht.

Bei B64/8 Workern entfallen im direkten Pfad pro Bild etwa 0,38 ms auf Lesen,
7,42 ms auf Decode/RGB, 14,84 ms auf die bisherige Metadaten-Erkennung und
32,79 ms auf Meta-Augmentationen; Collate benötigt umgerechnet 0,81 ms/Bild.
Mit Cache entfällt die wiederholte Metadaten-Erkennung vollständig, der Zugriff
braucht etwa 0,87 ms/Bild. Die weiter frisch erzeugten Augmentationen benötigen
35,56 ms/Bild, davon 17,83 ms Geometrie, 5,33 ms Farbe, 9,48 ms Blur/Solarization
und 2,82 ms Normalisierung. Die Unterschiede der Augmentationszeiten zeigen auch
CPU-/Speicher-Contention; sie sind keine geänderte Verteilung.

Der Sample-Cache selbst belegt einmalig **254 MiB** auf tmpfs. Gegenüber dem
direkten Fall steigt der gemessene Prozessgruppen-PSS bei acht Workern um
etwa **148 MiB bei B32** beziehungsweise **285 MiB bei B64**. Das sind
zeitpunktabhängige Snapshots einschließlich mmap-Seiten, keine zusätzliche
254-MiB-Kopie pro Worker und keine vollständige RAM-Spitzenmessung.

Rohdaten und Testbelege liegen unter
lokales Prüfprotokoll.
Kurze Vorversuche mit weniger Mess-Batches werden nicht für diese Tabelle
verwendet. Eine einzelne CPU-Messung mit kleinem, warmem Bildbestand quantifiziert
keine Kaltstart-/Shared-Filesystem-Leistung des vollständigen Datensatzes.

Der vollständige CPU-Testsatz ist mit `sem-gpu` durchgelaufen:
**193 passed, 3 skipped, 3 subtests passed** in 90,84 s. Darin enthalten sind
zwölf neue Tests zu Pixel-/Crop-/Maskenparität, frischen Ansichten, Cache-Identität,
Korruption, Budgets, Spawn-Serialisierung, Fehler-Cleanup, Loader-Optionen und
einem vollständigen CPU-SSL-Update mit dem Transfer-Hook. Bestehende Tests zu
Original-Loss-Parität, EMA, Checkpointing, Export und Resume bestehen weiterhin.
Die CLI-Hilfen und der GPU-Sweep als `--dry-run` sind zusätzlich geprüft.
Metas Checkout ist am dokumentierten SHA unverändert; keine Pakete wurden
installiert oder geändert.

## Gemessener A100-Vergleich vom 11.11.2026

Der Nutzer hat den oben beschriebenen GPU-Sweep auf `gpu-host` gestartet.
Codex hat ausschließlich die gemeinsamen Ergebnisdateien gelesen. Hardware:
eine NVIDIA A100-SXM4-80GB, BF16, native Attention, echte unveränderte NFFA-
Train-Auswahl. Pro isoliertem Batch-Test zehn Warmup- und 100 Mess-Updates mit
Forward, Original-Losses, Backward, AdamW und Teacher-EMA.

Quelle: `outputs/sem_ssl_benchmarks/loader_compare_v1/comparison.json`.
Eine kleine dauerhafte Kopie einschließlich Analyse und Ergebnis-Hashes liegt
unter lokales Prüfprotokoll.

| Einstellung | B32 Bilder/s | B64 Bilder/s | B64 Loader-Warten, Mittel |
| --- | ---: | ---: | ---: |
| 8 Worker, Originalpfad | 146,47 | 154,28 | 213,37 ms |
| 12 Worker, Originalpfad | 249,07 | 227,44 | 78,00 ms |
| 16 Worker, Originalpfad | **259,79** | **316,39** | **0,254 ms** |
| 8 Worker, Persistent | 159,14 | 163,81 | 192,48 ms |
| 8 Worker, Persistent + Prefetch 4 | 159,53 | 160,25 | 198,74 ms |
| 8 Worker, ohne Pinning | 162,98 | 158,13 | 206,29 ms |
| 8 Worker, blockierende Transfers | 162,37 | 159,22 | 202,88 ms |

Die Worker-Erhöhung liefert gegenüber der **neuen langen** Acht-Worker-Baseline
**+77,4 % bei B32** und **+105,1 % bei B64**. Diese Prozentwerte verwenden nicht
die frühere, kürzere 20-Update-Messung. B64 mit 16 Workern verarbeitet außerdem
etwa 21,8 % mehr Originalbilder/s als B32 mit 16 Workern.

Die Empfehlung für den nächsten bewusst gestarteten Piloten ist daher:

```yaml
train:
  batch_size_per_gpu: 64
  num_workers: 16
loader:
  decoded_cache: null
  persistent_workers: false
  prefetch_factor: 2
  pin_memory: true
  non_blocking: true
  profile: false
```

Diese Empfehlung wurde anschließend auf ausdrücklichen Wunsch des Nutzers in
`configs/sem_ssl/nffa_200step.yaml` übernommen. Es wurde kein Training gestartet.
Der Schedule bleibt bei 200 Updates; die bestehende originale LR-Skalierung
ergibt bei Batch 64 eine maximale LR von `5e-5`. Bei diesem Fall
betragen die GPU-Spitzen etwa **17,42 GiB allocated / 30,62 GiB reserved**.
Der Prozessgruppen-PSS beträgt **14,04 GiB**, gegenüber 7,23 GiB bei B64/8 Workern,
also etwa **6,80 GiB zusätzlich**. Es ist kein Bildcache enthalten.
Das p95/Median-Verhältnis der Schrittzeit sinkt von 7,53 auf **1,09**; das
p95 der Loader-Wartezeit beträgt nur **0,461 ms**. Im gemessenen Fenster ist
die Datenbereitstellung damit praktisch kein Engpass mehr.

**Unvollständiger Cache-Vergleich:** 14 Batch-Tests sind erfolgreich, die acht
Cache-Batch-Tests scheitern bereits beim Vorbereiten. Grund laut Fall-Logs:
`/dev/shm/sem_ssl_nffa_rgb_v1/complete.json` fehlt auf A100. Es gibt deshalb
weiterhin keine gemessene GPU-Cache-Beschleunigung und keinen gemessenen
RAM-Bedarf des vollständigen Caches. Ein gesetztes `--decoded-cache` baut den
Cache nicht automatisch. Die fehlgeschlagenen Fälle sind in den Ergebnisdateien
als `error` enthalten und dürfen nicht als durchgeführte Messungen zählen.

Alle 14 erfolgreichen Fälle starten mit demselben Student-/Teacher-Zustands-Hash
und bestehen die eingebauten Finite-Checks. Die BF16-GPU-Losskurven gleicher
Batch-Größe sind über separate Prozesse nicht bitidentisch; die größte beobachtete
absolute Total-Loss-Differenz gegenüber dem Acht-Worker-Fall beträgt 0,00420
bei B32 beziehungsweise 0,00301 bei B64. Die Ursache dieser kleinen Differenzen
wurde im Durchsatztest nicht isoliert. Die Konfiguration nutzt weiterhin
`deterministic: warn`; bitexakte GPU-Reproduzierbarkeit wird nicht behauptet.
Die nachgewiesene bitweise Pixel-/Augmentations-/Maskenparität bezieht sich auf
die separaten CPU-Integrationstests.

## Grenzen und noch offene Messung

Pro Konfiguration liegt eine sequenzielle Messung vor, keine randomisierte
Wiederholungsstudie. Kleine Unterschiede bei Persistent/Pinning/Transfers mit
acht Workern belegen keine allgemeine Überlegenheit der jeweiligen Option;
Warm-Caches und andere Hostlast wurden nicht unabhängig kontrolliert. Die große
Verbesserung durch 16 Worker und die nahezu entfallene Loader-Wartezeit machen
diese Einstellung zum sinnvollen Ausgangspunkt für den begrenzten Piloten.

Der Cache- und der instrumentierte H2D-GPU-Vergleich sind noch offen. Für den
aktuell empfohlenen ungecachten Pfad wird ein zusätzlicher Bildcache zunächst
nicht benötigt. Langzeitstabilität und Repräsentationsqualität sind durch einen
110-Update-Durchsatztest nicht bewiesen. Optimierungen wie feste Ansichten,
andere Crop-Verteilungen, anders normalisierte Bilder oder geänderte Losses
werden nicht eingesetzt. Keine neue Umgebung und kein volles GPU-Training.
