# CPU-Worker für die Frozen-Evaluation nach 5.000 SSL-Schritten

Die vorhandenen Skripte reichen `--workers` bereits korrekt durch:

| Pfad | Verwendung |
| --- | --- |
| `evaluate_sem_ssl.py --workers N` | Übergibt N für jeden Checkpoint an `evaluate_sem_encoder.py`. |
| NFFA `run_classification` → `extract_features` | DataLoader: Batch 64, **neuer Loader pro 512-Bilder-Chunk**, Pinning und nicht-blockierendes H2D. |
| NFFA `fit_mlp` | Übernimmt denselben Worker-Wert für den bestehenden Feature-Tensor-Loader. LogReg hat keinen Bild-DataLoader. |
| Aachen `run_segmentation` → `run_aachen_ablation.py` | Übergibt N an Extraktion und die bestehenden persistenten Train-/Val-Loader der Heads. |
| Aachen A3 `extract_cache` | Ein fortlaufender Spawn-Loader, ein Original pro Item; bestehende 896×672-Letterbox und Dense-Features. |

Es sind keine Änderungen an diesen Runnern erforderlich. Acht Worker bleiben
der unveränderte Standard. Die 16-Worker-Empfehlung für SSL-Multicrops ist kein
gemessenes Optimum für diese andere Datenpipeline. Insbesondere hat ein voller
NFFA-Chunk nur acht Batches; zusätzliche Worker müssen sich dort trotz ihrer
Startkosten erst lohnen.

## Ein kurzer Benchmark-Aufruf auf A100

Im Projektverzeichnis, auf einer zugewiesenen A100 (hier physische GPU 0):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_sem_eval_workers.py --output outputs/sem_eval_worker_benchmark/v1
```

Der Aufruf startet **keine Evaluation, kein Head-Training und kein SSL**. Er:

* protokolliert auf dem tatsächlich ausführenden Host physische/logische CPUs,
  die Affinity, verfügbare CPUs inklusive zugänglicher cgroup-v2-Quotas und RAM;
* testet 8/16/24/32 Worker sequenziell auf genau einer sichtbaren A100;
* überspringt Worker-Zahlen, die CPU-/RAM-Grenzen überschreiten;
* nutzt den bereits verifizierten Schritt-5000-Backbone strikt und offline;
* verwendet dieselben 1.024 NFFA-Train/Val-Bilder und 128 Aachen-Train/Val-Originale
  pro Einstellung, deterministisch mit Seed 42 ausgewählt;
* verwendet die bestehenden Dataset-Klassen, Transforms und globalen/dichten
  Forward-Methoden in eval/inference mode und **FP32 wie die echte Extraktion**;
* misst Lesen/Decodieren/Vorverarbeitung, Loader-Start/-Stop, H2D, Backbone und D2H.
  NFFA wird tatsächlich in 512-Bilder-Chunks mit Batch 64 verarbeitet, Aachen
  wie bisher mit einem Original pro Loader-Item und A3-Auflösung;
* berechnet **keine Dataset-SHA256** und schreibt keine Feature-Caches. Die
  unveränderte Daten-/Checkpoint-Prüfung des echten Evaluationsrunners bleibt aktiv.

Die Messung enthält keine Feature-Datei-Serialisierung und kein Hashing. Sie
bestimmt das passende Worker-Budget für die Bild-/Feature-Extraktion, nicht die
Laufzeit aller Head-Trainings. Modellinitialisierung und ein reales Warmup-Bild
je Task sind außerhalb der Messung. Loader-Startkosten sind enthalten und
zusätzlich als `first_batch_seconds` sichtbar.

Ergebnisse im neuen Ausgabeordner:

* `resources.json`: reale A100-Ressourcen und ausgewählte GPU.
* `benchmark.csv` / `benchmark.json`: Bilder/s, gesamte gemessene Wandzeit,
  CPU-Zeit, CPU-Auslastung und mittlere GPU-Auslastung je Task/Worker-Zahl.
* `*_gpu.csv`: nvidia-smi-Samples im Abstand von 500 ms, nur für die ausgewählte
  physische GPU. Die Auslastung ist geräteweit und kann andere Prozesse enthalten.
* `evaluate_all.sh`: **nur erzeugter, nicht ausgeführter** vollständiger Aufrufplan.

`cpu_percent` ist Prozessgruppen-CPU-Zeit geteilt durch Wandzeit: 100 % bedeutet
einen belegten logischen Kern. `cpu_percent_of_affinity` normiert zusätzlich
auf das verfügbare CPU-Budget. CPU-Zeit umfasst Elternprozess und beendete
Loader-Worker, keine anderen Jobs auf dem Server. GPU-Mittelwerte enthalten
auch Loader-Anlauf und Leerzeiten; niedrige Werte sind hier informativ.

Empfehlung je Task: kleinste Worker-Zahl innerhalb von 3 % des höchsten
gemessenen Durchsatzes. Bei ähnlichen Ergebnissen kann mit `--repeats 2` in
einem **neuen** Ausgabeordner wiederholt werden; der zweite Durchlauf kehrt die
Worker-Reihenfolge um und die Empfehlung verwendet die Median-Durchsätze.
Bestehende Ausgaben werden nicht überschrieben. Dies ist ein kurzer sequenzieller
Vergleich, keine garantierte Langzeit- oder Head-Training-Optimierung.

## Ein Aufruf für alle sechs Encoder und den Abschlussbericht

Nach erfolgreichem Benchmark die gewählten Worker in `benchmark.json` prüfen.
Dann den erzeugten Plan ausdrücklich starten:

```bash
bash /path/to/nffa-sem-dinov2/outputs/sem_eval_worker_benchmark/v1/evaluate_all.sh
```

Der Plan nutzt denselben bestehenden `sem-gpu`-Interpreter, eine A100,
OMP/MKL/OpenBLAS jeweils 1 und die gemessene Worker-Zahl **pro Task identisch für
alle Encoder**. Er führt nacheinander aus:

1. Checkpoints **0/500/1000/2000/3000/5000** aus
   `outputs/sem_ssl/nffa_5000step_seed42`: NFFA Split A/B, LogReg + Original-MLP.
2. Dieselben Checkpoints: Aachen A3, Linear + Original-CNN8.
3. `report_sem_ssl.py` für den gemeinsamen Vergleich.

Es werden ausschließlich die bestehenden Train-/Validierungsprotokolle genutzt;
Testdaten bleiben unberührt. Die Encoder werden nicht nachtrainiert.
Features und getrennte Head-Ergebnisse liegen unter
`outputs/encoder_evaluation/nffa_5000step_workers_v1`, der Abschlussbericht unter
`outputs/sem_ssl_reports/nffa_5000step_workers_v1`. Der bestehende Runner erhält
`--resume` und führt seine normalen Prüfungen aus. Diese Namen vor dem Start
ändern, falls dort bereits ein inkompatibler Versuch liegt.

## Stand der Messung

Codex hat diesen neuen GPU-Worker-Benchmark nicht gestartet. Direkter SSH-/GPU-
Der A100-Zugriff muss auf dem GPU-Rechner geprüft werden. Die gemeinsame Dateifreigabe allein
liefert keine A100-CPU-/GPU-Ressourcen. Bis der obige Aufruf auf A100 abgeschlossen
ist, gibt es keine neuen gemessenen A100-Ressourcen oder Durchsätze und keine
belegte optimale Worker-Zahl. Die Empfehlung wird aus den echten Ergebnissen
erzeugt; fehlende/fehlgeschlagene Task-Messungen erzeugen keinen Startplan.
