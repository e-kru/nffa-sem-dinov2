# Kurze A100-Diagnose nach erneutem Resume-Fehler

Der Lauf `nffa_split_a_vitl14_resumecheck_v2_seed42` scheiterte bei B16 und B32
erneut am exakten Update-Vergleich. Der zuvor auf CPU nachgewiesene AdamW-Payload-
Alias wurde behoben, erklärt aber nicht allein diesen zweiten GPU-Befund.
Die gespeicherten alten Fehlermeldungen enthalten keine Differenzgrößen oder
genauen Zustandsschlüssel. Deshalb lässt sich daraus keine weitere Ursache
behaupten. `tifffile` ist nicht der Auslöser dieser ValueError-Meldung.

**Ergebnis v3, gelesen am19.11.2026:** Der Nutzerlauf auf GPU2 unter
`outputs/sem_ssl_diagnostics/resume_v3_b16_gpu2/` ist ebenfalls `resume_error`.
Der registrierte Checkpoint-Zustand wird exakt rekonstruiert. Eingaben, RNG,
Modulmodi und Gesamt-Loss sind identisch; die Backbone-Gradientennorm beträgt
jedoch8.578800201416016 gegenüber8.584057807922363. Maximale Gewichtsabweichung
2.9802322387695312e-08, maximale Optimizer-Abweichung0.00011748215183615685.
Das stützt die Hypothese nichtdeterministischer CUDA-Backwards, identifiziert
aber noch keinen Kernel. Die geringe LR des dritten Updates dämpft die sichtbare
Gewichtsabweichung; deren geringe Größe allein gibt keine Trainingsfreigabe.
Der vollständige Befund und die CPU-Prüfung
trennen gemessene Ergebnisse von noch offenen Ursachen.

**Ergebnis v4 vom 19.11.2026.** Der strikte Gegencheck auf GPU 2 brach im
ersten Warmup-Backward bei `upsample_bicubic2d_backward_out_cuda` ab.
Die originale DINOv2-Positionsinterpolation nutzt bikubisches Resampling für
das 224er-Patchraster. Die installierte PyTorch-Version hat für diesen
CUDA-Backward keine deterministische Implementierung. Der Abbruch erfolgte
vor dem ersten Optimizer-Update und vor dem Resume- und Exportvergleich.
Wie stark dieser Operator zu den in v3 gemessenen Abweichungen beiträgt,
ist offen. [GPU-Fehler und Traceback](../outputs/sem_ssl_diagnostics/resume_v4_b16_gpu2_deterministic/runtime_error.json)

**Freigabeentscheidung für den NFFA-Pilot.** Nach Nutzerfreigabe ist
`runtime.allow_cuda_resume_update_drift: true` im NFFA-Pilotprofil gesetzt.
Das erlaubt im üblichen CUDA-Modus endliche Abweichungen nach dem
Vergleichsupdate in Student-/Teacher-Gleitkommatensoren und AdamW-Momenten.
Es wurde kein Grenzwert aus Messwiederholungen festgelegt. Form, Datentyp
und alle übrigen Werte bleiben exakt geprüft, darunter die Center,
AdamW-Schrittzähler, der Gesamt-Loss, der geladene Zustand, die Eingaben und
die Zufallszustände. Der Benchmark protokolliert den exakten Vergleich und
die Ausnahme getrennt. Speicher-, Export- und Kalibrierungschecks bleiben
Teil des Vorlaufs. Der separate Diagnosebefehl unten bleibt bitgenau.
[Startbefehl für den freigegebenen Pilot](sem_cellstyle_nffa_pilot.md#start-auf-a100)

## Was jetzt geprüft und gespeichert wird

Der unverändert strenge Vergleich trennt jetzt:

- exakte Parameter-/Center-/Optimizer-Gleichheit **vor** dem Update;
- Python-/NumPy-/CPU-/CUDA-RNG und Module-Train/Eval-Modi;
- tatsächlich identische augmentierte Batch-Tensoren, Masken, Indizes und Draws;
- Loss, Gewichte, Teacher, Centers, Optimizer-Momente und Schrittzähler **danach**;
- Anzahl betroffener Tensoren/Elemente, maximale absolute Differenz und erste
  betroffene Zustandsschlüssel.

Es wird weiterhin mit **Nulltoleranz** verglichen. Der Standardtest verändert
keine CUDA-Determinismuseinstellungen, Attention-Backends, Losses oder Updates.
Nur die ausdrückliche neue CLI-Option `--deterministic` aktiviert den unten
beschriebenen Gegencheck; die wissenschaftliche Trainingsrezeptur bleibt erhalten.
Bei identischen Eingaben, Pre-State und RNG können die Differenzen Hinweise auf
CUDA-Kernel-Reproduzierbarkeit liefern; diese Hypothese muss erst durch den
Diagnosebericht geprüft werden. Die bisherige Rezeptur erlaubt nichtdeterministische
CUDA-Algorithmen, während der bisherige Resume-Test exakte Gleichheit verlangt.
Das ist ein zu prüfender Protokollkonflikt, noch kein nachgewiesener GPU-Bug.

Der reguläre Benchmark speichert bei einem Resume-Fehler künftig trotzdem die
tatsächlich gemessenen endlichen Loss-/Throughput-/VRAM-Werte unter
`candidate_B_training_metrics.json`, dazu `candidate_B_resume_diagnostics.json`.
Der Kandidat bekommt `status=resume_error`, niemals eine bestandene Freigabe.
Die Empfehlung akzeptiert weiterhin ausschließlich vollständig bestandene
Kandidaten. Stufenmeldungen zeigen Warmup, Messende, Checkpoint und Resume.

## Reproduktion des v4-Gegenchecks auf GPU 2

```bash
cd /path/to/nffa-sem-dinov2
conda activate sem-gpu
CUDA_VISIBLE_DEVICES=2 python scripts/diagnose_sem_cellstyle_resume.py \
  --config outputs/sem_ssl_regularization/nffa_split_a_vitl14_resumecheck_v2_seed42/configs/D_memory_probe.yaml \
  --physical-batch 16 \
  --deterministic \
  --output outputs/sem_ssl_diagnostics/resume_v4_b16_gpu2_repeat
```

Das sind **zwei Warmup-Updates und zwei Vergleichsupdates**, nicht erneut der
gesamte Durchsatzbenchmark oder die vierarmige Studie. Dieselben echten
NFFA-Trainingsbilder, Large-Gewichte,518er-Crops, BF16, native SDPA, D-Referenz,
Accumulation, originale Losses, AdamW, Centering und EMA werden verwendet.
Die geringere Zahl von Warmup-Updates ist eine Diagnostik und kein Ersatz
für den später erforderlichen vollständigen Benchmark oder eine Reproduktion
des exakten41.Schritts des alten Laufs.

`--deterministic` setzt vor dem Torch-Import bei fehlender Vorgabe
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Nach der normalen Initialisierung werden
strikt deterministische Algorithmen und `cudnn.deterministic=True` für Warmup,
Checkpoint-/Exportprüfung und beide Vergleichspfade erzwungen. Die vorherigen
Prozessflags werden anschließend auch bei Fehlern wiederhergestellt. Kein
manueller Wechsel des SDPA-Backends und keine Toleranzlockerung.

Der Gegencheck darf früh mit einem benannten Operatorfehler abbrechen, falls
ein verwendeter CUDA-Backward keine deterministische Implementierung bietet.
Die installierte Torch-Dokumentation nennt insbesondere Interpolations-
Backwards einschließlich bicubic; die originale Meta-Positionsinterpolation
wird weiterhin verwendet. Im v4-Lauf trat genau dieser Operatorfehler ein.

Ergebnisse: immer `result.json`; nach dem Vergleich `resume_diagnostics.json`,
bei einem RuntimeError stattdessen `runtime_error.json` mit Phase und Traceback.
Der verworfene vollständige Diagnosecheckpoint bleibt unter `recovery/`
erhalten (mehrere GB), damit Fehler nicht sofort ihre Ausgangsevidenz verlieren.
Er ist **nur Diagnostik**, kein wissenschaftlicher Startzustand. Vorhandene
Ausgaben werden nicht überschrieben. Es wird kein vollständiges Training
gestartet und Codex verbindet sich nicht mit A100.

Neue Dateien: `src/sem_cellstyle/resume_check.py`,
`scripts/diagnose_sem_cellstyle_resume.py` und
`tests/test_sem_cellstyle_resume_diagnostics.py`.
Der strikte Gegencheck v4 bleibt fehlgeschlagen. Für den freigegebenen
NFFA-Pilot steht die A100-Bestätigung der neuen Prüfregel noch aus.

Aktuell16 gezielte CPU-Tests bestanden; Gesamtsuite331 bestanden,3 übersprungen,
3 erfolgreiche Subtests. Vollständiges Testprotokoll und genaue
Abgrenzung zum älteren Teststand:
CPU-Prüfnachweis.
