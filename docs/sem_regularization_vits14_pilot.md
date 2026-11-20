# ViT-S/14-Pilot für L2-SP und Feature-Distillation

Auf dem A100 aus dem Repository starten.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sem_regularization_vits14.sh
```

Die Geräte-ID kann auf eine freie A100 80 GB gesetzt werden. Der Einstieg verwendet die vorhandene Umgebung `sem-gpu` und prüft, dass genau eine BF16-fähige A100 sichtbar ist. Ein erneuter Aufruf mit derselben Konfiguration setzt einen unterbrochenen Arm ab dem letzten gültigen Checkpoint fort. `--plan` prüft auf der CPU nur die Gewichte, das Inventar und das gemeinsame Rezept.

## Versuchsaufbau

Alle vier Arme starten mit dem gepinnten Meta-Checkpoint und Seed 42. A läuft ohne Regularisierung, B mit L2-SP, C mit Feature-Distillation und D mit beiden. Der Datensatz ist das bereits geprüfte Inventar mit 12.801 NFFA-Bildern aus Split A Train. Die Reihenfolge wird über die Draw-IDs erzeugt und nach dem Training zwischen den Armen verglichen.

Jeder Arm läuft 500 AdamW-Updates mit 64 Bildpräsentationen pro Update (32 physisch, zwei Microbatches). Der Pilot verwendet die SEM-Augmentierungen der bestehenden Regularisierungsstudie mit zwei globalen 224er- und acht lokalen 98er-Crops. KoLeo wirkt pro physischem Batch. Backbone- und Head-Spitzen-LR betragen 1e-5 und 1e-4. LR-Warmup, Freeze der letzten Head-Schicht und Teacher-Temperatur-Warmup dauern 25, 5 und 100 Updates. Teacher-EMA und Zentren werden einmal pro Optimizer-Update aktualisiert. Die Originalverluste, Regularisierungsformeln und Checkpointfunktionen kommen aus den vorhandenen Modulen.

## Gradientenkalibrierung

An den Originalgewichten ist L2-SP exakt null. Identische Features sind auch ein stationärer Punkt des Cosinus-Losses. Im CPU-Kurztest mit echten ViT-S-Gewichten, Original-SSL-Head und vier Bildpräsentationen pro Update war der Feature-Gradient am Start etwa 5,3e-7. Nach einem diagnostischen SSL-Update betrugen die rohen Normen etwa 1,3e-9 für L2-SP und 5,0 für Feature-Distillation. Diese Kurztestwerte sind keine Kalibrierungskoeffizienten für die A100.

Beim Start auf A100 erzeugt die vorhandene Kalibrierung zunächst 20 Wegwerf-Updates am Ende des LR-Warmups, optimiert dabei nur SSL und misst danach fünf Updates. Die Feature-Gradienten müssen zehnmal über dem Rauschen am Start liegen. Jeder Koeffizient begrenzt das gemessene Verhältnis seiner Gradientennorm zur SSL-Gradientennorm auf maximal 0,10. Ein zweiter Diagnoseverlauf prüft mit den fixierten Koeffizienten das kombinierte Gradientenverhältnis. Die Kalibrierung nutzt ausschließlich SSL-Trainingsbilder. Bei erfolgreicher Prüfung starten alle vier Arme unabhängig mit Originalgewichten, neuen Optimizer-Zuständen und Schedule-Index 0.

## Auswertung

Die Teacher-Backbones an Schritt 0 und 500 werden geprüft. Die gemeinsame Referenz ist der originale ViT-S/14. An Schritt 500 werden alle vier Arme auf NFFA Split A mit LogReg und Residual-MLP sowie auf Aachen A3 mit Linear und CNN8 validiert. NFFA nutzt 8 Worker, Aachen 16. Die Pipeline verwendet bestehende geprüfte Dateninventare. Feature-Caches werden bei passender Identität wiederverwendet. Die geprüften historischen Referenzcaches haben andere Quellcode-Fingerprints und werden daher im Pilot neu extrahiert.

`report/comparison.csv` enthält die Validierungsmetriken und absoluten Differenzen gegenüber Original und Arm A. Aachen-Schwellenwerte bei 0,5 und nach Validierungsauswahl stehen getrennt. `report/comparison.json` enthält zusätzlich die CLS-/Patch-Cosinusähnlichkeit der exportierten Teacher auf denselben Trainings-Crops und die relative L2-Parameterdrift pro Block für Student und Teacher. Die Ausgaben liegen standardmäßig unter `outputs/sem_ssl_regularization/nffa_split_a_vits14_500step_v1_seed42/`.
