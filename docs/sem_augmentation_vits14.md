# SEM-Augmentierungsablation auf ViT-S/14

Der Pilot verwendet das geprüfte NFFA-Inventar mit 12.801 Bildern aus Split A Train und das Rezept des bisherigen unregularisierten ViT-S-Piloten. Seed 42, 500 Updates, Batch 32 × 2, DINO/iBOT/KoLeo, AdamW und Teacher-Schedules kommen aus dessen Konfiguration. Alle neu trainierten Arme starten vom originalen Meta-Checkpoint. Aug-A kann aus der früheren Regularisierungsstudie übernommen werden, wenn Trainingsvertrag, Bildreihenfolge und Checkpoint-Prüfsummen übereinstimmen.

**Aug-A ist der Standard für den 500-Schritt-Einstieg.** Der Aufruf ohne `--stage` verwendet ausschließlich Aug-A. Die vollständige Ablation beginnt mit `--stage run`. Der Standard schreibt seinen Bericht nach `outputs/sem_ssl_augmentation/nffa_split_a_vits14_500step_aug_a_v1_seed42/`.

| Arm | Zusätzliches Rauschen | Drehung |
|---|---|---|
| Aug-A | aus | aus |
| Aug-B | an | aus |
| Aug-C | aus | an |
| Aug-D | an | an |

Die Zusätze verwenden pro Bilddarstellung und Crop eigene Zufallsströme. Crop-Koordinaten, Helligkeits- und Kontrastfaktoren, Bildreihenfolge und iBOT-Masken sind gepaart. Rauschen wird auf der einzelnen Graustufenebene in `[0,1]` vor Kanalreplikation und Normalisierung erzeugt. Gaussian und Poisson werden unabhängig mit je 20 % Wahrscheinlichkeit angewandt. Gaussian hat Mittelwert 0 und Standardabweichung 0,005–0,02. Die Poisson-Skala 200–1000 steuert die Rauschstärke als Heuristik. Beide Operationen können sich überlagern. C und D ziehen für jeden Crop gleichverteilt 0°, 90°, 180° oder 270°. Die Drehung vertauscht Pixel auf dem quadratischen Raster ohne Interpolation. Spiegelungen sind deaktiviert.

## Geometrie-Policy

Das Datenarchiv und das Datenpapier beschreiben Morphologieklassen. Die veröffentlichten JPEGs enthalten keine wissenschaftlichen Aufnahme-Metadaten. Die SEM-Klassifikationsarbeit nennt beispielsweise gerichtetes Nanowire-Wachstum. Aus der Ordnerklasse allein lässt sich auch für Pulver und poröse Strukturen keine Freigabe für Spiegelung und Rotation ableiten. [NFFA-Datenarchiv](https://b2share.eudat.eu/records/zja8y-53j14) · [Datenpapier](https://doi.org/10.1038/sdata.2018.172) · [SEM-Klassifikationsarbeit](https://doi.org/10.1038/s41598-017-13565-z)

Die Ablationsarme C und D verwenden standardmäßig den Rotationstest auf dem ganzen geprüften Split A Train. Er misst, ob sich die Downstream-Validierung durch diese Drehungen verbessert. Die physikalische Relevanz der Orientierung bleibt Teil der Interpretation. Für einen engeren, anhand der Provenienz begründeten Drehungstest kann `--geometry-policy PFAD` eine Allowlist angeben. Ein Eintrag braucht `source_id`, `evidence`, `rationale` und exakte `image_ids` aus Split A Train. Der Runner prüft IDs, Überschneidungen und Abdeckung.

## Start und Auswertung

```bash
# Standard (Aug-A)
CUDA_VISIBLE_DEVICES=2 bash scripts/run_sem_augmentation_vits14.sh

# CPU-Vorabprüfung und Kontaktbögen
bash scripts/run_sem_augmentation_vits14.sh --stage prepare

# Kurzer Smoke-Test auf genau einer A100
CUDA_VISIBLE_DEVICES=2 bash scripts/run_sem_augmentation_vits14.sh --stage smoke

# Ausführbarer Noise-Vergleich mit gesondertem Ausgabeordner
CUDA_VISIBLE_DEVICES=2 bash scripts/run_sem_augmentation_vits14.sh --stage noise-only

# Vollständige 2×2-Studie mit Drehungen
CUDA_VISIBLE_DEVICES=2 bash scripts/run_sem_augmentation_vits14.sh --stage run
```

Die Kontaktbögen zeigen dieselben Crop-Ausschnitte mit A/B/C/D. Ausgewählt sind Beispiele mit angewandtem Noise und sichtbaren Drehungen. Die auf 98 Pixel berechneten Crops werden für die Anzeige mit Nearest Neighbor vergrößert.

Die Auswertung nutzt die bestehenden Validierungsprotokolle für NFFA Split A (LogReg und Residual-MLP) und Aachen A3 (Linear und CNN8). Die Split-B-Validierung überschneidet sich mit 1.919 Trainingsbildern. Unter `report/` stehen Metriken mit Differenzen zum Original und zu Aug-A, beide Aachen-Schwellenprotokolle, SSL-Losses, Durchsatz, Peak-VRAM und Vergleichsgrafiken. Die vollständige Studie ermittelt zusätzlich die beiden Haupteffekte und ihre Interaktion.
