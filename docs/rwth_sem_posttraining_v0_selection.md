# RWTH-Auswahl im SEM-Posttraining-Manifest V0

Am 19.11.2026 wurde der RWTH-Anteil im aktiven Manifest von 191.400 auf
**26.000 Dateien** reduziert. Die vollständige Kandidatentabelle enthält
weiterhin 636.224 Zeilen. Der Trainingsexport umfasst jetzt **179.622 Bilder**.

| Gruppe | Eingeschlossen | Deaktiviert | Auswahl |
| --- | ---: | ---: | --- |
| 16MnCrS5 | 16.000 | 128.000 | 100 Kacheln aus jedem der 160 Aufnahmeblöcke |
| DP/10mu | 10.000 | 15.281 | HR-IDs ohne Zurücklegen aus `range(25281)` mit `random.Random(42).sample` unter Python 3.11 |
| DP/100mu | 0 | 22.119 | Ganze Gruppe für V0 deaktiviert |

## Zuordnung und Auswahl

Für 16MnCrS5 gilt die im Audit belegte numerische Reihenfolge. Bei Tile-ID
`n` ist der Aufnahmeblock `n // 900`. Der Index `n % 900` durchläuft ein
30×30-Raster zeilenweise. Die Auswahl verwendet in beiden Achsen die
nullbasierten Positionen **1, 4, 7, …, 28**. Damit liegt je eine Kachel in der
Mitte jedes 3×3-Bereichs und die Auswahl verteilt sich über das gesamte
veröffentlichte Raster. Der nominale Abstand ausgewählter Kacheln beträgt
288 Quellpixel pro Achse.

Für alle 160 Blöcke wurden fünf über das Raster verteilte Stellen anhand
ihrer rechten und unteren Nachbarn geprüft. Alle 1.600 erwarteten
pixelgenauen Überlappungen waren eindeutig. Zur Berücksichtigung des
Bilddrifts prüft der Vergleich Verschiebungen innerhalb von vier Pixeln
um den nominalen Schritt von 96 Pixeln. Bei drei zusätzlichen Prüfungen
pro Blockgrenze ergaben sich keine passenden Überlappungen zwischen zwei
Blöcken. Diese Stichprobe bestätigt die Blockaufteilung zusammen mit dem
Generator und der vollständigen numerischen ID-Folge. Die Originalnamen
sind nur bei den fünf bereits im Audit zugeordneten Aufnahmen bekannt.
Maßgeblich für die Auswahl sind die geprüften Blocknummern. Bei einem
unklaren Block bricht das Skript vor der Veröffentlichung ab und meldet
die betroffenen Nummern. [RWTH-Audit](rwth_sem_superresolution_audit.md) ·
[Pixelprüfung aller Blöcke](../outputs/sem_corpus/sem_posttraining_v0/rwth_frame_verification.json)

Die deaktivierten Zeilen bleiben mit `included=0` und ihrem
`exclusion_reason` in der Haupttabelle. Der Export enthält nur die
eingeschlossenen HR-Bilder. Das Skript prüft den Export gegen die
erwarteten RWTH-IDs und vergleicht sämtliche übrigen Exportzeilen
eins zu eins mit der vorherigen Version. Dazu gehören die 12.801
NFFA-Bilder aus Split A Train und die 1.199 Aachen-Train-Bilder.
[Auswahlprotokoll](../outputs/sem_corpus/sem_posttraining_v0/curation_rwth_hr26k_v0.json)

Ein zweiter vollständiger Vergleich der 636.224 Hauptzeilen erlaubte nur die
vorgesehenen Änderungen von `included` und `exclusion_reason` bei RWTH.
Der bestehende V0-Loader las den aktiven Export mit 179.622 Einträgen und
erzeugte aus je einem ausgewählten HR-Bild beider aktiven RWTH-Gruppen einen
Batch mit globalen 518er-Crops, lokalen 224er-Crops und Meta-Masken.
Die CPU-Gesamtsuite bestand mit 334 Tests und 3 Subtests. 3 Tests wurden
übersprungen.

## Reproduktion

Der vorherige Manifeststand liegt unter `sem_posttraining_v0_before_rwth_hr26k`.
Ein separater Nachbau prüft die gleichen Pixelstichproben und schreibt ein
neues Verzeichnis neben dem archivierten Bestand.

```bash
cd /path/to/nffa-sem-dinov2
python \
  -m scripts.select_rwth_sem_posttraining_v0 \
  --manifest-dir outputs/sem_corpus/sem_posttraining_v0_before_rwth_hr26k \
  --output outputs/sem_corpus/sem_posttraining_v0_rwth_reproduction
```

Der aktive Export bleibt unter `outputs/sem_corpus/sem_posttraining_v0/training_manifest.csv`
und verwendet das bisherige V0-CSV-Schema. Der allgemeine Manifest-Builder
erstellt das Ausgangsinventar. Die RWTH-Auswahl ist ein eigener
Kurationsschritt. [Skript](../scripts/select_rwth_sem_posttraining_v0.py)
