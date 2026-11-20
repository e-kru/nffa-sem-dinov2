# MatSciFig SEM — Sichtprüfung und CLIP-Pilot

Der Pilot bezieht sich auf die 60.814 MatSciFig-Dateien im aktiven V0-Trainingsexport.
Eine Zufallsstichprobe mit Seed 42 wählt 400 verschiedene DOI aus 7.753
Publikationen und pro DOI ein zufälliges Panel. So können Publikationen mit
vielen Panels die Prüfung kaum dominieren. Jede der 400 Stichprobendateien
war dekodierbar. Die Stichprobe ist nach Publikationen verteilt und erlaubt
keine direkte Hochrechnung auf den Anteil der Dateitypen in allen 60.814 Bildern.

## Ausschlussvorschlag

Der Pilot verwendet OpenAI CLIP ViT-B/32 auf der CPU. Die Gewichte sind über
den offiziellen SHA-256 fixiert. Der [gepinnte OpenAI-Quellcode](https://github.com/openai/CLIP/tree/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1)
und die zusätzlichen Tokenizer-Pakete liegen separat unter
`outputs/dataset_audits/matscifig_clip_assets/`.

CLIP vergleicht zwei Ansichten desselben Bildes. Eine zeigt das gesamte Panel
maßstabsgetreu mit weißem Rand. Die zweite ist der native CLIP-Mittelausschnitt.
Englische Textprompts decken echte SEM-Aufnahmen, SEM mit Beschriftungen,
Pfeilen, Skalenbalken, farbigen Overlays und gemischte Panels ab. Ausschlussklassen
sind reine Diagramme, Schemata und reine Phasenkarten oder Masken ohne sichtbare
mikroskopische Textur. Ein Bild kommt nur auf die Prüfliste, wenn beide Ansichten
dieselbe Ausschlussklasse bevorzugen, deren Kosinus-Ähnlichkeit jeweils
mindestens 0,25 beträgt und der Abstand zum stärksten SEM-Prompt jeweils
mindestens 0,05 ist. Alle übrigen Bilder erhalten vorläufig `keep`.
Diese Schwellen sind konservative Startwerte, die Kosinuswerte keine
kalibrierten Wahrscheinlichkeiten. Reale Phasenkarten und künstliche Karten
lassen sich anhand des Aussehens allein nicht immer unterscheiden.

[`model_scores.csv`](../outputs/dataset_audits/matscifig_sem_clip_v0/model_scores.csv)
enthält die beiden Scoresätze aller 400 Bilder.
[`proposed_exclusions.csv`](../outputs/dataset_audits/matscifig_sem_clip_v0/proposed_exclusions.csv)
enthält ausschließlich die Modellvorschläge für die Sichtprüfung. Die
übrigen Dateien außerhalb der Stichprobe wurden nicht bewertet.

Im Pilot wurden 400 Bilder bewertet. **Kein Bild** erfüllte beide
konservativen Ausschlussschwellen. Damit bleiben alle 60.814 Dateien im
Bestand. Bei **19 Bildern** bevorzugten beide Ansichten dieselbe
Ausschlussklasse, mit zu wenig Abstand zur SEM-Alternative. Diese 14
Phasenkarten-, 3 Diagramm- und 2 Schema-Hinweise stehen als reine
Sichtprüfungspriorität in
[`priority_review.csv`](../outputs/dataset_audits/matscifig_sem_clip_v0/priority_review.csv).
Erkennungsraten bleiben bis zur manuellen Prüfung offen.

Der [Kontaktbogen der 19 priorisierten Fälle](../outputs/dataset_audits/matscifig_sem_clip_v0/priority_contact_sheet/sheet_01.jpg)
zeigt, warum Modellübereinstimmung allein zu wenig ist. Darauf stehen
zum Beispiel das Balkendiagramm MSF-059 und die beschriftete BSE-Aufnahme
MSF-147. Diese Sichtkontrolle ist keine systematische Label-Auswertung.

## Menschliche Prüfung

Die [16 Kontaktbögen als PDF](../outputs/dataset_audits/matscifig_sem_clip_v0/contact_sheets/contact_sheets.pdf)
und die einzelnen JPG-Seiten im selben Ordner zeigen keine Modellvorschläge.
Bewerte die Bilder anhand ihrer IDs in [manual_review.csv](../outputs/dataset_audits/matscifig_sem_clip_v0/manual_review.csv).
Nutze `sem` für Bilder mit echter SEM-Textur, auch mit Beschriftungen, Skalenbalken
oder Diagrammen neben einem echten SEM-Panel. Nutze `non_sem` für klare
Ausschlüsse und `uncertain` für unklare Fälle. Unsichere Fälle bleiben für die
Kuration erhalten. Für eine SEM-Recall-Schätzung muss die Sichtprüfung sowohl
vorgeschlagene Ausschlüsse als auch vom Modell behaltene Bilder umfassen.

```bash
cd /path/to/nffa-sem-dinov2
python \
  -m scripts.audit_matscifig_sem_clip evaluate \
  --output outputs/dataset_audits/matscifig_sem_clip_v0 \
  --labels outputs/dataset_audits/matscifig_sem_clip_v0/manual_review.csv
```

Die Auswertung meldet SEM-Recall und Präzision der Ausschlussvorschläge unter
den sicher gelabelten Beispielen. Bei leeren Labels gibt es keine Metriken.
Der Auditpfad ist separat vom Rohdaten- und Trainingsmanifestpfad.

## Reproduktion

Mit dem vorhandenen `sem-gpu`-Python und den separaten CLIP-Assets
`python -m scripts.audit_matscifig_sem_clip sample --output <neuer-auditordner>`
und danach
`python -m scripts.audit_matscifig_sem_clip score --output <neuer-auditordner>`
ausführen. Die SHA-256-Prüfung bindet den Stichprobenexport und den
Encoder-Checkpoint. Modell, Prompts, Schwellen und Ergebnisse stehen in
[`model_summary.json`](../outputs/dataset_audits/matscifig_sem_clip_v0/model_summary.json).
Die CPU-Gesamtsuite bestand mit 337 Tests und 3 Subtests. 3 Tests wurden übersprungen.
