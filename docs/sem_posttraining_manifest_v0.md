# Posttraining-Manifest und auflösungsabhängige V0-Datenpipeline

Stand 19.11.2026. Beide Audits und der automatische Manifest-Auftrag sind
abgeschlossen. Nach der Fields- und RWTH-Auswahl enthält das Gesamtinventar
636.224 Zeilen, davon **179.622 eingeschlossen** und 456.602 ausgeschlossen.
Das Receipt führt den Bestand weiter als Kandidatenmanifest mit
`training_ready: false`.

## Festgelegte Auswahl

- Mindestauflösung: **beide Originalseiten mindestens128 Pixel**.
- **LITHOS vollständig ausgeschlossen**, unabhängig von Größe und included-Edits.
- **Fields Metal Cement: ausschließlich Dateinamen mit `Rot0_`**. Das sind15.360
  aktive Dateien. Die46.080 Dateien mit anderen Rotationen bleiben mit
  `included=0`, Grund `excluded_fields_non_rot0`, im Master erhalten.
  Diese Auswahl gilt auch für neue Aufbauten mit dem Manifest-Builder.
- **RWTH Steel SEM: 26.000 HR-Dateien.** 16MnCrS5 liefert je 100 Kacheln aus
  160 Aufnahmeblöcken. DP/10mu liefert 10.000 per Seed 42 ausgewählte HR-Bilder.
  DP/100mu ist vollständig deaktiviert.
  [Auswahl und Reproduktion](rwth_sem_posttraining_v0_selection.md)
- NFFA: bestehendes bereinigtes V3-Manifest, ausschließlich **Split A Train**.
  Die12.801 bisherigen Trainingskandidaten werden vor dem Größenfilter über die
  bestehende Inventar-/Split-Provenienz geprüft. Keine Schnittmenge mit B;
  Split-B-Holdouts sind damit nicht geschützt.
- Aachen: ausschließlich die1.199 Bilder des bestehenden gruppierten
  Trainingssplits; keine Masken oder Val-/Testbilder werden geöffnet.
- Weitere Quellen: lesbare Mikroskopiebilder und erkannte Bildvarianten aus
  beiden Audits. Masken, Annotationen, unbekannte Rollen und zu kleine Bilder
  bleiben als ausgeschlossene Zeilen nachvollziehbar.
- Farbige Bilder werden zunächst **nicht** herausgefiltert. Farb-/Qualitätsprüfung,
  Varianten-/Gruppen-Sampling und datensatzübergreifende Leakage-Prüfung folgen
  beim späteren Cleaning. Externe Split-Hinweise bleiben in `original_split`.

Die Größenentscheidung basiert auf Originalheadern vor der bestehenden
Metadatenstreifen-Entfernung. `width` und `height` sind diese unveränderten
Originalabmessungen. Es gibt keine neue SHA-256-Prüfung von Bilddateien.

## Automatischer Auftrag und Dateien

Der abgekoppelte CPU-Auftrag wartet auf vollständige Discovery und abgeschlossene
Headerprüfung. Ein fertiger Audit mit einzelnen unlesbaren Headern kann verwendet
werden; diese Zeilen werden ausgeschlossen. Discovery-Lücken führen zu einem
sichtbaren Fehler statt einem stillschweigend unvollständigen Trainingssatz.
Es werden nur Status-JSON und fertige CSVs gelesen, keine Live-SQLite-Abfragen.

Auftrag: `outputs/sem_corpus/sem_posttraining_v0_job/`.
`launch.json` dokumentiert PID, eingefrorenen Builder, Plan und Quellhash.
`status.json` enthält `waiting_for_audits`, `building_manifest`, `complete` oder
`failed`; `pipeline.log` protokolliert den Ablauf. Der abgeschlossene Erstaufbau
benutzte `builder_snapshot.py` und `plan.json`. Spätere Auswahländerungen haben
eigene Protokolle; die Start-Snapshots des Erstaufbaus bleiben unverändert.

Ziel: `outputs/sem_corpus/sem_posttraining_v0/`:

| Datei | Zweck |
| --- | --- |
| `manifest.csv` | Gesamtinventar mit editierbarem `included=0/1` und `exclusion_reason` |
| `training_manifest.csv` | Initialer Export ausschließlich eingeschlossener Train-Zeilen |
| `manifest_summary.json` | Quellen-/Größenstatistiken, Ausschlussgründe und Provenienz |
| `receipt.draft.json` | Ausdrücklich vorläufiger Datennachweis; keine behauptete Cleaning-Freigabe |
| `README.md` | Bedeutung der Dateien und noch offene Prüfungen |

Zusätzliche Größenflags: `min_side_ge_128`, `min_side_ge_224`,
`min_side_ge_256`, `min_side_ge_448`, `min_side_ge_512`, `min_side_ge_518`.
Jedes Flag prüft **beide** Seiten;518 ist kein Ausschlusskriterium.
`mode`/`channels` sind Headerinformationen und keine sichere Farbinhaltsprüfung.
Die Zahl eingeschlossener Dateien ist keine bestätigte Zahl unabhängiger
Originalaufnahmen. Unbekannte Original-/Probenidentitäten bleiben `unknown`.

Der Builder schreibt zunächst in einen separaten temporären Ordner und
veröffentlicht das fertige Ergebnis atomar. Vorhandene Versionen werden nicht
überschrieben. Nach manuellen included-Edits muss der spätere Cleaning-Schritt
einen neuen Trainingsexport erzeugen; die initiale Exportdatei aktualisiert sich
nicht automatisch bei Tabellenänderungen.

Die ausdrücklich beauftragte Fields-Auswahl aktualisiert den bisherigen V0-Pfad
einschließlich Trainingsexport, Summary, Receipt und Status. Der vorherige Stand
mit391.102 eingeschlossenen Dateien liegt unverändert unter
`outputs/sem_corpus/sem_posttraining_v0_before_fields_rot0_2026-11-19/`.
Das Änderungsprotokoll mit alten Metadatenhashes und vollständiger Exportprüfung
steht unter `outputs/sem_corpus/fields_rot0_only_2026-11-19/` sowie im neuen
Manifestordner in `curation_fields_rot0_2026-11-19.json`. Alle anderen Quellen,
Originalauflösungen und Train-only-Auswahlen blieben bei der Fields-Auswahl
erhalten. Der spätere RWTH-Schritt bewahrt den bisherigen Stand mit 345.022
Dateien in `outputs/sem_corpus/sem_posttraining_v0_before_rwth_hr26k/`.

Status lesen:

```bash
cat outputs/sem_corpus/sem_posttraining_v0_job/status.json
tail -n 10 outputs/sem_corpus/sem_posttraining_v0_job/pipeline.log
```

## V0-Transformation beim Laden

`src/sem_corpus_v0/data.py::prepare_training_data` ist eine ausdrückliche neue
Datenanbindung. Sie liefert eine separate Konfigurationskopie, Dataset, originales
Meta-Masking/Collate und eine Quellen-/Manifestidentität. Sie wird im laufenden
NFFA-Large-Pilot nicht automatisch aktiviert.

**Standardausgabe global224×224**, wahlweise518×518. Lokale Ausgabe ist passend
98×98 bzw.224×224. Anzahl der Views, Graustufenumwandlung, milde Photometrie und
Normalisierung stammen aus der bestehenden SEM-Pipeline. DINO/iBOT/KoLeo, Teacher,
EMA und Optimizer werden nicht neu implementiert.

Die neue Crop-Untergrenze gilt auf dem Bild **nach** der unveränderten
Metadatenstreifen-Entfernung:

| Kurze Quellseite | Effektive Crop-Regel |
| --- | --- |
| ≥518 | Bisherige globalen/lokalen Scale-Verteilungen unverändert |
| 224–517 | Jede Crop-Seite mindestens224 Quellpixel; Scale-Bereich wird entsprechend begrenzt |
| <224 | Vollständige kurze Quellseite behalten; bei rechteckigen Bildern darf die Position entlang der langen Seite variieren |

Auch128×128 und512×512 dürfen damit in224er- oder518er-Ausgaben transformiert
werden. Die Ausgabegröße entscheidet nicht über die Eignung des Originals.
Alle Größenänderungen passieren im RAM beim Laden. Keine hochskalierten
Dateikopien und keine fest gecachten zufälligen Views. `orientation_invariant`
bleibt für sämtliche Quellen zunächst false.

Ein CPU-Crop-Check nach Fertigstellung des Manifests:

```bash
conda activate sem-gpu
python scripts/preview_sem_posttraining_v0.py \
  --manifest outputs/sem_corpus/sem_posttraining_v0/training_manifest.csv
```

Optional518:

```bash
python scripts/preview_sem_posttraining_v0.py \
  --manifest outputs/sem_corpus/sem_posttraining_v0/training_manifest.csv \
  --training-resolution 518
```

Beide Befehle erzeugen nur einige CPU-Views und zeigen deren Formen an. Sie
starten kein Modelltraining und speichern keine Bildkopien. Die Datenanbindung
ist separat vorbereitet; neue wissenschaftliche Läufe benötigen ein eigenes
auflösungsgebundenes Rezept, gültige Korpusfreigabe und einen passenden A100-Check.
Der generische historische Receipt verlangt noch andere Ausschlussregeln und
wird deshalb nicht mit einer unzutreffenden `verified`-Bescheinigung umgangen.

## Implementierung und Tests

- `scripts/build_sem_posttraining_manifest.py`: Builder und Audit-Wartefunktion,
  nur Metadaten plus NFFA-Trainingsheader; vorhandener eingefrorener Headerprüfer.
- `configs/datasets/sem_posttraining_manifest_v0.json`: konkrete Quellen,
  Split-Nachweise, Mindestgröße und V0-Standardauflösung.
- `src/sem_corpus_v0/data.py`: ausschließlich neue, explizite Daten-/Crop-Anbindung.
- `scripts/preview_sem_posttraining_v0.py`: CPU-Ansichtsprüfung mit224/518-Option.
- `tests/test_sem_posttraining_manifest.py`: Audit-Automatik, Größenflags,
  Farbzulassung, LITHOS-Ausschluss und Train-only-Sicherheit.
- `tests/test_sem_corpus_v0.py`: Crop-Untergrenzen, identische Verteilung für
  große Quellen, unveränderte Originale und komplette kleine Meta-SSL-Schritte
  für224 und518 auf CPU.

Die18 gezielten Tests bestanden in20.59s. Das ist kein GPU-Leistungsnachweis
und keine visuelle Freigabe des späteren gemischten Korpus.
