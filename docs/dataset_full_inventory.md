# Vollständiger Bildaudit: Auflösung und exakte Dateiduplikate

Das Skript [audit_image_inventory.py](../scripts/audit_image_inventory.py) prüft
die neun freigegebenen Datensätze unter `/path/to/datasets`:
`lithos`, `cross_microscope_steel`, `cigrocksem`, `uhcsdb`, `microal`,
`jlu_cathodes`, `bil_nmc811`, `bil_si_graphite`, `kit_sem_powder`.

Mit `--datasets` können seit18.11.2026 explizit auch `drill_cuttings_sem`,
`fields_metal_cement`, `porous_zirconia_fibsem`, `rwth_steel_sem` und
`matscifig_sem` in einem **separaten** Audit erfasst werden. Ohne diese Option
bleibt der Standardumfang unverändert. Der reale Ergänzungslauf wurde mit zwei
I/O-Threads und `--skip-hashes` gestartet unter
`/path/to/datasets/_audit/additional_sources_2026-11-18/inventory/`.
Startprotokoll, Tests und Statusbefehle.
Der ältere vollständige Audit wird nicht erweitert oder überschrieben.

Die neuen Pfadregeln unterscheiden Drill-Cuttings-Targets von den tatsächlichen
`RawImage_downscaled`-Bildern, erhalten veröffentlichte Split-Hinweise und
gruppieren rotationsbenannte Cement-Varianten bzw. FIB-SEM-Stacks vorsichtig.
MatSciFig erfordert einen exakten lokalen Manifesttreffer mit
`subtype_valid=True`; DOI und benannte Figuren-/Panelgruppen werden als
Metadaten-Gruppierung protokolliert. Dies ist keine visuelle Qualitäts- oder
Leakage-Prüfung und gibt keine Daten automatisch zum Training frei.

Der alte Audit ist abgeschlossen. Sein eingefrorener Quellcode liegt unter
`full_inventory/audit_source_v3.py`; wegen der Quellenbindung für ein späteres
Rendern/Fortsetzen diesen damaligen Quellstand verwenden. Die nachstehenden
historischen Befehle mit dem aktuellen Repository-Skript sind kein Auftrag,
die alte Datenbank mit geändertem Quellcode neu zu öffnen.

`nffa_sem_100` und `aachen_heerlen` sind fest ausgeschlossen. Das Skript listet
ihre Verzeichnisse nicht auf und öffnet oder hasht keine ihrer Dateien. Auch
Symlinks werden nicht verfolgt. Es verändert keine Rohdateien, erzeugt keine
Trainingssplits und startet keine Modelle oder GPU-Prozesse.

Der vorherige strukturelle Stichprobenaudit liegt unter
`/path/to/datasets/_audit/structure_report.md`. Seine Hinweise auf
Maskenpfade, unbekannte KIT-Varianten, optische LITHOS-Bilder und
Proben-/Slice-/Tile-Strukturen bilden die Grundlage der vorsichtigen Pfadregeln.

## Ausführen und fortsetzen

Ein neuer Audit benötigt einen leeren Ausgabeordner:

```bash
cd /path/to/nffa-sem-dinov2
python \
  scripts/audit_image_inventory.py --workers 2 \
  --output /path/to/datasets/_audit/full_inventory
```

Nach Unterbrechung denselben Befehl um `--resume` ergänzen. Fertige
Header-/Hash-Belege werden wiederverwendet. Discovery wird spätestens nach 64
Verzeichnissen oder zwei Sekunden gespeichert, Header und Hashes nach je 32
Dateien. Nur der zuletzt noch nicht gespeicherte Block kann erneut gelesen
werden. Root, Skript und Regeln müssen zum gespeicherten Audit-Vertrag passen.
Ein neuer Datenstand benötigt einen neuen Ausgabeordner.

`--resume --render-only` erzeugt die Berichte allein aus der SQLite-Datenbank,
ohne Rohdaten zu öffnen. Ein Lock verhindert zwei gleichzeitige Audit-Writer.
`--workers` erlaubt ausschließlich 1 oder 2 I/O-Threads. Es werden keine Pakete
installiert; Pillow und SQLite sind in `sem-gpu` vorhanden.

**Aktuelle Nutzerentscheidung vom 18.11.2026: SHA-256 abbrechen.** Der reale
Audit wird mit `--resume --skip-hashes` nur für Header/Rollen/Auflösungen
fortgesetzt. Die Entscheidung wird in der Zustandsdatenbank gespeichert und
bleibt auch bei späterem `--resume` ohne das Flag wirksam. Vorhandene Hashes
bleiben erhalten; noch offene Hashes tragen `skipped_by_request`. Es wird
keine vollständige Duplikataussage oder deduplizierte Kandidatenzahl erzeugt.
`inventory_complete` und `exact_hash_check_complete` sind getrennte Felder.
Die aktuelle Fortsetzung lautet:

```bash
python \
  scripts/audit_image_inventory.py --workers 2 --resume --skip-hashes \
  --output /path/to/datasets/_audit/full_inventory
```

## Methode und Bedeutung der Zählungen

1. `os.scandir()` listet alle regulären Dateien der neun Bäume auf. Alle
   Dateipfade stehen in `all_files.csv`; bekannte Bildsuffixe werden zusätzlich
   in `images_manifest.csv` erfasst. OS-Ressourcenforks sind keine Bilder.
2. Pillow liest Format, Modus, Kanäle und Originalabmessungen aus dem Header
   der ersten Seite/des ersten Frames. Es findet keine vollständige
   Pixeldekodierung, Skalierung oder Konvertierung statt. Header-Erfolg ist
   deshalb kein Beweis für fehlerfrei dekodierbare Pixel oder weitere TIFF-Seiten.
3. Eine Mindestauflösung ist erfüllt, wenn **beide** Seiten mindestens 128 Pixel
   haben. Die Klassen sind `<128`, `≥128 und <224`, `≥224 und <448`, `beide ≥448`,
   jeweils nach der kleineren Seite. Nicht lesbare Header stehen separat.
4. Rollen beruhen auf bekannten Dataset-Pfaden und Namensregeln. Masken und
   Labels haben Vorrang vor Mikroskopie-Hinweisen. Unklare Bilder bleiben
   `unknown`. Geeignete Mikroskopie-Crops/Tiles dürfen `derived_image` und
   gleichzeitig `is_microscopy=true` sein; Overlays und Vorschaubilder nicht.
5. `ssl_candidate=true` erfordert einen lesbaren Header, Mindestauflösung und
   eine wahrscheinliche Mikroskopierolle. Es ist eine Audit-Kategorie und keine
   endgültige Freigabe für Training. KIT BO/RL/Phenom bleiben unbekannt;
   LITHOS PPL/XPL zählen als optische Mikroskopie, nicht als SEM.
6. Dateigrößen werden **global über alle Bildrollen und Datensätze** gruppiert.
   Nur Gruppen mit mindestens zwei Dateien benötigen SHA-256. Hashes werden in
   1-MiB-Blöcken gelesen; Dateigröße, mtime, ctime und inode müssen zum
   gespeicherten Discovery-Stand passen. Es gibt keinen pHash, Pixelvergleich
   oder quadratischen Paarvergleich.

Nach Discovery steht die Größenfilterung bereits fest. Bei einem noch nicht
geprüften Header mit ausstehendem Hash liest ein Worker zuerst den SHA-256
streamingbasiert und anschließend den Pillow-Header über **dieselbe geöffnete
Originaldatei**. Es gibt keinen zweiten Dateizugriff; die für den Header
benötigten Seiten wurden durch den Hash bereits gelesen. Dateien mit einmaliger
Größe erhalten nur einen Headerzugriff mit POSIX-Hinweis für zufälligen Zugriff,
um spekulatives Vorablesen ganzer Bilder zu vermeiden. Dieser Hinweis ist
optional und ändert weder Headerdaten noch Hashwerte. Fehlende Hashes zu schon
gespeicherten Headern werden beim Fortsetzen separat ergänzt.

Ein einmaliges Dateigrößenmaß beweist, dass keine andere erfasste Bilddatei
byteidentisch sein kann. Ein identischer SHA-256 weist byteidentischen Inhalt
nach. Gleiche Namen, ähnliche Crops, benachbarte Slices und unterschiedlich
kodierte Versionen werden dadurch nicht gleichgesetzt.

`exact_duplicates.csv` enthält jeden Gruppenteilnehmer, Datensatz, Rolle,
Gruppengröße sowie Flags für datensatz-/rollenübergreifende Gruppen und KIT
train/val. Die Haupttabelle zählt als „Exakte Duplikate“ **überzählige Dateien
innerhalb des jeweiligen Datensatzes**, also `n−1` je Gruppe. Globale und
rollenspezifische Zählungen stehen separat im Bericht/JSON. Niemand entfernt
diese Dateien automatisch.

## Ergebnisse und Zwischenstände

Im Ausgabeordner liegen:

- `images_manifest.csv`: Bilddateien, Rollen, Header, Auflösung und Hashstatus.
- `dataset_summary.csv`: Zahlen pro Dataset, einschließlich unbekannter Rollen,
  Header-/Hashfehler und geeigneter Mikroskopie-Kandidaten.
- `resolution_summary.csv`: Verteilung pro Dataset und Rolle; Minima,
  25./50./75./95. Perzentile, Maxima und Mittelwerte von Breite, Höhe, Pixelzahl.
- `exact_duplicates.csv`: bestätigte SHA-256-Gruppen mit allen Teilnehmern.
- `report.md` und `audit_summary.json`: wissenschaftliche Zusammenfassung,
  Vollständigkeitsstatus, KIT train/val und verbleibende Unklarheiten.
- `all_files.csv`, `scan_issues.csv`: übrige reguläre Dateien und Scanprobleme.
- `audit_state.sqlite3`, `audit.log`, `progress.json`, `audit.lock`:
  Fortsetzungsstand, Laufprotokoll und Writer-Lock.

Unbekannte oder falsche Suffixe sowie Inhalte von Archiven, Arraydateien und
Dokumentcontainern sind keine automatisch erkannten Rasterbilder. Sie bleiben
in der Dateiliste sichtbar. Keine Aussage über visuelle Bildqualität,
unabhängige Proben oder mögliche Ähnlichkeits-Leakage wird daraus abgeleitet.
Fehler und offene Prüfungen führen ausdrücklich zu `complete=false` und
Exitcode 2, ein sauberer Abschluss zu 0, eine Unterbrechung zu 130.

## CPU-Tests

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q \
  tests/test_image_inventory_audit.py tests/test_dataset_audit_fast.py
```

Die Tests verwenden kleine synthetische Dateien und prüfen unter anderem
Grenzauflösungen, Headerzugriffe ohne `load()`, geschützte Datensätze,
Dateigrößenfilter, Duplikate über Rollen/Datensätze, echte und nur namensgleiche
KIT-train/val-Paare, Abbruch/Fortsetzen, Quellenänderungen und unveränderte
Rohdateien.

Am 18.11.2026 bestanden die 15 gezielten Audit-Tests. Die gesamte CPU-Suite
bestand mit 222 erfolgreichen Tests, drei übersprungenen Tests und drei
erfolgreichen Subtests; die bestehenden Warnungen betreffen unter anderem
Meta/PyTorch-Deprecations. Das Protokoll liegt unter
lokales Prüfprotokoll.

Im realen Audit wurde die SQLite-Warteschlange zusätzlich mit
`CREATE INDEX directory_pending_queue ON directories(status)` indiziert.
Dadurch nutzt die Suche nach dem nächsten offenen Verzeichnis einen Index
statt wiederholt die Verzeichnistabelle zu durchsuchen. Das ist nur ein Index
in der abgeleiteten Zustandsdatenbank; Dateiauswahl, Header und Hashes bleiben
gleich. `queue_index_note.json` dokumentiert diesen Eingriff. Der Index bleibt
beim Fortsetzen dieses Audits erhalten.
Neue Audits legen diesen Index inzwischen automatisch an.

Der reale Lauf wurde nach 10.688 gespeicherten Headern ausschließlich für die
gebündelten Dateizugriffe kontrolliert unterbrochen und mit unveränderten
Rollen-/Auflösungs-/Hashregeln fortgesetzt. `io_optimization_revision.json`
verbindet die beiden Quellcode-Verträge; `audit_source.py` und
`audit_source_v2.py` archivieren beide Versionen. Zu diesem Zeitpunkt gab es noch
keine gespeicherten Bildhashes. Zwei zusätzliche Tests prüfen identische
Header-/SHA-Ergebnisse gegenüber getrennten Zugriffen und gültige Bytehashes
auch bei kaputten Bildheadern. Ein kleiner Header-I/O-Vergleich und eine
Lesemengen-Spur sind als `header_io_probe.json` und `header_read_trace.json`
gespeichert: Bei den geprüften JLU-Dateien forderte Pillow ungefähr 9 KB Header,
keine dekodierten Bildpixel an. Die kleine Stichprobe erlaubt keine belastbare
Aussage über den Geschwindigkeitsgewinn des POSIX-Hinweises.

Nach der ausdrücklichen SHA-Stornierung wurde der Lauf bei 17.504 gespeicherten
Headern und 1.824 Hashes kontrolliert angehalten. `sha_cancellation_revision.json`
und `audit_source_v3.py` dokumentieren den Übergang zur dauerhaften
Header-only-Fortsetzung. Teilweise bestätigte Duplikatgruppen bleiben erhalten,
tragen aber `group_complete=false`; Gruppengrößen sind dann nur Untergrenzen.
Offene Header werden ausdrücklich als `headers_pending` und nicht als
Lesefehler gezählt. Nach dieser Änderung bestanden 19 gezielte Audit-Tests;
die zuvor geprüfte gebündelte I/O-Version bestand die gesamte Suite mit 224
Tests, drei übersprungenen Tests und drei erfolgreichen Subtests.
