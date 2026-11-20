# Residual-Heads mit 10 % Dropout

Stand:18.11.2026. Änderung der eingefrorenen **Downstream-Heads** des NFFA-Pilots.
Die Meta-SSL-Projektionsköpfe, SSL-Losses und Backbone-Updates bleiben gleich.

## Architektur

Der bisherige Klassifikations-MLP hatte drei Hidden-Layer512/256/128 mit
Linear → BatchNorm1d → ReLU → Dropout0.3, anschließend den linearen
10-Klassen-Ausgang. Er hatte keine Residualverbindungen.

Das neue Profil `residual_v1` verwendet weiterhin die Hauptbreiten512/256/128:

```text
Encoderfeature (1024 bei Large, 384 bei Small)
  → Residualblock512 → Residualblock256 → Residualblock128 → Linear128→10

Block: y = shortcut(x) + Dropout0.1(ReLU(BatchNorm(Linear(x))))
Shortcut: Identität bei gleicher Breite, sonst Linear(in,out,bias=False)
```

Bei den üblichen Eingabedimensionen kommen drei gelernte Shortcut-Projektionen
hinzu. Es sind drei Haupt-Hidden-Layer und insgesamt sieben lineare Module
einschließlich Projektionen und Klassifikator. Der Dropout betrifft nur den
transformierten Zweig; der Shortcut bleibt direkt.

**CNN8 war bereits residual und hatte bereits Dropout2d(0.1):**
1×1-Projektion auf128 Kanäle, vier Residualblöcke mit je zwei räumlichen
3×3-Convs und anschließend1×1-Klassifikator. Der Name zählt die acht räumlichen
Convs. Die jeweils letzte Block-Conv ist nullinitialisiert, der Block anfänglich
eine Identität. Diese Architektur wird weiterverwendet.

Beim MLP ist Dropout elementweise, beim CNN kanalweise. Beide verwenden0.1
im Training und deaktivieren Dropout bei der Evaluation. MLP-Lernrate1e-3,
AdamW, Weight Decay1e-4, Batch256, maximal100 Epochen, Patience15, StandardScaler
und Auswahl anhand der Validierungs-Macro-F1 bleiben gleich. LogReg und lineare
Segmentierung behalten ihr bisheriges Rezept.

## Einbindung

Der NFFA-Pilot setzt `evaluation.head_profile: residual_v1` und verwendet dieses
Profil für Original-Large und alle vier SSL-Arme an allen Auswertungszeitpunkten.
Neuer Studienordner:
`outputs/sem_ssl_regularization/nffa_split_a_vitl14_residual_heads_v1_seed42/`.

Der manuelle A100-Start bleibt:

```bash
cd /path/to/nffa-sem-dinov2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sem_cellstyle_nffa_pilot.sh
```

Das startet nach den Vorprüfungen die Studie; der Befehl wurde bei dieser Änderung
nicht ausgeführt. Weiterhin Split A,12.801 Trainingsbilder, Seed42, vier Arme
mit je1.001 Updates und Checkpoints0/101/501/1001; NFFA-A- und Aachen-Validation.

Der gemeinsame Evaluationsrunner bietet `--head-profile residual_v1`.
`--head-profile legacy` bleibt sein CLI-Standard für ältere Wrapper und
Reproduktionen mit dem ursprünglichen MLP ohne Residualverbindungen/Dropout0.3.
Der aktuelle NFFA-Pilot wählt das neue Profil ausdrücklich. Alte Ergebnisse
werden nicht mit neuen Head-Konfigurationen zusammengeführt. Resume und
Evaluation prüfen die gespeicherte MLP-Konfiguration einschließlich Version
und Dropout. Alle bestehenden Feature-/Quellenprüfungen bleiben aktiv.

## Dateien und Prüfung

- `src/downstream/residual_heads.py`: neuer Residual-MLP und seine Blöcke.
- `src/downstream/probes.py`: Profilkonfiguration, bestehender Trainer und
  Checkpoint-/Resume-Prüfung. `HEAD_CONFIGS` bleibt das historische Rezept.
- `scripts/evaluate_sem_encoder.py`: Profilwahl und Konfigurationsprüfung.
- `scripts/evaluate_sem_cellstyle.py`: Profil an alle Checkpoints weiterreichen.
- `src/ssl/reporting.py`, `src/sem_cellstyle/{reporting,regularization_report}.py`:
  passende Head-Konfiguration für Originalreferenz und alle Arme prüfen.
- NFFA-Pilot-YAML/Bash-Einstieg: neues Profil und eigener Ausgabeordner.
- `tests/test_residual_heads.py`: tatsächliche Skip-Pfade und Gradienten,
  384-/1024-Eingänge, CNN-Dropout/Identität, Legacy-Profil, CPU-Training,
  exakter Resume und gemeinsame Runner-Integration.

Nachweise: das lokale Prüfarchiv.
Gezielte Prüfung: **77 Tests bestanden**, einschließlich echter Skip-Pfade und
Gradienten, des identischen CPU-MLP-Resumes, der Legacy-Regression und der
gemeinsamen Runner-Integration. `plan.json` bestätigt das neue Profil im
tatsächlichen NFFA-Einstieg. Die Gesamtsuite steht in `pytest_full.txt`.
Gesamtsuite: **289 bestanden, 3 übersprungen, 3 Subtests bestanden**.
Noch keine neuen GPU-Durchsatz- oder Qualitätswerte für diese Heads.
