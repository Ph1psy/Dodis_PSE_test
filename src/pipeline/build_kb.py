import argparse
import json
import os
import sqlite3

import spacy
from spacy.kb import InMemoryLookupKB

# Für lokales Testen: Limit auf eine kleine Anzahl Entities setzen.
# Auf None setzen für vollständiges Training (z.B. auf Ubelix).
LOCAL_TEST_LIMIT = 10000


def get_label(data, qid):
    """Gibt den besten verfügbaren Label-String für eine Entity zurück."""
    labels = data.get("labels", {})
    for lang in ["de", "en"]:
        label = labels.get(lang)
        if isinstance(label, dict):
            label = label.get("value")
        if isinstance(label, str) and label.strip():
            return label
    return qid  # Fallback auf Q-ID falls kein Label vorhanden


def compute_vector(nlp, text):
    """
    Berechnet einen 768-dimensionalen Vektor für einen Text via BERT-Transformer.
    Gibt einen Nullvektor zurück falls die Berechnung fehlschlägt.
    """
    try:
        doc = nlp.make_doc(text[:512])  # BERT-Limit beachten
        nlp.get_pipe("transformer")(doc)
        if doc._.trf_data is not None and doc._.trf_data.tensors:
            # Letztes Transformer-Layer, erste Token-Sequenz, Mittelwert über alle Tokens
            vector = doc._.trf_data.tensors[-1][0].mean(axis=0).tolist()
            return vector
    except Exception:
        pass
    return [0.0] * 768


def build_kb(database, output_path, limit=None):

    assert database is not None, "DB_PATH ist None!"
    assert output_path is not None, "KB_OUTPUT_PATH ist None!"
    assert isinstance(
        database, str
    ), f"DB_PATH muss ein String sein, ist aber: {type(database)}"
    assert isinstance(
        output_path, str
    ), f"KB_OUTPUT_PATH muss ein String sein, ist aber: {type(output_path)}"
    assert os.path.isfile(database), f"Datenbankdatei nicht gefunden: {database}"

    print(f"Datenbank:   {database}")
    print(f"Ausgabe:     {output_path}")
    if limit is not None:
        print(f"Test-Limit:  {limit:,} Entities (LOCAL_TEST_LIMIT aktiv)")
    else:
        print("Limit:       Keins – vollständiger Durchlauf")

    # Modell laden
    print("\nLade spaCy-Modell (de_dep_news_trf)...")
    nlp = spacy.load("de_dep_news_trf")
    assert nlp is not None, "spaCy-Modell konnte nicht geladen werden!"

    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=768)
    assert kb is not None, "KnowledgeBase konnte nicht erstellt werden!"

    # Datenbankverbindung
    conn = sqlite3.connect(database)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='entities'")
    assert (
        cur.fetchone() is not None
    ), "Tabelle 'entities' existiert nicht in der Datenbank!"

    cur.execute("SELECT COUNT(*) FROM entities")
    total_rows = cur.fetchone()[0]
    assert total_rows > 0, "Tabelle 'entities' ist leer!"
    print(f"Datenbank enthält {total_rows:,} Einträge gesamt")

    # Rows laden (mit oder ohne Limit)
    if limit is not None:
        cur.execute("SELECT id, data FROM entities LIMIT ?", (limit,))
    else:
        cur.execute("SELECT id, data FROM entities")
    rows = cur.fetchall()
    print(f"Verarbeite {len(rows):,} Einträge\n")

    # ----------------------------------------------------------------
    # Schritt 1: Entities registrieren mit echten Vektoren
    # ----------------------------------------------------------------
    print("Schritt 1: Registriere Entities und berechne Vektoren...")
    full_alias_map = {}
    registered_ids = set()

    for i, (qid, raw_json) in enumerate(rows):

        assert qid is not None, "qid ist None!"
        assert raw_json is not None, f"raw_json ist None für qid={qid}!"

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            print(f"  WARNUNG: JSON-Fehler für qid={qid}: {e} – übersprungen")
            continue

        # Entity mit echtem Vektor registrieren
        if qid not in registered_ids:
            label = get_label(data, qid)
            vector = compute_vector(nlp, label)
            kb.add_entity(entity=qid, entity_vector=vector, freq=3)
            registered_ids.add(qid)

        # Namen und Aliases sammeln
        names = set()
        labels_data = data.get("labels", {})
        aliases_data = data.get("aliases", {})

        for lang in ["de", "en"]:
            label_data = labels_data.get(lang)
            if isinstance(label_data, dict):
                label = label_data.get("value")
            elif isinstance(label_data, str):
                label = label_data
            else:
                label = None

            if label and label.strip():
                names.add(label)

            for entry in aliases_data.get(lang, []):
                if isinstance(entry, dict):
                    val = entry.get("value")
                elif isinstance(entry, str):
                    val = entry
                else:
                    val = None
                if val and val.strip():
                    names.add(val)

        # Alias-Map befüllen
        for name in names:
            if name not in full_alias_map:
                full_alias_map[name] = []
            if qid not in full_alias_map[name] and len(full_alias_map[name]) < 30:
                full_alias_map[name].append(qid)

        # Fortschritt
        if (i + 1) % 500 == 0:
            print(f"  {i + 1:,} / {len(rows):,} verarbeitet...")

    assert len(registered_ids) > 0, "Keine Entitäten wurden registriert!"
    assert len(full_alias_map) > 0, "Alias-Map ist leer – keine Namen gefunden!"
    print(f"  -> {len(registered_ids):,} Entities registriert")

    # ----------------------------------------------------------------
    # Schritt 2: Aliases in KB schreiben
    # ----------------------------------------------------------------
    print(f"\nSchritt 2: Schreibe {len(full_alias_map):,} Aliases in die KB...")

    for name, qid_list in full_alias_map.items():
        assert len(qid_list) > 0, f"qid_list ist leer für name='{name}'!"

        # Nur registrierte IDs verwenden
        valid_qids = [q for q in qid_list if q in registered_ids]
        if not valid_qids:
            continue

        probs = [1.0 / len(valid_qids)] * len(valid_qids)
        kb.add_alias(alias=name, entities=valid_qids, probabilities=probs)

    # ----------------------------------------------------------------
    # KB speichern
    # ----------------------------------------------------------------
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    kb.to_disk(output_path)
    assert os.path.exists(output_path), f"KB-Datei wurde nicht erstellt: {output_path}"

    conn.close()
    print(
        f"\nFertig! {len(registered_ids):,} Entities gespeichert unter: {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--database", required=True, help="Pfad zur SQLite-Datenbank"
    )
    parser.add_argument(
        "-o", "--outputPath", required=True, help="Ausgabepfad für die KB"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Vollständiger Durchlauf ohne Limit (für Ubelix). Standard: LOCAL_TEST_LIMIT",
    )
    args = parser.parse_args()

    limit = None if args.full else LOCAL_TEST_LIMIT

    if os.path.isfile(args.database):
        build_kb(args.database, args.outputPath, limit=limit)
    else:
        print(f"Datenbankpfad nicht gefunden: {args.database}")
