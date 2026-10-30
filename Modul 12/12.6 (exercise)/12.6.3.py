from transformers import pipeline

# Label bawaan model CoNLL-03 -> nama yang diminta soal
LABEL_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}

_ner = None


def _get_ner():
    global _ner
    if _ner is None:  # model dimuat sekali saja
        _ner = pipeline(
            task="ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple",
            device=-1,
        )
    return _ner


def extract_entities_from_corpus(texts: list[str], min_score: float = 0.5) -> dict:
    """Return {'PERSON': [...], 'ORG': [...], 'LOC': [...]} berisi nilai unik dari seluruh teks."""
    result = {label: [] for label in LABEL_MAP.values()}
    seen = {label: set() for label in LABEL_MAP.values()}

    for entities in _get_ner()(texts):
        for ent in entities:
            label = LABEL_MAP.get(ent["entity_group"])
            if label is None or ent["score"] < min_score:
                continue  # abaikan MISC dan prediksi berskor rendah
            value = ent["word"].strip()
            key = value.lower()
            if value and key not in seen[label]:
                seen[label].add(key)
                result[label].append(value)
    return result


if __name__ == "__main__":
    corpus = [
        "Anthropic was founded in San Francisco by Dario Amodei and Daniela Amodei.",
        "Sundar Pichai leads Google, which is headquartered in Mountain View.",
        "OpenAI, based in San Francisco, was co-founded by Sam Altman.",
    ]
    for label, values in extract_entities_from_corpus(corpus).items():
        print(f"{label:<7}: {values}")