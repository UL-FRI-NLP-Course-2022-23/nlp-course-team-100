# Specific Imports
from utils import *


def compare_ner_coref(path_ner, path_coref):
    ner = load_json_from_file(path_ner)
    coref = load_json_from_file(path_coref)

    differences = []

    j = 0

    for entities in ner:
        differences.append((list(set(entities) - set(coref[j]))))

        j += 1

    return differences


def main():
    # Paths
    stanza_ner_path = "../results/stanza/ner"
    stanza_ner_coref_path = "../results/stanza/ner_coref"
    spacy_ner_path = "../results/spacy/ner"
    spacy_ner_coref_path = "../results/spacy/ner_coref"

    num_ner = len([name for name in os.listdir(stanza_ner_path) if str(name).endswith('.json')])
    num_coref = len([name for name in os.listdir(stanza_ner_coref_path) if str(name).endswith('.json')])

    if num_ner == num_coref:
        result_stanza = compare_ner_coref(stanza_ner_path, stanza_ner_coref_path)

    num_ner = len([name for name in os.listdir(spacy_ner_path) if str(name).endswith('.json')])
    num_coref = len([name for name in os.listdir(spacy_ner_coref_path) if str(name).endswith('.json')])

    if num_ner == num_coref:
        result_spacy = compare_ner_coref(spacy_ner_path, spacy_ner_coref_path)


    print(result_stanza)
    print(result_spacy)


if __name__ == "__main__":
    main()