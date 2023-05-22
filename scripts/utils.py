import json
import os


def concatenate_sentences(sentences):
    concatenated_text = " ".join(sentences)
    return concatenated_text


def remove_articles(strings):
    article_words = ['a', 'an', 'the']
    output_strings = []
    for string in strings:
        words = string.split()
        filtered_words = [word for word in words if word.lower() not in article_words]
        output_strings.append(" ".join(filtered_words))
    return output_strings


def get_unique_entities(strings):
    unique_entities = []
    for string in strings:
        if string not in unique_entities:
            unique_entities.append(string)
    return unique_entities


def write_characters_to_json(unique_strings, filename):
    data = {"characters": unique_strings}
    with open(filename, 'w') as f:
        json.dump(data, f)


def write_concat_story_to_json(unique_strings, filename):
    data = {"concat_story": unique_strings}
    with open(filename, 'w') as f:
        json.dump(data, f)


def write_coref_to_json(unique_strings, filename):
    data = {"coref": unique_strings}
    with open(filename, 'w') as f:
        json.dump(data, f)


def write_diff_to_json(unique_strings, filename):
    data = {"differences": unique_strings}
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_json_from_file(path):

    files = []

    for filename in os.listdir(path):
        if filename.endswith('.json'):

            with open(path + '/' + filename, 'r') as f:
                try:
                    files.append(json.load(f)['characters'])
                except Exception as e:
                    print(e)

    return files
