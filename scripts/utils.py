import json
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