import re
import seaborn as sns
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from utils import *


def plot_top_non_stopwords_barchart(text):
    stop = set(stopwords.words('english'))

    corpus = [word for word in text]

    counter = Counter(corpus)
    most = counter.most_common()

    x, y = [], []
    for word, count in most[:40]:
        if (word not in stop):
            x.append(word)
            y.append(count)

    sns.barplot(x=y, y=x)
    plt.title("Top most used non-stopwords")
    plt.savefig("../plots/top_non_stopwords")
    plt.clf()

    topMostNonStopwords = (most[0:10])

    return topMostNonStopwords


def write_to_txt(results):
    filename = "../results/statistics/stories_stats.txt"

    with open(filename, 'w') as f:
        f.write(results)


def main():

    corpus_path = "../data/corpus/AesopFables.json"
    with open(corpus_path) as f:
        data = json.load(f)
    stories = data["stories"]

    sentences = []
    words = []
    chars = []
    results = ""

    storiesCount = 0

    for story in stories:

        # Concatenate the stories
        concat_story = concatenate_sentences(story["story"])
        words += concat_story.split()
        chars += [char for i in concat_story for char in i]
        sentences += re.split(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=\.|\?)", concat_story)
        storiesCount += 1

    topNonStopwords = plot_top_non_stopwords_barchart(words)

    avgLetters = len(chars) / len(words)
    results += ("Average letters per word: " + str(avgLetters)) + "\n"

    avgWords = len(words) / len(sentences)
    results += ("Average words per sentence: " + str(avgWords)) + "\n"

    avgSentences = len(sentences) / storiesCount
    results += ("Average sentences per story: " + str(avgSentences)) + "\n"

    results += ("Characters: " + str(len(chars))) + "\n"
    results += ("Words: " + str(len(words))) + "\n"
    results += ("Sentences: " + str(len(sentences))) + "\n"
    results += ("Top 10 non-Stopwords: " + str(topNonStopwords)) + "\n"
    write_to_txt(results)


if __name__ == "__main__":
    main()