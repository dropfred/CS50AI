import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}
    if os.path.isdir(directory):
        with os.scandir(directory) as sd:
            for e in sd:
                if e.is_file and e.path.endswith(".txt"):
                    with open(e.path, 'r', encoding="utf8") as f:
                        files[os.path.basename(e.path)] = f.read()
    return files

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stop = set(nltk.corpus.stopwords.words('english'))
    words = []
    for w in document.split():
        w = w.lower().strip(string.punctuation)
        if w and w not in stop:
            words.append(w)
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}
    documents = {f: set(ws) for f, ws in documents.items()}
    words = set()
    for ws in documents.values():
        words |= ws
    for w in words:
        idfs[w] = math.log(len(documents) / sum(1 if w in ws else 0 for ws in documents.values()))
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfs = {}
    for f, ws in files.items():
        tfidfs[f] = sum(ws.count(w) * idfs[w] if w in idfs else 0 for w in query)
    return sorted(tfidfs, key=lambda f: tfidfs[f], reverse=True)[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    mwms = {}
    qtds = {}
    for s, ws in sentences.items():
        qtds[s] = sum(1 if w in query else 0 for w in ws) / len(s)
        mwms[s] = sum(idfs[w] if w in ws and w in idfs else 0 for w in query)
    return sorted(mwms, key=lambda s: (mwms[s], qtds[s]), reverse=True)[:n]


if __name__ == "__main__":
    main()
