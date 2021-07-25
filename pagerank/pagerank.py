import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    tm = {p: 0 for p in corpus}
    ls = corpus[page]
    if len(ls) == 0: damping_factor = 0
    for p in ls: tm[p] = damping_factor / len(ls)
    for p in corpus: tm[p] += (1 - damping_factor) / len(corpus)
    return tm


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pr = {p: 0 for p in corpus}
    p = random.choice(list(corpus.keys()))
    for _ in range(n):
        p = random.choices(*zip(*transition_model(corpus, p, damping_factor).items()))[0]
        pr[p] += 1
    pr = {p: r / n for (p, r) in pr.items()}
    return pr


def iterate_pagerank(corpus, damping_factor, epsilon=0.001):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    corpus = {p : ls if len(ls) > 0 else set(corpus.keys()) for (p, ls) in corpus.items()}
    pr = {p: 1 / len(corpus) for p in corpus}
    while True:
        npr = {p: (1 - damping_factor) / len(corpus) + damping_factor * sum(pr[pp] / len(pls) for (pp, pls) in corpus.items() if p in pls) for p in corpus}
        if next((False for p in corpus if abs(npr[p] - pr[p]) > epsilon), True): break
        pr = npr
    return pr


if __name__ == "__main__":
    main()
