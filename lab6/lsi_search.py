import argparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
import numpy as np


def build_matrix(docs, min_df=2):
    vectorizer = CountVectorizer(stop_words='english', min_df=min_df)
    term_counts = vectorizer.fit_transform(docs)
    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
    tfidf_matrix = tfidf.fit_transform(term_counts)
    tfidf_matrix = normalize(tfidf_matrix, norm='l2', axis=1)
    return tfidf_matrix, vectorizer, tfidf


def search(query, matrix, vectorizer, tfidf, topk=5):
    q_counts = vectorizer.transform([query])
    q_vec = tfidf.transform(q_counts)
    q_vec = normalize(q_vec, norm='l2', axis=1)
    scores = matrix @ q_vec.T
    scores = scores.toarray().ravel()
    top_idx = np.argsort(scores)[::-1][:topk]
    return top_idx, scores[top_idx]


def apply_svd(matrix, k=100):
    svd = TruncatedSVD(n_components=k, random_state=0)
    reduced = svd.fit_transform(matrix)
    reduced = normalize(reduced, norm='l2', axis=1)
    return reduced, svd


def search_svd(query, reduced_matrix, vectorizer, tfidf, svd, topk=5):
    q_counts = vectorizer.transform([query])
    q_vec = tfidf.transform(q_counts)
    q_vec = svd.transform(q_vec)
    q_vec = normalize(q_vec, norm='l2', axis=1)
    scores = reduced_matrix @ q_vec.T
    scores = scores.ravel()
    top_idx = np.argsort(scores)[::-1][:topk]
    return top_idx, scores[top_idx]


def main():
    parser = argparse.ArgumentParser(description="Simple LSI search using SVD")
    parser.add_argument('--k', type=int, default=100, help='SVD rank')
    parser.add_argument('--query', type=str, required=True, help='Search query')
    args = parser.parse_args()

    news = fetch_20newsgroups(subset='train')
    docs = news.data

    matrix, vectorizer, tfidf = build_matrix(docs)
    idx, scores = search(args.query, matrix, vectorizer, tfidf)
    print('Top results without SVD:')
    for i, s in zip(idx, scores):
        print(f'[{i}] score={s:.4f} -- {news.filenames[i]}')

    reduced_matrix, svd = apply_svd(matrix, k=args.k)
    idx, scores = search_svd(args.query, reduced_matrix, vectorizer, tfidf, svd)
    print('\nTop results with SVD rank', args.k)
    for i, s in zip(idx, scores):
        print(f'[{i}] score={s:.4f} -- {news.filenames[i]}')


if __name__ == '__main__':
    main()
