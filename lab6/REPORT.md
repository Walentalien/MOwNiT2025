# Lab 6 - Singular Value Decomposition for Information Retrieval

This lab implements a simple search engine based on Latent Semantic Indexing (LSI).
The program downloads the **20 Newsgroups** dataset from scikit-learn which
contains over 11k English documents. For each document we compute a bag-of-words
representation, weight it with inverse document frequency (IDF) and normalise the
vectors.

The script `lsi_search.py` allows querying the collection and compares results
obtained directly from the TF–IDF matrix with results using a low rank
approximation obtained from Singular Value Decomposition (SVD).

Example usage:

```bash
python3 lsi_search.py --query "space shuttle" --k 100
```

Outputs the most similar documents with and without SVD based on cosine
similarity. The rank `k` controls the dimension of the reduced latent space.

During experiments the reduced representation offered clearer results for
moderate values of `k` (around 100), effectively filtering noise present in the
original term-by-document matrix.
