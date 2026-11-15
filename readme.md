# langfit

Approximating fitness landscapes over the space of English words.

Theory behind the model(s) are fully described in the webpage hosted on GitHub Pages; please access this at 02dtan.github.io/langfit/ to read.

I have reviewed the formulation of the Stephens & Bialek (2007) reconstruction and it appears correct. I have less confidence in an equivalent claim about the "new model", where additional theory is used to account for words of varying length (relying on importance sampling over Twitter data with tokens of length N ≤ 8), as well as to incorporate additional letterwise operations such as swaps and indels. Until this readme is updated, please treat the "new model" as a heuristic exploration of letter-space in the English language under the observable frequency of words. An ongoing effort involves scaling up the span of the data used to fit these models, for increased resolution and (hopefully) interesting time dynamics which might reveal something relevant to claims commonly thrown around in semiotics and memetics.

Work performed by a multitude of agents spanning model classes from Anthropic, OpenAI, and DeepSeek with omp harness.

DT 11-15-2026
