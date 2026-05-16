"""
Unigram + Viterbi — desde cero
==============================
Tokenizador subword usado por SentencePiece (T5, mBART, ALBERT, XLNet y
muchos modelos multilingües).

A diferencia de BPE (bottom-up, fusionar el par más frecuente), Unigram trabaja
top-down:
  1. Empieza con un vocabulario GRANDE de candidatos
  2. Estima probabilidades de cada candidato con EM
  3. Poda las subpalabras menos útiles
  4. Repite 2-3 hasta llegar al tamaño deseado

La inferencia usa Viterbi para encontrar la segmentación de máxima probabilidad.

Cero dependencias. Solo Python estándar.

Conexión histórica con NLP clásico:
- Viterbi (Andrew Viterbi, 1967) es el mismo algoritmo de los POS taggers HMM
  de los 80-90.
- EM (Dempster, Laird, Rubin, 1977) es el motor de muchos modelos clásicos.
- El modelo unigram de subpalabras es análogo al query likelihood model en IR.
"""

import math
from collections import Counter


# Paso 1: Vocabulario inicial grande

def build_initial_vocab(text, max_subword_len=8, min_freq=2):
    """Construye vocabulario inicial con substrings frecuentes.

    En SentencePiece real se usa un seed de BPE o suffix array. Aquí
    enumeramos todas las substrings de cada palabra hasta cierta longitud
    y nos quedamos con las que aparecen al menos min_freq veces.
    """
    words = text.replace("\n", " ").split()
    word_freq = Counter(words)

    substr_freq = Counter()
    for word, freq in word_freq.items():
        marked = "▁" + word
        for i in range(len(marked)):
            for j in range(i + 1, min(i + max_subword_len + 1, len(marked) + 1)):
                substr_freq[marked[i:j]] += freq

    vocab = {s: f for s, f in substr_freq.items() if f >= min_freq}

    for char in set(text):
        if char != " ":
            vocab[char] = vocab.get(char, 1)
    vocab["▁"] = max(vocab.values())

    return vocab


# Paso 2: Viterbi — segmentación óptima dada una probabilidad por subpalabra

def viterbi_segment(text, vocab_scores):
    """Encuentra la segmentación con mayor log-probabilidad.

    El lattice se construye sobre posiciones del string. Para cada posición
    end, calculamos el mejor score sumando el mejor score acumulado hasta
    begin más el score de la subpalabra text[begin:end].
    """
    n = len(text)
    UNK_SCORE = -20.0

    best_score = [-math.inf] * (n + 1)
    best_score[0] = 0.0
    best_edge = [0] * (n + 1)

    for end in range(1, n + 1):
        char = text[end - 1]
        char_score = vocab_scores.get(char, UNK_SCORE)
        score = best_score[end - 1] + char_score
        if score > best_score[end]:
            best_score[end] = score
            best_edge[end] = end - 1

        for begin in range(max(0, end - 16), end):
            sub = text[begin:end]
            if sub in vocab_scores:
                score = best_score[begin] + vocab_scores[sub]
                if score > best_score[end]:
                    best_score[end] = score
                    best_edge[end] = begin

    tokens = []
    pos = n
    while pos > 0:
        begin = best_edge[pos]
        tokens.append(text[begin:pos])
        pos = begin
    tokens.reverse()
    return tokens


# Paso 3: EM — estimar probabilidades del vocabulario

def e_step(corpus_words, vocab_scores):
    counts = Counter()
    total_log_likelihood = 0.0
    for word, freq in corpus_words.items():
        marked = "▁" + word
        segments = viterbi_segment(marked, vocab_scores)
        for seg in segments:
            counts[seg] += freq
            total_log_likelihood += freq * vocab_scores.get(seg, -20.0)
    return counts, total_log_likelihood


def m_step(counts, prev_vocab_scores):
    total = sum(counts.values()) + len(prev_vocab_scores)
    new_scores = {}
    for sub in prev_vocab_scores:
        count = counts.get(sub, 0) + 1
        new_scores[sub] = math.log(count / total)
    return new_scores


# Paso 4: Poda

def compute_loss_if_removed(subword, counts, vocab_scores):
    """Pérdida aproximada al podar esta subpalabra: freq × |score|."""
    if len(subword) <= 1 or subword == "▁":
        return math.inf  # nunca podar caracteres ni marcador
    freq = counts.get(subword, 0)
    if freq == 0:
        return 0.0
    return freq * abs(vocab_scores.get(subword, 0))


# El tokenizador completo

class UnigramTokenizer:
    """Tokenizador Unigram con Viterbi.

    Flujo:
      tokenizer = UnigramTokenizer()
      tokenizer.train(corpus, vocab_size=300)
      ids = tokenizer.encode("texto")
      texto = tokenizer.decode(ids)
    """

    def __init__(self):
        self.vocab_scores = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, text, vocab_size=300, n_em_steps=3, shrink_factor=0.9,
              verbose=False):
        initial_vocab = build_initial_vocab(text)
        if verbose:
            print(f"Vocabulario inicial: {len(initial_vocab)} subpalabras")

        total = sum(initial_vocab.values())
        self.vocab_scores = {
            sub: math.log(freq / total) for sub, freq in initial_vocab.items()
        }

        words = text.replace("\n", " ").split()
        corpus_words = Counter(words)

        iteration = 0
        while len(self.vocab_scores) > vocab_size:
            iteration += 1

            for _ in range(n_em_steps):
                counts, log_lik = e_step(corpus_words, self.vocab_scores)
                self.vocab_scores = m_step(counts, self.vocab_scores)

            if verbose:
                print(f"Iter {iteration}: vocab={len(self.vocab_scores)}, "
                      f"log-likelihood={log_lik:.1f}")

            current_size = len(self.vocab_scores)
            target_size = max(vocab_size, int(current_size * shrink_factor))

            losses = {
                sub: compute_loss_if_removed(sub, counts, self.vocab_scores)
                for sub in self.vocab_scores
            }
            sorted_subs = sorted(losses.keys(), key=lambda s: losses[s])
            to_remove = set(sorted_subs[:current_size - target_size])
            self.vocab_scores = {
                s: score for s, score in self.vocab_scores.items()
                if s not in to_remove
            }

        self._build_id_mappings()

        if verbose:
            print(f"Entrenamiento completado: {len(self.vocab_scores)} subpalabras")

    def _build_id_mappings(self):
        for c in (chr(i) for i in range(32, 127)):
            if c != " " and c not in self.vocab_scores:
                self.vocab_scores[c] = -20.0

        sorted_tokens = sorted(self.vocab_scores.keys(),
                               key=lambda t: self.vocab_scores[t], reverse=True)
        self.token_to_id = {token: i for i, token in enumerate(sorted_tokens)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def encode(self, text):
        marked = "▁" + text.replace(" ", "▁")
        segments = viterbi_segment(marked, self.vocab_scores)
        ids = []
        for seg in segments:
            if seg in self.token_to_id:
                ids.append(self.token_to_id[seg])
            else:
                for ch in seg:
                    if ch in self.token_to_id:
                        ids.append(self.token_to_id[ch])
        return ids

    def encode_as_pieces(self, text):
        marked = "▁" + text.replace(" ", "▁")
        return viterbi_segment(marked, self.vocab_scores)

    def decode(self, ids):
        pieces = [self.id_to_token[i] for i in ids]
        text = "".join(pieces).replace("▁", " ")
        if text.startswith(" "):
            text = text[1:]
        return text

    @property
    def vocab_size_actual(self):
        return len(self.vocab_scores)

    def show_vocab(self, top_n=30):
        sorted_v = sorted(self.vocab_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {top_n} subpalabras (de {len(sorted_v)} total):")
        for token, score in sorted_v[:top_n]:
            print(f"  {token:20s}  log_prob={score:.4f}  prob={math.exp(score):.6f}")


if __name__ == "__main__":
    corpus = """
    La recuperación de información es el proceso de buscar y encontrar
    documentos relevantes en una colección. Los modelos clásicos como TF-IDF
    y BM25 representan documentos y consultas como vectores. Los modelos de
    lenguaje transformaron la recuperación de información: en lugar de
    representar documentos con frecuencias, los transformers aprenden
    embeddings densos que capturan el significado semántico.
    """ * 5

    tok = UnigramTokenizer()
    tok.train(corpus, vocab_size=200, verbose=False)

    print(f"Vocabulario aprendido: {tok.vocab_size_actual} subpalabras")
    print()
    tok.show_vocab(top_n=15)
    print()

    for frase in ["recuperación de información", "embeddings densos", "Mercadona"]:
        pieces = tok.encode_as_pieces(frase)
        ids = tok.encode(frase)
        print(f"'{frase}'")
        print(f"  -> {len(ids)} tokens: {pieces}")
        print()
