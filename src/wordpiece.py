"""
WordPiece — desde cero
======================
Tokenizador subword usado por BERT y derivados (DistilBERT, ELECTRA,
MobileBERT, etc.). Sigue siendo dominante en modelos de búsqueda semántica
y clasificación (encoders).

Diferencia con BPE:
- BPE fusiona el par MÁS FRECUENTE.
- WordPiece fusiona el par que MÁS MEJORA la probabilidad del corpus:

      score(A, B) = freq(AB) / (freq(A) × freq(B))

  El numerador premia coapariciones frecuentes. El denominador penaliza
  cuando A o B aparecen también muy frecuentemente por separado. El cociente
  selecciona pares cuyos componentes "casi siempre van juntos". En espíritu
  es una versión simplificada del pointwise mutual information (PMI) clásico
  de lingüística computacional.

Convención: los tokens que NO son inicio de palabra llevan prefijo "##"
(como en el tokenizer real de BERT). Eso permite distinguir "ana" como
inicio de palabra (`ana`) frente a sufijo dentro de "manzana" (`##ana`).

Cero dependencias. Solo Python estándar.

Referencia histórica: Schuster & Nakajima publicaron WordPiece en 2012
("Japanese and Korean Voice Search", Google) para reconocimiento de voz.
Se hizo famoso en 2018 con BERT.
"""

from collections import Counter, defaultdict


def _words_to_initial_pieces(word_freq):
    """Tokeniza cada palabra única en sus caracteres iniciales.

    El primer carácter de cada palabra se queda tal cual. Los demás llevan
    prefijo "##" (sufijo de palabra). Esto sigue la convención de BERT y
    permite distinguir "in" inicio-de-palabra de "##in" sufijo.

    Returns:
        dict {palabra: [piezas]}
    """
    splits = {}
    for word in word_freq:
        if not word:
            continue
        chars = [word[0]] + ["##" + c for c in word[1:]]
        splits[word] = chars
    return splits


def _count_pieces(splits, word_freq):
    """Cuenta apariciones de cada pieza ponderadas por la frecuencia de su palabra."""
    counts = Counter()
    for word, pieces in splits.items():
        freq = word_freq[word]
        for p in pieces:
            counts[p] += freq
    return counts


def _count_pairs(splits, word_freq):
    """Cuenta apariciones de cada par adyacente ponderadas por word_freq."""
    pair_counts = Counter()
    for word, pieces in splits.items():
        if len(pieces) < 2:
            continue
        freq = word_freq[word]
        for a, b in zip(pieces, pieces[1:]):
            pair_counts[(a, b)] += freq
    return pair_counts


def _merge_pair(splits, pair):
    """Fusiona el par (a, b) en todas las palabras donde aparezca.

    Cuando fusionamos a + b, el token resultante es:
      - `a + b[2:]` si b empieza por "##"   (ej: "ca" + "##sa" -> "casa")
      - `a + b` en otro caso (raramente sucede dado el preprocesamiento)
    """
    a, b = pair
    new_token = a + b[2:] if b.startswith("##") else a + b
    new_splits = {}
    for word, pieces in splits.items():
        if len(pieces) < 2:
            new_splits[word] = pieces
            continue
        merged = []
        i = 0
        while i < len(pieces):
            if i < len(pieces) - 1 and pieces[i] == a and pieces[i + 1] == b:
                merged.append(new_token)
                i += 2
            else:
                merged.append(pieces[i])
                i += 1
        new_splits[word] = merged
    return new_splits


class WordPieceTokenizer:
    """Tokenizador WordPiece mínimo pero completo.

    Flujo:
      tokenizer = WordPieceTokenizer()
      tokenizer.train(corpus, vocab_size=400)
      ids = tokenizer.encode("texto a tokenizar")
      texto = tokenizer.decode(ids)
    """

    def __init__(self):
        self.vocab = []           # lista ordenada de tokens
        self.token_to_id = {}     # {token: id}
        self.unk_token = "[UNK]"

    def train(self, text, vocab_size=400, verbose=False):
        words = text.replace("\n", " ").split()
        word_freq = Counter(words)

        splits = _words_to_initial_pieces(word_freq)

        # Vocabulario inicial: todos los caracteres únicos. Para garantizar
        # que cualquier texto sea tokenizable, cada carácter del corpus debe
        # estar disponible tanto como inicio-de-palabra (sin "##") como
        # interior-de-palabra (con "##"). Esto sigue la convención de BERT.
        initial_pieces = set()
        for pieces in splits.values():
            initial_pieces.update(pieces)
        for piece in list(initial_pieces):
            if piece.startswith("##"):
                initial_pieces.add(piece[2:])
            else:
                initial_pieces.add("##" + piece)
        vocab = sorted(initial_pieces) + [self.unk_token]

        if verbose:
            print(f"Vocabulario inicial (caracteres): {len(vocab)} piezas")
            print(f"Objetivo: {vocab_size} piezas. "
                  f"Merges a realizar: {vocab_size - len(vocab)}")
            print("=" * 60)

        while len(vocab) < vocab_size:
            piece_counts = _count_pieces(splits, word_freq)
            pair_counts = _count_pairs(splits, word_freq)
            if not pair_counts:
                break

            # Score WordPiece para cada par
            best_pair = None
            best_score = 0.0
            for pair, c_ab in pair_counts.items():
                a, b = pair
                denom = piece_counts[a] * piece_counts[b]
                if denom == 0:
                    continue
                score = c_ab / denom
                if score > best_score:
                    best_score = score
                    best_pair = pair

            if best_pair is None:
                break

            splits = _merge_pair(splits, best_pair)
            a, b = best_pair
            new_token = a + b[2:] if b.startswith("##") else a + b
            if new_token not in vocab:
                vocab.append(new_token)

            if verbose:
                print(f"Merge {len(vocab) - len(initial_pieces):3d}: "
                      f"{a:>10} + {b:<10} -> {new_token:<15} "
                      f"score={best_score:.4f}  "
                      f"(c_ab={pair_counts[best_pair]}, "
                      f"c_a={piece_counts[a]}, c_b={piece_counts[b]})")

        self.vocab = vocab
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}

        if verbose:
            print("=" * 60)
            print(f"Vocabulario final: {len(self.vocab)} piezas")

    def _tokenize_word(self, word):
        """Greedy longest-match para una palabra: misma estrategia que BERT.

        Empezando por el principio, busca la subcadena más larga que esté
        en el vocabulario. Las posiciones interiores buscan con prefijo "##".
        """
        if not word:
            return []
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            cur_substr = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.token_to_id:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                return [self.unk_token]
            tokens.append(cur_substr)
            start = end
        return tokens

    def encode_as_pieces(self, text):
        """Texto -> lista de piezas (strings)."""
        pieces = []
        for word in text.split():
            pieces.extend(self._tokenize_word(word))
        return pieces

    def encode(self, text):
        """Texto -> lista de token IDs."""
        ids = []
        for piece in self.encode_as_pieces(text):
            ids.append(self.token_to_id.get(piece, self.token_to_id[self.unk_token]))
        return ids

    def decode(self, ids):
        """Lista de token IDs -> texto.

        Las piezas con prefijo "##" se concatenan a la pieza anterior sin espacio.
        Las piezas sin prefijo empiezan una palabra nueva.
        """
        pieces = [self.vocab[i] for i in ids]
        out = []
        for p in pieces:
            if p.startswith("##"):
                if out:
                    out[-1] = out[-1] + p[2:]
                else:
                    out.append(p[2:])
            else:
                out.append(p)
        return " ".join(out)

    @property
    def vocab_size(self):
        return len(self.vocab)


if __name__ == "__main__":
    corpus = """
    La recuperación de información es el proceso de buscar y encontrar
    documentos relevantes en una colección. Los modelos clásicos como TF-IDF
    y BM25 representan documentos y consultas como vectores. Los modelos de
    lenguaje transformaron la recuperación de información: en lugar de
    representar documentos con frecuencias, los transformers aprenden
    embeddings densos que capturan el significado semántico.
    """ * 5

    tok = WordPieceTokenizer()
    tok.train(corpus, vocab_size=300, verbose=False)

    print(f"Vocabulario aprendido: {tok.vocab_size} piezas")
    print()

    for frase in ["recuperación de información", "embeddings densos", "Mercadona"]:
        pieces = tok.encode_as_pieces(frase)
        ids = tok.encode(frase)
        print(f"'{frase}'")
        print(f"  -> {len(ids)} tokens: {pieces}")
        print()
