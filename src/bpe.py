"""
BPE (Byte Pair Encoding) — desde cero
=====================================
Implementación pedagógica del algoritmo que usan GPT, Claude, Llama 3, Mistral
y la mayoría de los LLMs generativos modernos.

Funciona en tres pasos:
  1. Empieza con un vocabulario de los 256 bytes posibles.
  2. Cuenta la frecuencia de pares adyacentes en el corpus.
  3. Fusiona el par más frecuente como un nuevo token. Repite.

Cero dependencias. Solo Python estándar.
"""


def get_stats(ids):
    """Cuenta la frecuencia de cada par adyacente de tokens.

    Es el corazón de BPE: encontrar el par más frecuente para fusionarlo.

    Args:
        ids: lista de enteros (tokens)
    Returns:
        dict {(token_a, token_b): frecuencia}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Reemplaza todas las ocurrencias de 'pair' en 'ids' con 'new_id'."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    """Tokenizador BPE mínimo pero completo.

    Flujo:
      tokenizer = BPETokenizer()
      tokenizer.train(corpus_text, vocab_size=500)
      ids = tokenizer.encode("texto a tokenizar")
      texto = tokenizer.decode(ids)
    """

    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def train(self, text, vocab_size, verbose=False):
        """Entrena BPE sobre un texto.

        Args:
            text: string de entrenamiento
            vocab_size: tamaño deseado del vocabulario (>= 256)
            verbose: si True, imprime cada fusión
        """
        assert vocab_size >= 256, "El vocabulario mínimo es 256 (todos los bytes)"
        num_merges = vocab_size - 256

        tokens = list(text.encode("utf-8"))

        if verbose:
            print(f"Texto original: {len(text)} caracteres")
            print(f"Tokens iniciales (bytes): {len(tokens)}")
            print(f"Merges a realizar: {num_merges}")
            print("=" * 60)

        self.merges = {}
        for i in range(num_merges):
            stats = get_stats(tokens)
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            new_id = 256 + i
            tokens = merge(tokens, best_pair, new_id)
            self.merges[best_pair] = new_id

            if verbose:
                a, b = best_pair
                print(
                    f"Merge {i+1:3d}: ({a:4d}, {b:4d}) -> {new_id:4d}  "
                    f"freq={stats[best_pair]:4d}  "
                    f"secuencia: {len(tokens)} tokens"
                )

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

        if verbose:
            ratio = len(text.encode("utf-8")) / len(tokens) if tokens else 1.0
            print("=" * 60)
            print(f"Vocabulario final: {len(self.vocab)} tokens")
            print(f"Compresión: {ratio:.2f}x")

    def encode(self, text):
        """Convierte texto a una secuencia de token IDs."""
        tokens = list(text.encode("utf-8"))
        for pair, new_id in self.merges.items():
            tokens = merge(tokens, pair, new_id)
        return tokens

    def decode(self, ids):
        """Convierte una secuencia de token IDs a texto."""
        tokens_bytes = b"".join(self.vocab[idx] for idx in ids)
        return tokens_bytes.decode("utf-8", errors="replace")

    def token_str(self, token_id):
        """Representación legible de un token (para depuración)."""
        b = self.vocab.get(token_id, bytes([token_id]) if token_id < 256 else b"?")
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return f"<{token_id}>"


if __name__ == "__main__":
    corpus = """
    La recuperación de información es el proceso de buscar y encontrar
    documentos relevantes en una colección. Los modelos clásicos como TF-IDF
    y BM25 representan documentos y consultas como vectores. Los modelos de
    lenguaje transformaron la recuperación de información: en lugar de
    representar documentos con frecuencias, los transformers aprenden
    embeddings densos que capturan el significado semántico.
    """ * 5

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=400, verbose=False)

    print(f"Vocabulario aprendido: {len(tok.vocab)} tokens")
    print()

    for frase in ["recuperación de información", "embeddings densos", "Mercadona"]:
        ids = tok.encode(frase)
        piezas = [tok.token_str(i) for i in ids]
        print(f"'{frase}'")
        print(f"  -> {len(ids)} tokens: {piezas}")
        print()
