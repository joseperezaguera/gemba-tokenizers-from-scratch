"""
Compare — tokeniza el mismo texto con los tres algoritmos y muestra
side-by-side cómo segmenta cada uno.

Uso:
    python3 src/compare.py "texto a tokenizar"
    python3 src/compare.py --file examples/corpus_demo.txt --train examples/corpus_demo.txt

Si no pasas --train, los tres tokenizadores entrenan sobre un corpus pequeño
incluido en el script. Si quieres ver diferencias reales con tus datos, pasa
un fichero de entrenamiento grande con --train.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bpe import BPETokenizer
from unigram_viterbi import UnigramTokenizer
from wordpiece import WordPieceTokenizer


DEFAULT_CORPUS = """
La recuperación de información es el proceso de buscar y encontrar documentos
relevantes en una colección. Los modelos clásicos como TF-IDF y BM25 representan
documentos y consultas como vectores en un espacio de términos.

Los modelos de lenguaje transformaron la recuperación de información. En lugar
de representar documentos con frecuencias de términos, los transformers
aprenden representaciones densas que capturan el significado semántico. La
recuperación densa usa estos embeddings para encontrar documentos relevantes
sin necesidad de coincidencia exacta de términos.

La tokenización en modelos de lenguaje es fundamentalmente diferente a la
tokenización clásica en recuperación de información. Mientras que en IR
clásica se usa stemming y lematización con reglas lingüísticas, los LLMs
aprenden la tokenización de los datos usando algoritmos como BPE, Unigram con
Viterbi o WordPiece. Esto permite manejar cualquier idioma sin reglas
específicas.
""" * 4


def train_all(corpus, vocab_size=400):
    """Entrena los tres tokenizadores sobre el mismo corpus."""
    bpe = BPETokenizer()
    bpe.train(corpus, vocab_size=vocab_size)

    uni = UnigramTokenizer()
    uni.train(corpus, vocab_size=vocab_size // 2)  # vocab Unigram más chico

    wp = WordPieceTokenizer()
    wp.train(corpus, vocab_size=vocab_size)

    return bpe, uni, wp


def tokenize_all(text, bpe, uni, wp):
    """Tokeniza el mismo texto con los tres y devuelve dict con resultados."""
    return {
        "BPE": {
            "ids": bpe.encode(text),
            "pieces": [bpe.token_str(i) for i in bpe.encode(text)],
        },
        "Unigram+Viterbi": {
            "ids": uni.encode(text),
            "pieces": uni.encode_as_pieces(text),
        },
        "WordPiece": {
            "ids": wp.encode(text),
            "pieces": wp.encode_as_pieces(text),
        },
    }


def render_comparison(text, results):
    """Imprime una comparación bonita."""
    print()
    print(f"Texto: \"{text}\"")
    print()
    print(f"{'Algoritmo':<20} {'#Tokens':>10}   Segmentación")
    print("-" * 90)
    for algo, data in results.items():
        n = len(data["ids"])
        pieces = data["pieces"]
        display = " | ".join(repr(p)[1:-1] if isinstance(p, str) else str(p)
                             for p in pieces)
        if len(display) > 60:
            display = display[:57] + "..."
        print(f"{algo:<20} {n:>10}   {display}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compara cómo tokenizan BPE, Unigram+Viterbi y WordPiece "
                    "el mismo texto."
    )
    parser.add_argument(
        "text", nargs="?", default=None,
        help="Texto a tokenizar. Si no se pasa, se usan ejemplos predefinidos."
    )
    parser.add_argument(
        "--file", default=None,
        help="Fichero con texto a tokenizar (sustituye al argumento posicional)."
    )
    parser.add_argument(
        "--train", default=None,
        help="Fichero con corpus de entrenamiento. Si no se pasa, se usa "
             "un corpus pequeño embebido."
    )
    parser.add_argument(
        "--vocab-size", type=int, default=400,
        help="Tamaño del vocabulario objetivo (default: 400)"
    )

    args = parser.parse_args()

    if args.train:
        corpus = Path(args.train).read_text(encoding="utf-8")
        print(f"Entrenando con {Path(args.train).name} "
              f"({len(corpus)} caracteres, vocab_size={args.vocab_size})...")
    else:
        corpus = DEFAULT_CORPUS
        print(f"Entrenando con corpus embebido "
              f"({len(corpus)} caracteres, vocab_size={args.vocab_size})...")

    bpe, uni, wp = train_all(corpus, vocab_size=args.vocab_size)

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
        results = tokenize_all(text, bpe, uni, wp)
        render_comparison(text[:100] + ("..." if len(text) > 100 else ""), results)
    elif args.text:
        results = tokenize_all(args.text, bpe, uni, wp)
        render_comparison(args.text, results)
    else:
        # Ejemplos predefinidos
        examples = [
            "recuperación de información",
            "embeddings densos",
            "Mercadona",
            "tokenización en LLMs",
            "🛒 carrito de la compra",
        ]
        for text in examples:
            results = tokenize_all(text, bpe, uni, wp)
            render_comparison(text, results)


if __name__ == "__main__":
    main()
