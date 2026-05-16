# 02 — Unigram + Viterbi: el algoritmo que trabaja al revés

Mientras BPE construye el vocabulario fusionando piezas de abajo arriba, **Unigram** lo hace al revés: empieza con un vocabulario gigante de candidatos y va podando los menos útiles.

## Las dos piezas conceptuales

### Piezas con probabilidad

Cada candidato a subpalabra tiene una **probabilidad** estimada a partir del corpus. La probabilidad de una palabra entera —por ejemplo, `lowest`— se calcula multiplicando las probabilidades de los trozos en los que se segmenta.

| Segmentación posible de `lowest` | Probabilidad | Cálculo |
|----------------------------------|--------------|---------|
| `low` + `est` | **0,060** | 0,30 × 0,20 |
| `lowest` (una sola pieza) | 0,050 | 0,05 |
| `lo` + `west` | 0,016 | 0,10 × 0,16 |
| `l` + `o` + `w` + `e` + `s` + `t` | 0,00001 | producto de seis P() |

### Viterbi para encontrar la mejor segmentación

Viterbi es un algoritmo de programación dinámica. Recorre todas las segmentaciones posibles eficientemente y se queda con la de máxima probabilidad. Sin Viterbi tendrías que probar todas a mano: con palabras largas, eso explota combinatoriamente.

El mismo algoritmo lo usaban los **POS taggers HMM** de los años 80-90 para etiquetar gramaticalmente palabras. Andrew Viterbi lo publicó originalmente en **1967** para decodificar señales en telecomunicaciones.

## Verlo funcionando

```python
from src.unigram_viterbi import UnigramTokenizer

corpus = """
La recuperación de información es buscar documentos relevantes.
Los modelos de lenguaje aprenden embeddings densos.
""" * 20

tok = UnigramTokenizer()
tok.train(corpus, vocab_size=150, verbose=True)

tok.show_vocab(top_n=10)

print()
print("Encode:", tok.encode_as_pieces("recuperación de información"))
print("IDs:   ", tok.encode("recuperación de información"))
```

En la salida verás que el vocabulario empieza con cientos de candidatos y se va podando iteración a iteración hasta los 150 que pediste. El log de cada iteración te muestra cómo la log-likelihood total del corpus va aumentando conforme el vocabulario se afina.

## ¿Cuándo usar Unigram en vez de BPE?

Unigram suele rendir mejor para:

- **Idiomas con morfología compleja**: japonés, coreano, finés, turco.
- **Modelos multilingües**: porque la estimación probabilística maneja mejor los desequilibrios entre lenguas.
- **Casos con segmentación ambigua**: donde una palabra podría partirse en varios sitios y necesitas el "mejor" objetivamente.

Por eso es la elección habitual en SentencePiece para T5, mBART, ALBERT y muchos modelos asiáticos.

> Continúa en [03_wordpiece_vs_bpe.md](03_wordpiece_vs_bpe.md) para ver por qué WordPiece (BERT) puede elegir un par distinto al que elegiría BPE con el mismo corpus.
