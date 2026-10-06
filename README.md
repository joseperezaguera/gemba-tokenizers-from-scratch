# gemba-tokenizers-from-scratch

> **Los tres algoritmos de tokenización subword que sostienen los LLMs modernos, implementados desde cero en Python sin librerías externas.** Material complementario del artículo de [Gemba](https://www.gemba.es/) [«¿Qué es un token?»](https://www.gemba.es/p/que-es-un-token) (18 de mayo de 2026).

Este repositorio no es código de producción. Es código **para entender**. Cada uno de los tres algoritmos tiene su propio fichero, comentado paso a paso, con un `__main__` que se puede ejecutar para verlo en acción. Cero dependencias.

## Los tres algoritmos

| Algoritmo | Quién lo usa | Estrategia | Fichero |
|-----------|--------------|------------|---------|
| **BPE** (Byte Pair Encoding) | GPT-4, GPT-4o, Claude, Mistral, Qwen, Llama 3 | Bottom-up: fusiona el par más frecuente | [`src/bpe.py`](src/bpe.py) |
| **Unigram + Viterbi** | T5, mBART, ALBERT, XLNet (SentencePiece) | Top-down: poda iterativa con EM + Viterbi para inferencia | [`src/unigram_viterbi.py`](src/unigram_viterbi.py) |
| **WordPiece** | BERT, DistilBERT, ELECTRA, MobileBERT | Bottom-up: fusiona el par con mejor `freq(AB) / (freq(A)·freq(B))` | [`src/wordpiece.py`](src/wordpiece.py) |

## Empezar en 30 segundos

```bash
git clone https://github.com/joseperezaguera/gemba-tokenizers-from-scratch.git
cd gemba-tokenizers-from-scratch
python3 src/bpe.py
python3 src/unigram_viterbi.py
python3 src/wordpiece.py
```

Cada uno entrena con un corpus pequeño embebido y tokeniza varias frases de ejemplo.

## Comparar los tres con el mismo texto

```bash
python3 src/compare.py
python3 src/compare.py "tu texto aquí"
python3 src/compare.py --file un_fichero.txt --train tu_corpus.txt
```

Salida ejemplo:

```
Texto: "Mercadona"

Algoritmo               #Tokens   Segmentación
------------------------------------------------------------------------
BPE                           6   M | er | ca | do | n | a
Unigram+Viterbi              10   ▁ | M | e | r | c | a | d | o | n | a
WordPiece                     8   M | ##e | ##r | ##c | ##a | ##do | ##n | ##a
```

## Estructura

```
gemba-tokenizers-from-scratch/
├── src/
│   ├── bpe.py                  # BPE desde cero
│   ├── unigram_viterbi.py      # Unigram + Viterbi desde cero
│   ├── wordpiece.py            # WordPiece desde cero
│   └── compare.py              # Tokeniza el mismo texto con los tres
├── examples/
│   ├── 01_bpe_paso_a_paso.md   # El ejemplo "cuenta de la vieja" del artículo
│   ├── 02_unigram_viterbi.md   # Cómo funciona Viterbi paso a paso
│   ├── 03_wordpiece_vs_bpe.md  # Por qué BPE y WordPiece eligen pares distintos
│   └── corpus_demo.txt         # Corpus pequeño para experimentar
├── claude-playbook/            # Skill de Claude Code para experimentar conversacionalmente
└── tests/
    └── test_roundtrip.py       # encode → decode == original
```

## Playbook para Claude Code

El directorio [`claude-playbook/`](claude-playbook/) contiene una skill de Claude Code que carga el repo y permite hacer preguntas tipo:

- *«Tokeniza esta frase con los tres algoritmos y compara»*
- *«¿Por qué BPE elige fusionar `e+r` en lugar de `o+w` en este corpus?»*
- *«Entrena un BPE con vocab_size=600 sobre este texto y muéstrame los primeros 20 merges»*

Para usarlo en Claude Code:

```bash
# Desde la raíz del repo
claude --plugin claude-playbook
```

O simplemente abre Claude Code en este directorio y pregunta directamente.

## ⚠️ Esto no es código de producción

Estas implementaciones están escritas para que la lógica del algoritmo sea visible y fácil de seguir, no para ser rápidas ni para ser equivalentes a las librerías de referencia. Para tokenizar en producción usa [`tiktoken`](https://github.com/openai/tiktoken) (OpenAI), [`transformers`](https://github.com/huggingface/transformers) de HuggingFace o [`sentencepiece`](https://github.com/google/sentencepiece).

Si lo que quieres es **calcular el coste real** de un texto en varios modelos comerciales, mira el repo hermano: [`gemba-token-cost-calculator`](https://github.com/joseperezaguera/gemba-token-cost-calculator). Uno para entender, el otro para decidir.

## Tests

```bash
python3 -m unittest tests/test_roundtrip.py -v
```

## Licencia

MIT. Úsalo, modifícalo, enséñalo. Si te resulta útil para enseñar a alguien, házmelo saber por [LinkedIn](https://www.linkedin.com/in/joseaguera/).

---

*José Ramón Pérez Agüera — Gemba · gemba.es*
