---
description: Tokeniza un texto con uno o los tres algoritmos del repo y muestra una comparación side-by-side
allowed-tools: Bash, Read
argument-hint: <texto a tokenizar> [opciones]
---

# /tokenizar

Tokeniza el texto dado usando uno o varios de los tres tokenizadores del repo (BPE, Unigram+Viterbi, WordPiece) y muestra una comparación visual.

## Argumentos

- `<texto>` — el texto a tokenizar (obligatorio)
- `--algo bpe|unigram|wordpiece|all` — qué algoritmo usar (default: `all`)
- `--vocab-size N` — tamaño del vocabulario (default: 400)
- `--corpus <path>` — fichero con corpus de entrenamiento (default: corpus embebido)

## Comportamiento

1. Ejecuta `python3 src/compare.py "$1"` desde la raíz del repo.
2. Si se pasa `--corpus`, añade `--train <path>` al comando.
3. Si se pasa `--algo` distinto de `all`, ejecuta solo el script correspondiente (`src/bpe.py`, `src/unigram_viterbi.py` o `src/wordpiece.py`).
4. Muestra al usuario:
   - El número de tokens por algoritmo
   - La segmentación de cada uno
   - Una breve interpretación: cuál genera más/menos tokens y por qué.

## Ejemplos

```
/tokenizar recuperación de información
```

```
/tokenizar "🛒 carrito de la compra" --vocab-size 600
```

```
/tokenizar "tu texto largo aquí" --corpus mi_corpus.txt
```

## Cosas a recordar

- Si el texto contiene caracteres muy raros (emojis, idiomas no presentes en el corpus de entrenamiento), explica al usuario que muchos tokens son normales — eso es el punto del repo.
- Estos resultados son **pedagógicos**, no producción. Si el usuario quiere coste real, redirígelo a `gemba-token-cost-calculator`.
