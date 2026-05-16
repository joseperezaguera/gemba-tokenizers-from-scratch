---
name: tokenizers-from-scratch
description: >
  Esta skill se activa cuando el usuario quiere experimentar con los tres
  tokenizadores subword implementados desde cero en este repo: BPE,
  Unigram+Viterbi y WordPiece. Útil para entender cómo se segmenta un texto,
  comparar algoritmos, ver el efecto del tamaño de vocabulario, o explicar
  por qué un tokenizador concreto toma determinada decisión.

  Triggers: "tokeniza esta frase", "compara los tres algoritmos", "cuántos
  tokens es esto en BPE", "explícame por qué WordPiece eligió este merge",
  "entrena un tokenizador con este corpus", "muéstrame los merges paso a paso".
version: 0.1.0
---

# Skill: Tokenizers from Scratch

## Cuándo usar

Cuando el usuario quiera experimentar con cualquiera de los tres tokenizadores subword del repo: **BPE**, **Unigram + Viterbi** o **WordPiece**.

Casos típicos:

- *"Tokeniza esta frase con los tres algoritmos."*
- *"Entrena BPE con este corpus y muéstrame los primeros 10 merges."*
- *"¿Por qué WordPiece fusionaría `i + d` antes que `e + r`?"*
- *"Compara la segmentación de 'embeddings densos' en BPE vs Unigram."*
- *"Si entreno con un corpus 100% inglés, ¿cuántos tokens es 'Mercadona'?"*

## Mapeo a ficheros

| Acción del usuario | Fichero a usar | Función relevante |
|--------------------|----------------|-------------------|
| Tokenizar con BPE | `src/bpe.py` | `BPETokenizer.encode()` |
| Tokenizar con Unigram | `src/unigram_viterbi.py` | `UnigramTokenizer.encode_as_pieces()` |
| Tokenizar con WordPiece | `src/wordpiece.py` | `WordPieceTokenizer.encode_as_pieces()` |
| Comparar los tres | `src/compare.py` | `compare.main()` o invocación directa |
| Ver merges paso a paso | `src/bpe.py` con `verbose=True` | `BPETokenizer.train(..., verbose=True)` |

## Cómo proceder

1. **Identifica el algoritmo** que el usuario quiere usar. Si no lo especifica, pregunta o aplica los tres.
2. **Decide qué corpus usar para entrenamiento**:
   - Si el usuario no aporta corpus, usa `DEFAULT_CORPUS` de `src/compare.py`.
   - Si lo aporta, escríbelo a un fichero temporal y pásalo con `--train`.
3. **Ejecuta el script apropiado** vía Bash y muestra los resultados.
4. **Interpreta el resultado** para el usuario: cuántos tokens salen, qué piezas, por qué.
5. **Ofrece variaciones**: cambiar el tamaño de vocabulario, comparar con otro algoritmo, ver los merges.

## Comportamiento por defecto

- **Vocab size por defecto:** 400 para BPE y WordPiece, 200 para Unigram. Si el corpus es muy pequeño puede bajar automáticamente.
- **Mostrar siempre el número de tokens** además de las piezas, para que el usuario pueda compararlas con su factura mental.
- **Cuando aparezca un emoji o un nombre propio raro**, comenta que el comportamiento esperado es muchos tokens — eso es exactamente lo que el artículo de Gemba explica en §5.

## Cosas que NO hace este repo

- No usa `tiktoken`, `transformers` ni ningún tokenizador real. Los resultados son **aproximaciones pedagógicas**, no equivalentes a los de OpenAI/Anthropic/HuggingFace.
- Si el usuario pregunta "cuánto cuesta tokenizar X en producción", redirige al repo hermano [`gemba-token-cost-calculator`](https://github.com/josemerca/gemba-token-cost-calculator).

## Comandos disponibles

- `/tokenizar` — Tokeniza un texto con uno o los tres algoritmos. Ver `commands/tokenizar.md`.

---

*Material complementario del artículo de Gemba [«¿Qué es un token?»](https://www.gemba.es/p/que-es-un-token) — 18 de mayo de 2026.*
