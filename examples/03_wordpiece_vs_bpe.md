# 03 — WordPiece vs BPE: el mismo corpus, otro ganador

BPE y WordPiece comparten estructura: ambos construyen el vocabulario de abajo arriba, fusionando un par por iteración. La diferencia está en **qué par eligen**.

| Algoritmo | Criterio de fusión |
|-----------|--------------------|
| **BPE** | El par **más frecuente** en el corpus |
| **WordPiece** | El par con mayor **`freq(AB) / (freq(A) × freq(B))`** |

El score de WordPiece premia los pares cuyos componentes **casi siempre van juntos**, aunque su frecuencia bruta no sea la más alta. Es, en espíritu, una versión simplificada del *pointwise mutual information* (PMI) de lingüística computacional.

## El mismo corpus, dos ganadores

Con el corpus de ejemplo (`low: 5`, `lower: 2`, `newer: 6`, `wider: 3`):

| Par | Frecuencia bruta | Score WordPiece | Lectura |
|-----|------------------|-----------------|---------|
| `e + r` | **11** | 11 / (17 × 11) ≈ **0,059** | Frecuente, pero `e` y `r` aparecen también muchísimo por separado |
| `i + d` | 3 | 3 / (3 × 3) ≈ **0,333** | Poco frecuente, pero cuando aparece `i` casi siempre aparece `d` justo después |

- **BPE** elegiría `e + r` (la frecuencia bruta manda).
- **WordPiece** elegiría `i + d` (la fusión más "informativa").

En la práctica, los vocabularios resultantes se parecen, pero WordPiece tiende a capturar mejor unidades con asociación fuerte —prefijos, sufijos, raíces poco frecuentes pero cohesionadas— y peor las coapariciones de piezas individualmente comunes.

## Verlo funcionando

```python
from src.bpe import BPETokenizer
from src.wordpiece import WordPieceTokenizer

corpus = open("examples/corpus_demo.txt").read() * 5

bpe = BPETokenizer()
bpe.train(corpus, vocab_size=270)

wp = WordPieceTokenizer()
wp.train(corpus, vocab_size=270, verbose=True)
```

El log verbose de WordPiece mostrará para cada merge tanto el score como las frecuencias individuales, así puedes ver con qué criterio se está eligiendo cada paso.

## Una nota sobre el prefijo `##`

Si miras el vocabulario de WordPiece verás piezas como `##er`, `##ing`, `##ción`. El prefijo `##` significa "esta pieza solo aparece como continuación de palabra, nunca como inicio". Es la convención del tokenizador real de BERT y nuestra implementación la respeta para que los ejemplos sean fácilmente comparables con cualquier tokenizador `bert-*` de HuggingFace.

## Conclusión

Los tres algoritmos están resolviendo el mismo problema —construir un vocabulario de subpalabras a partir de un corpus— con tres criterios distintos. Lo que importa entender es:

1. **Ninguno usa reglas lingüísticas.** Todo es estadística sobre el corpus.
2. **El criterio de fusión cambia qué vocabulario aprendes.** Y el vocabulario que aprendes cambia cuántos tokens consume cada texto en producción.
3. **Cuántos tokens consume cada texto cambia tu factura.** Por eso la elección del tokenizador no es un detalle técnico.

> Si quieres calcular cuánto te costaría un texto real con tokenizadores reales (`tiktoken`, HuggingFace, Anthropic), salta al repo hermano: [`gemba-token-cost-calculator`](https://github.com/joseperezaguera/gemba-token-cost-calculator).
