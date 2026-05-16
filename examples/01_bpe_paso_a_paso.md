# 01 — BPE paso a paso, con la cuenta de la vieja

Este ejemplo replica el caso del artículo. Lo puedes ejecutar y verlo en directo.

## El corpus

Un corpus minúsculo, perfecto para entender qué pasa:

| Palabra | Frecuencia |
|---------|------------|
| `low`   | 5 |
| `lower` | 2 |
| `newer` | 6 |
| `wider` | 3 |

Ese corpus está en [`corpus_demo.txt`](corpus_demo.txt) repetido tantas veces como hace falta para que el BPE tenga material suficiente.

## Lo que vas a ver

Si ejecutas el siguiente script:

```bash
cd /ruta/al/repo
python3 -c "
from src.bpe import BPETokenizer

corpus = open('examples/corpus_demo.txt').read()
tok = BPETokenizer()
tok.train(corpus, vocab_size=260, verbose=True)
"
```

el algoritmo te mostrará paso a paso cada fusión. En las primeras iteraciones verás algo parecido a esto:

```
Merge   1: (101, 114) -> 256  freq=  11   secuencia: ...
Merge   2: (256, ...) -> 257  freq=   8   secuencia: ...
```

`(101, 114)` son los códigos ASCII de `e` y `r`. La primera fusión gana `e + r` con 11 apariciones — exactamente el cálculo que viene en el artículo:

> El par más frecuente es `e + r` con 11 apariciones, porque aparece en `lower` (×2), `newer` (×6) y `wider` (×3). Lo fusionamos: nace un nuevo token, `er`, y todo el corpus se reescribe con él.

## Qué pasa después

En la segunda iteración, el par ganador es `w + er`. Por qué:

| Palabra | Tras el primer merge | Apariciones de `w + er` |
|---------|----------------------|-------------------------|
| `low`   | `l, o, w`            | 0 |
| `lower` | `l, o, w, er`        | 2 |
| `newer` | `n, e, w, er`        | 6 |
| `wider` | `w, i, d, er`        | 0 |

Total: 2 + 6 = **8**. Sigue siendo el par más frecuente.

## Después de 5 fusiones

```python
from src.bpe import BPETokenizer

corpus = open('examples/corpus_demo.txt').read() * 10
tok = BPETokenizer()
tok.train(corpus, vocab_size=261, verbose=True)

print("\nVocabulario aprendido (solo merges):")
for pair, new_id in tok.merges.items():
    print(f"  {new_id}: {tok.vocab[new_id].decode('utf-8', errors='replace')!r}")

print("\nTokenizar 'newer':", tok.encode('newer'))
print("Pedacitos:", [tok.token_str(i) for i in tok.encode('newer')])
```

Verás que `newer` ya no son 5 tokens sino solo 2: `n` + `ewer` (o `ne` + `wer`, depende del orden exacto de merges). Eso es BPE comprimiendo el corpus.

## Lo importante

Ningún paso del algoritmo conoce el español, el inglés ni la lingüística. Todo lo que está pasando ahí son recuentos de coapariciones sobre los documentos del corpus. La gramática es un epifenómeno: si el corpus la refleja, el tokenizador la captura.

> Continúa en [02_unigram_viterbi.md](02_unigram_viterbi.md) para ver cómo el mismo problema se ataca desde el otro extremo.
