"""
Tests de roundtrip: encode(decode) == identidad para los tres tokenizadores.

Estos no son tests de equivalencia con tokenizadores reales — son tests de
consistencia interna. Si encodificas un texto y luego lo decodificas con el
mismo tokenizador, deberías obtener el texto original.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bpe import BPETokenizer
from unigram_viterbi import UnigramTokenizer
from wordpiece import WordPieceTokenizer


SAMPLE_CORPUS = """
La recuperación de información es el proceso de buscar y encontrar documentos
relevantes en una colección. Los modelos de lenguaje aprenden representaciones
densas. La tokenización es la base de todo.
""" * 5

ROUNDTRIP_PHRASES = [
    "recuperación de información",
    "embeddings densos",
    "modelos de lenguaje",
    "documentos relevantes",
]


class TestBPERoundtrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = BPETokenizer()
        cls.tok.train(SAMPLE_CORPUS, vocab_size=400)

    def test_roundtrip_phrases(self):
        for phrase in ROUNDTRIP_PHRASES:
            with self.subTest(phrase=phrase):
                ids = self.tok.encode(phrase)
                decoded = self.tok.decode(ids)
                self.assertEqual(decoded, phrase,
                                 f"Roundtrip falló para: {phrase!r}")

    def test_byte_fallback(self):
        # Texto que el corpus no vio: debe seguir codificándose vía bytes
        text = "ñandú ölvido 🛒"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, text)


class TestUnigramSanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = UnigramTokenizer()
        cls.tok.train(SAMPLE_CORPUS, vocab_size=150)

    def test_encode_non_empty(self):
        for phrase in ROUNDTRIP_PHRASES:
            with self.subTest(phrase=phrase):
                ids = self.tok.encode(phrase)
                self.assertGreater(len(ids), 0,
                                   f"Encoding vacío para: {phrase!r}")

    def test_decode_recovers_text(self):
        # Unigram puede tener pérdidas mínimas en espacios; aceptamos
        # que el texto reconstruido contenga las mismas palabras.
        for phrase in ROUNDTRIP_PHRASES:
            with self.subTest(phrase=phrase):
                ids = self.tok.encode(phrase)
                decoded = self.tok.decode(ids)
                # Comparamos por palabras tras normalizar espacios
                self.assertEqual(decoded.split(), phrase.split())


class TestWordPieceSanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = WordPieceTokenizer()
        cls.tok.train(SAMPLE_CORPUS, vocab_size=300)

    def test_encode_non_empty(self):
        for phrase in ROUNDTRIP_PHRASES:
            with self.subTest(phrase=phrase):
                ids = self.tok.encode(phrase)
                self.assertGreater(len(ids), 0)

    def test_decode_recovers_words(self):
        for phrase in ROUNDTRIP_PHRASES:
            with self.subTest(phrase=phrase):
                ids = self.tok.encode(phrase)
                decoded = self.tok.decode(ids)
                # WordPiece elimina y reconstruye espacios entre palabras
                self.assertEqual(decoded.split(), phrase.split())

    def test_unknown_character_falls_back_to_unk(self):
        # Carácter que no está en vocab inicial ni alcanzable
        ids = self.tok.encode("Ω")
        decoded = self.tok.decode(ids)
        self.assertIn("[UNK]", decoded)


if __name__ == "__main__":
    unittest.main()
