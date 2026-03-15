"""Chemistry-aware BPE tokenizer for SMILES and IUPAC strings.

Uses HuggingFace `tokenizers` library with domain-specific pre-tokenization
regexes. Separate tokenizer instances for SMILES and IUPAC vocabularies.
"""

import json
import os
import re
from typing import ClassVar, Dict, List

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


# ── Pre-tokenization Regexes ─────────────────────────────────────────────────

SMILES_REGEX = r"(\[[^\]]+\]|Br|Cl|Si|Se|@@|@|[=#$:\/\\]|[A-Z][a-z]?|[0-9]|[().])"

IUPAC_REGEX = (
    r"(cyclo|methyl|ethyl|propyl|butyl|pentyl|hexyl|phenyl|amino|hydroxy|oxo|"
    r"chloro|bromo|fluoro|nitro|oxy|thio|one|al|ol|ane|ene|yne|oic|amide|"
    r"amine|acid|di|tri|tetra|penta|hexa|mono|\(|\)|\[|\]|,|-|[0-9]+|[a-zA-Z])"
)

SPECIAL_TOKENS: List[str] = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


class ChemBPETokenizer:
    """Chemistry-aware BPE tokenizer with separate SMILES/IUPAC vocabularies.

    Special tokens:
        <pad> (0), <sos> (1), <eos> (2), <unk> (3)
    """

    _REGEX_MAP: ClassVar[Dict[str, str]] = {
        "smiles": SMILES_REGEX,
        "iupac": IUPAC_REGEX,
    }

    def __init__(self, name: str) -> None:
        """Initialize tokenizer.

        Args:
            name: Either "smiles" or "iupac".
        """
        if name not in self._REGEX_MAP:
            raise ValueError(f"name must be 'smiles' or 'iupac', got '{name}'")
        self.name = name
        self._regex = self._REGEX_MAP[name]
        self._tokenizer: Tokenizer | None = None
        self._vocab_size_val: int = 0

    def _pre_tokenize(self, text: str) -> List[str]:
        """Split text using the domain-specific regex."""
        return re.findall(self._regex, text)

    def train(self, texts: List[str], vocab_size: int, min_freq: int) -> None:
        """Train BPE on a corpus of texts.

        Args:
            texts: List of raw strings (SMILES or IUPAC names).
            vocab_size: Target vocabulary size including special tokens.
            min_freq: Minimum frequency for a BPE merge.
        """
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Split(
            pattern=self._regex,
            behavior="isolated",
            invert=False,
        )

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)

        self._tokenizer = tokenizer
        self._vocab_size_val = tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token IDs.

        Args:
            text: Raw input string.

        Returns:
            List of integer token IDs.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded. Call train() or load() first.")
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to a string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string with special tokens removed.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded. Call train() or load() first.")
        # Filter out special token IDs before decoding
        filtered = [i for i in ids if i not in (PAD_ID, SOS_ID, EOS_ID)]
        return self._tokenizer.decode(filtered)

    def save(self, path: str) -> None:
        """Save tokenizer to a JSON file.

        Args:
            path: File path for saving (e.g., 'tokenizers/smiles.json').
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Save the HuggingFace tokenizer
        self._tokenizer.save(path)

        # Save metadata alongside
        meta_path = path.replace(".json", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"name": self.name, "vocab_size": self._vocab_size_val}, f)

    @classmethod
    def load(cls, path: str) -> "ChemBPETokenizer":
        """Load a tokenizer from a JSON file.

        Args:
            path: File path to load from.

        Returns:
            A ChemBPETokenizer instance ready for encoding/decoding.
        """
        meta_path = path.replace(".json", "_meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        instance = cls(name=meta["name"])
        instance._tokenizer = Tokenizer.from_file(path)
        instance._vocab_size_val = meta["vocab_size"]
        return instance

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self._vocab_size_val

    @property
    def pad_id(self) -> int:
        """Token ID for <pad>."""
        return PAD_ID

    @property
    def sos_id(self) -> int:
        """Token ID for <sos>."""
        return SOS_ID

    @property
    def eos_id(self) -> int:
        """Token ID for <eos>."""
        return EOS_ID

    @property
    def unk_id(self) -> int:
        """Token ID for <unk>."""
        return UNK_ID
