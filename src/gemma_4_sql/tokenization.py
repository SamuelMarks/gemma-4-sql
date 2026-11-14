"""Tokenization module for processing Text-to-SQL datasets without control-token byte collisions."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, SupportsInt, overload, runtime_checkable

from gemma_4_sql.backends.lazy_loader import LazyLoader, ModuleNamespaceProtocol
from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.type_hints import HFTokenizerProtocol

logger = logging.getLogger(__name__)

DEFAULT_BYTE_VOCAB_SIZE: int = 259


@runtime_checkable
class AutoTokenizerClassProtocol(Protocol):
    """Protocol representing Hugging Face `AutoTokenizer` class with `from_pretrained`."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> HFTokenizerProtocol:
        """Instantiate a tokenizer from a pretrained model identifier or path.

        Args:
            pretrained_model_name_or_path: Model identifier or directory path.

        Returns:
            Loaded `HFTokenizerProtocol` instance.

        """
        ...


@runtime_checkable
class _TransformersModuleProtocol(Protocol):
    """Protocol for a loaded `transformers` module exposing `AutoTokenizer`."""

    @property
    def AutoTokenizer(self) -> AutoTokenizerClassProtocol | Callable[..., HFTokenizerProtocol] | None:
        """Provide the `AutoTokenizer` factory class.

        Returns:
            `AutoTokenizer` class or `None`.

        """
        ...


@runtime_checkable
class _PadTokenCarrierProtocol(Protocol):
    """Protocol for tokenizers exposing `pad_token_id`."""

    @property
    def pad_token_id(self) -> int | None:
        """Provide the padding token ID.

        Returns:
            Integer padding token ID or `None`.

        """
        ...


@runtime_checkable
class _EosTokenCarrierProtocol(Protocol):
    """Protocol for tokenizers exposing `eos_token_id`."""

    @property
    def eos_token_id(self) -> int | None:
        """Provide the end-of-sequence token ID.

        Returns:
            Integer EOS token ID or `None`.

        """
        ...


@runtime_checkable
class _BosTokenCarrierProtocol(Protocol):
    """Protocol for tokenizers exposing `bos_token_id`."""

    @property
    def bos_token_id(self) -> int | None:
        """Provide the beginning-of-sequence token ID.

        Returns:
            Integer BOS token ID or `None`.

        """
        ...


@runtime_checkable
class _VocabCarrierProtocol(Protocol):
    """Protocol for tokenizers exposing `get_vocab()`."""

    def get_vocab(self) -> Mapping[str, int]:
        """Return vocabulary mapping token strings to integer IDs.

        Returns:
            Mapping from token string to integer ID.

        """
        ...


_TRANSFORMERS_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("transformers")
PreTrainedTokenizerProtocol = HFTokenizerProtocol
AutoTokenizer: AutoTokenizerClassProtocol | Callable[..., HFTokenizerProtocol] | None = None


def get_auto_tokenizer_class() -> AutoTokenizerClassProtocol | Callable[..., HFTokenizerProtocol] | None:
    """Lazily resolve the Hugging Face `AutoTokenizer` class without import-time side effects.

    Returns:
        `AutoTokenizerClassProtocol` implementation or `None` if `transformers` is unavailable.

    """
    if AutoTokenizer is not None:
        return AutoTokenizer
    mod = _TRANSFORMERS_LOADER.get_module()
    if isinstance(mod, _TransformersModuleProtocol):
        return mod.AutoTokenizer
    return None


class ByteFallbackTokenizer:
    """Deterministic, reversible UTF-8 byte-level fallback tokenizer with disjoint special-token IDs.

    When `vocab_size - len(special_ids) >= 256` (e.g., default `vocab_size = 259` with
    `pad_token_id = 0`, `eos_token_id = 1`, `bos_token_id = 2`), raw UTF-8 bytes `0..255`
    map 1:1 onto non-special token IDs (`3..258`), preventing collisions between `NUL`/`0x01`/`0x02`
    bytes and `PAD`/`EOS`/`BOS` control tokens. When `2 <= usable_vocab < 256`, each UTF-8 byte
    is reversibly decomposed into fixed-width radix-`usable_vocab` digits over the non-special IDs.

    Attributes:
        vocab_size: Total vocabulary size (`>= 1`).
        pad_token_id: Reserved padding control token ID.
        eos_token_id: Reserved end-of-sequence control token ID.
        bos_token_id: Reserved beginning-of-sequence control token ID.
        byte_offset: First non-special usable token ID for byte digit `0`.
        tokens_per_byte: Number of emitted token IDs per UTF-8 byte.

    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_BYTE_VOCAB_SIZE,
        *,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
    ) -> None:
        """Initialize the byte-level fallback tokenizer.

        Args:
            vocab_size: Maximum token ID bound (`>= 1`, defaults to `259` for 3 special IDs + 256 bytes).
            pad_token_id: Padding token ID.
            eos_token_id: End-of-sequence token ID.
            bos_token_id: Beginning-of-sequence token ID.

        """
        self.vocab_size: int = max(1, int(vocab_size))
        self.pad_token_id: int = int(pad_token_id)
        self.eos_token_id: int = int(eos_token_id)
        self.bos_token_id: int = int(bos_token_id)

        in_range_specials = frozenset(
            tid for tid in (self.pad_token_id, self.eos_token_id, self.bos_token_id) if 0 <= tid < self.vocab_size
        )
        candidate_usable = tuple(tid for tid in range(self.vocab_size) if tid not in in_range_specials)
        if len(candidate_usable) >= 2:
            self._reserves_special_ids: bool = True
            self._special_ids: frozenset[int] = in_range_specials
            self._usable_ids: tuple[int, ...] = candidate_usable
        else:
            self._reserves_special_ids = False
            self._special_ids = frozenset()
            self._usable_ids = tuple(range(self.vocab_size))

        self.byte_offset: int = self._usable_ids[0]
        self._id_to_digit: dict[int, int] = {tid: idx for idx, tid in enumerate(self._usable_ids)}
        usable_count = len(self._usable_ids)
        self.tokens_per_byte: int = 1
        if 2 <= usable_count < 256:
            count = 1
            cap = usable_count
            while cap < 256:
                cap *= usable_count
                count += 1
            self.tokens_per_byte = count

    def byte_to_token_ids(self, byte_value: int) -> list[int]:
        """Convert a single raw byte (`0..255`) into its disjoint token ID sequence.

        Args:
            byte_value: Integer byte value in `[0, 255]`.

        Returns:
            List of token IDs representing `byte_value`.

        """
        clamped = byte_value % 256
        usable_count = len(self._usable_ids)
        if usable_count >= 256:
            return [self._usable_ids[clamped]]
        if usable_count == 1:
            return [self._usable_ids[0]]
        k = self.tokens_per_byte
        digits = [0] * k
        val = clamped
        for pos in range(k - 1, -1, -1):
            digits[pos] = self._usable_ids[val % usable_count]
            val //= usable_count
        return digits

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary mapping special tokens and byte/character labels to integer IDs.

        Returns:
            Dictionary of size `vocab_size` mapping token labels to unique IDs in `[0, vocab_size)`.

        """
        vocab: dict[str, int] = {}
        assigned_ids: set[int] = set()
        if self._reserves_special_ids:
            for label, tid in (
                ("<pad>", self.pad_token_id),
                ("<eos>", self.eos_token_id),
                ("<bos>", self.bos_token_id),
            ):
                if 0 <= tid < self.vocab_size and tid not in assigned_ids:
                    vocab[label] = tid
                    assigned_ids.add(tid)

        if len(self._usable_ids) >= 256:
            for byte_val in range(256):
                tid = self._usable_ids[byte_val]
                label = chr(byte_val) if 32 <= byte_val <= 126 else f"<byte_{byte_val}>"
                vocab[label] = tid
                assigned_ids.add(tid)

        for tid in range(self.vocab_size):
            if tid not in assigned_ids:
                label = f"<tok_{tid}>" if f"<byte_{tid}>" in vocab else f"<byte_{tid}>"
                vocab[label] = tid
                assigned_ids.add(tid)
        return vocab

    def encode_str(self, text: str) -> list[int]:
        """Encode a single UTF-8 string into a list of integer token IDs in `[0, vocab_size)`.

        Args:
            text: Input string to tokenize.

        Returns:
            List of integer token IDs disjoint from special control tokens when `usable_vocab >= 2`.

        """
        raw_bytes = str(text).encode("utf-8", errors="replace")
        usable_count = len(self._usable_ids)
        if usable_count >= 256:
            return [self._usable_ids[b] for b in raw_bytes]
        if usable_count == 1:
            return [self._usable_ids[0] for _ in raw_bytes]

        tokens: list[int] = []
        k = self.tokens_per_byte
        for b in raw_bytes:
            val = b
            digits = [0] * k
            for pos in range(k - 1, -1, -1):
                digits[pos] = self._usable_ids[val % usable_count]
                val //= usable_count
            tokens.extend(digits)
        return tokens

    def decode_tokens(self, tokens: Sequence[int | str | bytes | SupportsInt]) -> str:
        """Decode a sequence of integer token IDs back into a UTF-8 string, ignoring control tokens.

        Args:
            tokens: Sequence of integer token IDs.

        Returns:
            Decoded UTF-8 string.

        """
        parsed_ints: list[int] = []
        for t in tokens:
            if isinstance(t, float):
                continue
            try:
                t_int = int(t)
            except (ValueError, TypeError):
                continue
            if self._reserves_special_ids and t_int in self._special_ids:
                continue
            parsed_ints.append(t_int)

        usable_count = len(self._usable_ids)
        if usable_count >= 256 or usable_count == 1:
            valid_bytes = [
                self._id_to_digit[t]
                if t in self._id_to_digit and self._id_to_digit[t] < 256
                else (t - self.byte_offset) % 256
                for t in parsed_ints
            ]
            return bytes(valid_bytes).decode("utf-8", errors="replace")

        k = self.tokens_per_byte
        reconstructed: list[int] = []
        for idx in range(0, len(parsed_ints), k):
            chunk = parsed_ints[idx : idx + k]
            if len(chunk) < k:
                reconstructed.extend(self._id_to_digit.get(val, val % usable_count) % 256 for val in chunk)
                break
            byte_val = 0
            for tid in chunk:
                digit = self._id_to_digit.get(tid, tid % usable_count)
                byte_val = byte_val * usable_count + digit
            reconstructed.append(byte_val % 256)

        return bytes(reconstructed).decode("utf-8", errors="replace")


class SQLTokenizer:
    """Tokenizer for Text-to-SQL datasets wrapping Hugging Face `AutoTokenizer` or `ByteFallbackTokenizer`.

    Attributes:
        vocab_size: Configured fallback vocabulary size.
        model_name: Optional Hugging Face model identifier or local path.
        allow_fallback: Whether fallback to `ByteFallbackTokenizer` is permitted when HF loading fails.
        allow_byte_fallback: Alias matching `allow_fallback`.
        byte_fallback: Underlying `ByteFallbackTokenizer` instance.
        hf_tokenizer: Loaded Hugging Face tokenizer instance or `None`.
        is_fallback: `True` when operating in byte-fallback mode.
        is_byte_fallback: Alias matching `is_fallback`.

    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_BYTE_VOCAB_SIZE,
        model_name: str | None = None,
        *,
        strict_vocab: bool = False,
        allow_fallback: bool = True,
        allow_byte_fallback: bool | None = None,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
    ) -> None:
        """Initialize the SQL tokenizer.

        Args:
            vocab_size: Vocabulary size for `ByteFallbackTokenizer` (defaults to `259`).
            model_name: Optional model identifier or path to a Hugging Face tokenizer.
            strict_vocab: If `True`, raise `ConfigurationError` when `vocab_size <= 0`.
            allow_fallback: If `False` and `model_name` is set, raise `DependencyMissingError`
                when HF tokenizer fails to load.
            allow_byte_fallback: Alias for `allow_fallback`; if provided, overrides `allow_fallback`.
            pad_token_id: Padding token ID for fallback tokenization.
            eos_token_id: End-of-sequence token ID for fallback tokenization.
            bos_token_id: Beginning-of-sequence token ID for fallback tokenization.

        Raises:
            ConfigurationError: If `strict_vocab=True` and `vocab_size <= 0`.
            DependencyMissingError: If fallback is disabled and the Hugging Face tokenizer cannot be loaded.

        """
        if strict_vocab and int(vocab_size) <= 0:
            raise ConfigurationError(f"vocab_size must be positive, got {vocab_size}")
        self.vocab_size: int = max(1, int(vocab_size))
        self.model_name: str | None = model_name
        resolved_fallback = allow_byte_fallback if allow_byte_fallback is not None else allow_fallback
        self.allow_fallback: bool = resolved_fallback
        self.allow_byte_fallback: bool = resolved_fallback
        self.byte_fallback: ByteFallbackTokenizer = ByteFallbackTokenizer(
            vocab_size=self.vocab_size,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )
        self.hf_tokenizer: HFTokenizerProtocol | None = self._try_load_hf_tokenizer()
        self.is_fallback: bool = self.hf_tokenizer is None
        self.is_byte_fallback: bool = self.hf_tokenizer is None
        if self.model_name and self.hf_tokenizer is None:
            if not self.allow_fallback:
                raise DependencyMissingError(
                    f"Failed to load Hugging Face tokenizer for '{self.model_name}' and allow_fallback=False.",
                    dependency_name="transformers",
                )
            logger.warning("Falling back to deterministic byte-level SQLTokenizer for '%s'.", self.model_name)

    @property
    def pad_token_id(self) -> int:
        """Provide the padding token ID.

        Returns:
            Active padding token ID from `hf_tokenizer` or `byte_fallback`.

        """
        if isinstance(self.hf_tokenizer, _PadTokenCarrierProtocol):
            hf_pad = self.hf_tokenizer.pad_token_id
            if isinstance(hf_pad, int):
                return hf_pad
        return self.byte_fallback.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """Provide the end-of-sequence token ID.

        Returns:
            Active EOS token ID from `hf_tokenizer` or `byte_fallback`.

        """
        if isinstance(self.hf_tokenizer, _EosTokenCarrierProtocol):
            hf_eos = self.hf_tokenizer.eos_token_id
            if isinstance(hf_eos, int):
                return hf_eos
        return self.byte_fallback.eos_token_id

    @property
    def bos_token_id(self) -> int:
        """Provide the beginning-of-sequence token ID.

        Returns:
            Active BOS token ID from `hf_tokenizer` or `byte_fallback`.

        """
        if isinstance(self.hf_tokenizer, _BosTokenCarrierProtocol):
            hf_bos = self.hf_tokenizer.bos_token_id
            if isinstance(hf_bos, int):
                return hf_bos
        return self.byte_fallback.bos_token_id

    def _try_load_hf_tokenizer(self) -> HFTokenizerProtocol | None:
        """Attempt to load a Hugging Face `AutoTokenizer` if `model_name` is set.

        Returns:
            Loaded Hugging Face tokenizer instance or `None` if unavailable.

        """
        if not self.model_name:
            return None
        tok_cls = get_auto_tokenizer_class()
        if tok_cls is None or not isinstance(tok_cls, AutoTokenizerClassProtocol):
            return None
        try:
            loaded: HFTokenizerProtocol = tok_cls.from_pretrained(self.model_name)
            return loaded
        except (ImportError, OSError, ValueError, RuntimeError):
            return None

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary mapping tokens or bytes to token IDs.

        Returns:
            Dictionary mapping token strings or byte representations to integer IDs.

        """
        if isinstance(self.hf_tokenizer, _VocabCarrierProtocol):
            return dict(self.hf_tokenizer.get_vocab())
        return self.byte_fallback.get_vocab()

    @overload
    def encode(self, text: str) -> list[int]:
        """Encode a single string into a list of integer token IDs."""

    @overload
    def encode(self, text: list[str] | tuple[str, ...]) -> list[list[int]]:
        """Encode a sequence of strings into a list of integer token ID lists."""

    def encode(self, text: str | Sequence[str]) -> list[int] | list[list[int]]:
        """Encode a string or sequence of strings into token IDs.

        Args:
            text: Single string or sequence of strings to be tokenized.

        Returns:
            List of integer token IDs for a single string, or list of token ID lists for a sequence of strings.

        """
        if not isinstance(text, str):
            return [self.encode(item) for item in text]

        if self.hf_tokenizer is not None:
            encoded: Sequence[int] = self.hf_tokenizer.encode(str(text), add_special_tokens=False)
            return [int(x) for x in encoded]

        return self.byte_fallback.encode_str(text)

    def decode(self, tokens: Sequence[int | str | bytes | SupportsInt]) -> str:
        """Decode a sequence of token IDs back into a UTF-8 string.

        Args:
            tokens: Sequence of integer token IDs.

        Returns:
            Decoded UTF-8 text string.

        """
        if self.hf_tokenizer is not None:
            int_tokens: list[int] = []
            for t in tokens:
                if isinstance(t, float):
                    continue
                try:
                    int_tokens.append(int(t))
                except (ValueError, TypeError):
                    continue
            return str(self.hf_tokenizer.decode(int_tokens))

        return self.byte_fallback.decode_tokens(tokens)


__all__ = [
    "DEFAULT_BYTE_VOCAB_SIZE",
    "AutoTokenizerClassProtocol",
    "ByteFallbackTokenizer",
    "PreTrainedTokenizerProtocol",
    "SQLTokenizer",
    "get_auto_tokenizer_class",
]
