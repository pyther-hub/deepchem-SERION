"""Transformer encoder-decoder for sequence-to-sequence translation.

Uses PyTorch's built-in nn.TransformerEncoder / nn.TransformerDecoder modules
with sinusoidal positional encoding.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn

from utils import generate_causal_mask


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (not learned).

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout rate applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of same shape with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """Transformer encoder-decoder for seq2seq translation.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        dim_feedforward: Feed-forward hidden dimension.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Embeddings + positional encoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.src_pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.tgt_pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for embeddings and linear layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through encoder-decoder.

        Args:
            src_ids: Source token IDs, shape (batch, src_len).
            tgt_input_ids: Decoder input token IDs, shape (batch, tgt_len).
            src_padding_mask: Bool mask, True where padded, shape (batch, src_len).
            tgt_padding_mask: Bool mask, True where padded, shape (batch, tgt_len).

        Returns:
            Logits of shape (batch, tgt_len, tgt_vocab_size).
        """
        # Embed and scale
        scale = math.sqrt(self.d_model)
        src_emb = self.src_pos_enc(self.src_embedding(src_ids) * scale)
        tgt_emb = self.tgt_pos_enc(self.tgt_embedding(tgt_input_ids) * scale)

        # Causal mask for decoder self-attention
        tgt_len = tgt_input_ids.size(1)
        tgt_mask = generate_causal_mask(tgt_len, src_ids.device)

        # Encode
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        # Decode
        decoder_out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        # Project to vocab
        logits = self.output_proj(decoder_out)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        src_padding_mask: torch.Tensor,
        max_len: int,
        sos_id: int,
        eos_id: int,
        device: torch.device,
    ) -> List[List[int]]:
        """Autoregressive greedy decoding for inference.

        Args:
            src_ids: Source token IDs, shape (batch, src_len).
            src_padding_mask: Bool mask, shape (batch, src_len).
            max_len: Maximum decoding length.
            sos_id: Start-of-sequence token ID.
            eos_id: End-of-sequence token ID.
            device: Device to run on.

        Returns:
            List of lists of token IDs, one per batch element.
        """
        self.eval()
        batch_size = src_ids.size(0)
        scale = math.sqrt(self.d_model)

        # Encode source
        src_emb = self.src_pos_enc(self.src_embedding(src_ids) * scale)
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        # Initialize decoder input with <sos>
        ys = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_emb = self.tgt_pos_enc(self.tgt_embedding(ys) * scale)
            tgt_mask = generate_causal_mask(ys.size(1), device)

            decoder_out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )
            logits = self.output_proj(decoder_out[:, -1, :])  # (batch, vocab)
            next_tokens = logits.argmax(dim=-1)  # (batch,)

            # Replace finished sequences' tokens with eos
            next_tokens = next_tokens.masked_fill(finished, eos_id)
            ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)

            finished = finished | (next_tokens == eos_id)
            if finished.all():
                break

        # Convert to list of lists, stripping sos and eos
        results: List[List[int]] = []
        for i in range(batch_size):
            tokens = ys[i].tolist()
            # Remove leading sos
            if tokens and tokens[0] == sos_id:
                tokens = tokens[1:]
            # Remove eos and everything after
            try:
                eos_pos = tokens.index(eos_id)
                tokens = tokens[:eos_pos]
            except ValueError:
                pass
            results.append(tokens)

        return results


def model_summary(model: nn.Module) -> str:
    """Return a string with total params, trainable params, and layer-by-layer counts.

    Args:
        model: PyTorch module.

    Returns:
        Formatted summary string.
    """
    lines = ["Model Summary:", "-" * 60]
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        lines.append(f"  {name:50s} {n:>12,}  {'T' if param.requires_grad else 'F'}")
    lines.append("-" * 60)
    lines.append(f"  Total parameters:     {total:>12,}")
    lines.append(f"  Trainable parameters: {trainable:>12,}")
    lines.append(
        f"  Size: ~{total * 4 / 1e6:.1f} MB (float32)"
    )
    return "\n".join(lines)
