from __future__ import annotations

import re
from typing import Any, Literal

from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.core.config import Settings

ChunkingStrategy = Literal["fixed", "recursive", "semantic"]

_RUSSIAN_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?…])(?:[\"»)\]]*)\s+|\n+"
)


def russian_sentence_tokenizer(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in _RUSSIAN_SENTENCE_BOUNDARY.split(text)
        if sentence.strip()
    ]


def build_e5_embedding(settings: Settings) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        query_instruction="query: ",
        text_instruction="passage: ",
        normalize=True,
        embed_batch_size=settings.embedding_batch_size,
        cache_folder=str(settings.embedding_cache_dir),
        show_progress_bar=False,
    )


def build_fixed_size_parser(
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    *,
    tokenizer: Any | None = None,
) -> TokenTextSplitter:
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer,
        include_metadata=True,
    )


def build_recursive_parser(
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    *,
    tokenizer: Any | None = None,
) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer,
        paragraph_separator="\n\n",
        chunking_tokenizer_fn=russian_sentence_tokenizer,
        include_metadata=True,
    )


def build_semantic_parser(
    embed_model: Any,
    *,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
) -> SemanticSplitterNodeParser:
    return SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        sentence_splitter=russian_sentence_tokenizer,
        include_metadata=True,
    )


def build_chunk_parser(
    strategy: ChunkingStrategy,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    tokenizer: Any | None = None,
    embed_model: Any | None = None,
) -> Any:
    if strategy == "fixed":
        return build_fixed_size_parser(
            chunk_size,
            chunk_overlap,
            tokenizer=tokenizer,
        )
    if strategy == "recursive":
        return build_recursive_parser(
            chunk_size,
            chunk_overlap,
            tokenizer=tokenizer,
        )
    if strategy == "semantic":
        if embed_model is None:
            raise ValueError("semantic chunking requires embed_model")
        return build_semantic_parser(embed_model)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")
