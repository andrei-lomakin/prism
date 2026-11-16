# Text Chunking Strategies

## Fixed Windows
A fixed word or token window slides over the text with optional overlap. Overlap must not split words: boundaries should snap to whitespace or tokenizer tokens.

## Paragraph and Sentence Packing
Paragraph packing groups paragraphs until a size budget is reached. Sentence packing does the same with sentence units. Both preserve linguistic boundaries better than naive character cuts.

## Recursive Splitting
Recursive splitting tries large units first (sections, paragraphs), then sentences, then words, until each piece fits the token budget of the embedding model.

## Structure-Aware Chunking
Markdown and HTML headings, PDF page markers, and code fences provide natural boundaries. Metadata should store `section_path` and page numbers for citation UX and evaluation.

## Semantic Chunking
Semantic chunking places a boundary where cosine similarity between consecutive sentence embeddings drops. It is slower but can keep topic-coherent passages together.

## Parent-Child (Small-to-Big)
Small child chunks are used for retrieval precision; parent chunks are returned to the generator for broader context.
