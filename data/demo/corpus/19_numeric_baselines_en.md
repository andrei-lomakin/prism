# Numeric Baselines Used in Prism Ablations

## Fusion Weights
Hybrid ablations start with dense weight α values of 0.5, 0.72, 0.75, and 0.9. These are starting points for search, not production truths.

## RRF Constant
Reciprocal Rank Fusion uses constant k = 60 in the default hybrid configuration.

## Candidate Widths
Reranking experiments retrieve K ∈ {20, 50} candidates and keep n ∈ {3, 5, 8} after the cross-encoder.

## Chunk Sizes
Default Stage 0 packing uses chunk_size = 256 approximate tokens with overlap = 32 (about 12.5% overlap). Ablations also compare 128 and 512.
