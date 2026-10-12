# Changelog

## v0.2.0

- Default morphology segmentation uses existing Cellpose 3 yeast brightfield weights; classical segmentation remains explicitly selectable.
- Fluorescence rings are proposed independently across overlapping FOV tiles, deduplicated and then associated with cell interiors. Unassigned/edge candidates are retained for review.
- Separate morphology, fluorescence and association overlays make counting and failure review clearer.
- Near-closed angular coverage, signal/noise and original display contrast gates reduce noise amplification. Sparse small rings use a significant-signal normalization fallback.
- Cross-channel co-located proposals share ring group IDs; cell counts, ring counts and unassigned counts are separate.
- Cell identities bind the source morphology and mask, rejecting stale labels/masks after segmentation changes.
- Colab and AutoDL entry points install the compatible pretrained-segmentation dependencies.

No phenotype classifier has been trained or fine-tuned. Candidate scores are not probabilities. Microscopy data, experimental metadata, review labels and run reports remain local. See README for installation and review; THIRD_PARTY_NOTICES.md for model provenance.
