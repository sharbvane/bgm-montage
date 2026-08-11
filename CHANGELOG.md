# Changelog

## v1.1 — 2026-08-02

- Moved the canonical Skill layout to `.agents/skills/bgm-montage` and corrected project-root installation and invocation examples.
- Unified Pixabay cache paths at `.bgm-montage-cache/pixabay/{search,thumbnails}` and added non-destructive migration from the legacy nested path.
- Added a machine-level material catalog and hard-link/shared-reference reuse for already downloaded Pixabay assets across themes and projects.
- Added bounded three-round query expansion plus hard material gates for independent assets, scene diversity, theoretical coverage, reuse count, per-asset screen share, low-face inventory and prominent-face screen share.
- Added pretrained CLIP zero-shot semantics to sampled reference shots, alongside the existing OpenCV color, exposure, cut, motion, framing and text-region measurements.
- Added saliency and frontal-face subject geometry for source QA and subject-aware crop planning, with blur-fill fallback when crop retention is unsafe.
- Added cached reference-audio analysis and `editing_grammar.json`, including cut alignment to accents, strong/weak beats, phrases, sections, pauses and energy, plus shot-duration and adjacency statistics.
- Connected the learned grammar to actual boundary weighting, energy-dependent shot duration, scale/motion progression, repeated-section variation and ending duration in the timeline.
- Added UTC/random `run_id` directories and refusal to overwrite existing run outputs by default.
- Expanded source manifests with search/rejection records, reuse mode and actual usage intervals; expanded validation with repetition, face-budget and crop-plan checks.
- Added a pinned CPython 3.11 dependency lock, automated tests, and an allowlisted secret-safe ZIP builder using portable `/` member paths.

### Deliberate limitations

The v1.1 analyzer does not identify or reproduce complex transitions, match cuts, masks, speed ramps, subtitle styling, OCR, motion graphics or people identities. CLIP action/emotion output is appearance-based, and subject-aware crop geometry is sampled rather than tracked frame by frame. Linux and Docker extraction are supported by the ZIP layout, but runtime/render verification for those environments is not claimed.
