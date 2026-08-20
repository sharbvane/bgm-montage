# bgm-montage v1.4.4 release verification

This file records the release checks for the published v1.4.4 source snapshot. It intentionally contains no local media, cache, test-output, credential, or machine-specific path.

## Release checks

- CLI version must print `bgm-montage 1.4.4`.
- Contract/unit tests, JSON integration tests, render smoke tests, dedicated failure fixtures, and Golden frozen QA replay are run from the development source before release.
- The runtime ZIP is checked in a clean extraction without importing the project checkout or an installed `.agents/skills/bgm-montage` directory.
- The release asset is checked by SHA-256 after downloading it from the GitHub Release.

The final pre-release source-root run completed with `124 passed, 5 warnings` in approximately 42.75 seconds. The CLI check printed `bgm-montage 1.4.4`; the warnings are existing audio-library deprecation/fallback warnings. The clean-extract package test passed from the canonical Skill root.

The exact command outputs and gate-by-gate decision are recorded in the release task evidence. This repository snapshot is the source of truth for the released code; the attached runtime ZIP is a separate distribution artifact.

## Known boundaries

- Duration: `not reproduced / instrumented / known historical risk`.
- Color: `pl=0 fixed / evidence boundary retained`.
- Planner: further Reference Grammar `beat_cut` rhythm optimization is deferred.
