# Release Status

This document records the current open-source readiness state for VFM-FP.

## Current Public Release Candidate

A source-only release candidate can be generated at:

    release/VFM-FP-open-source

Create or refresh it with:

    powershell -NoProfile -ExecutionPolicy Bypass -File tools/prepare_release.ps1 -Force
    powershell -NoProfile -ExecutionPolicy Bypass -File tools/prepare_lean_release.ps1 -Force

## Verification Commands

Run these checks before publishing:

    powershell -NoProfile -ExecutionPolicy Bypass -File tools/smoke_check.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File tools/audit_release.ps1 -Root release/VFM-FP-open-source -Limit 100
    powershell -NoProfile -ExecutionPolicy Bypass -File tools/content_audit.ps1 -Root release/VFM-FP-open-source -Limit 100
    python tools/syntax_check_release.py release/VFM-FP-open-source

The latest verified clean release copy passed the smoke, release audit, content audit, and syntax checks.

## Completed Cleanup

- Added repository-level README, requirements, configs, release notes, and audit tools.
- Added .gitignore and .gitattributes for datasets, generated images, logs, caches, archives, and model weights.
- Added a clean release copy generator that excludes local data and generated artifacts.
- Added a lean release generator for Keep and Review inventory items.
- Added asset, content, and Python syntax audit scripts for the clean release copy.
- Added a file-review guide for the remaining owner decisions before final trimming.
- Grouped one-off and historical scripts into legacy folders while keeping compatibility wrappers.
- Renamed typo-prone modules such as dataloader_lastest.py to dataloader_latest.py while preserving old imports.
- Parameterized SDA DINO ranking entrypoints and Mul_Ab_norway.py while preserving accepted-paper defaults.
- Preserved core model definitions, training loop logic, metrics, inference behavior, DINO ranking formula, and ControlNet generation settings.

## Release Candidate Scope

The clean release copy is intended to include:

- Source code for SDA data expansion and DINO ranking.
- Source code for VCFS, the VFM-CNN fusion segmentor for facade parser training, inference, and evaluation.
- Lightweight configs and class metadata.
- Documentation, release checklist, and audit tools.

The clean release copy intentionally excludes:

- Local datasets and VOC-style image folders.
- Synthetic images, masks, visualizations, and paper figures.
- Model checkpoints, weights, archives, and Hugging Face caches.
- Logs, debug folders, mIoU outputs, Python caches, and IDE state.
- Local experiment-only folders skipped by tools/prepare_release.ps1.

## Required User Decisions Before Public GitHub Release

These items should be finalized by the project owner before publishing. See docs/owner_decisions.md for the current decision sheet:

1. Root project LICENSE intentionally omitted in this pass per owner decision.
2. Provide final publication metadata and add CITATION.cff once DOI and official issue details are available.
3. Source-only release selected; datasets and trained weights are not redistributed.
4. Add official download links and checksums for datasets or weights if redistribution is allowed.
5. Confirm the final paper title, author list, venue, year, pages, and DOI.

## Known Metadata Gap

The final PDF metadata has been scanned for title and author fields, and docs/citation_template.md now contains a partially filled citation draft. Official publication metadata is still needed before publishing a root CITATION.cff: year, journal issue details, pages, and DOI.
