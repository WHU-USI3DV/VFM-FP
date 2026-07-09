# Release Checklist

Use this checklist before publishing VFM-FP as a public repository. See docs/release_status.md for the latest verified release-candidate status and docs/owner_decisions.md for project-owner choices.

## Required Metadata

- Add the final paper title.
- Add author names and affiliations as they should appear in the repository.
- Add citation information once the publication metadata is final. Use docs/citation_template.md as the draft.
- Root project LICENSE is intentionally omitted in this pass per owner decision; revisit before public GitHub release if explicit reuse terms are needed.
- Preserve the upstream MIT license notice for the DeepLabv3+ baseline code.

## Data and Weights

- Remove local datasets from the repository history before publishing.
- Provide download links for allowed datasets or explain how users should prepare their own VOC-style data.
- Provide download links and checksums for trained model weights if redistribution is allowed.
- Keep DINOv2, Stable Diffusion, ControlNet, and other model caches out of git.
- Include only a tiny sample dataset if all rights are clear.

## Code Hygiene

- Keep algorithmic defaults unchanged unless the paper reproduction protocol is updated.
- Keep compatibility wrappers for renamed files during the first public release.
- Move one-off cleanup scripts into documented legacy folders or leave them out of the release.
- Replace hard-coded local dataset paths with CLI arguments or documented config defaults.
- Create a clean local release copy before committing:
  powershell -NoProfile -ExecutionPolicy Bypass -File tools/prepare_release.ps1 -Force
- Run release audit against the clean copy:
  powershell -NoProfile -ExecutionPolicy Bypass -File tools/audit_release.ps1 -Root release/VFM-FP-open-source -Limit 100
- Run content audit against the clean copy:
  powershell -NoProfile -ExecutionPolicy Bypass -File tools/content_audit.ps1 -Root release/VFM-FP-open-source -Limit 100
- The full working directory may still fail audit while local datasets and weights remain beside the code.

## Reproducibility

- Document the expected class order:
  background, window, door, facade, balcony, roof, shop
- Document dataset split files used by the accepted experiments.
- Document training checkpoints used for reported results.
- Document CUDA, PyTorch, diffusers, transformers, and DINOv2 versions that were actually tested.
- Add a minimal smoke test for importability, config parsing, and syntax checking.

## Smoke Check

- Run the lightweight smoke check after cleanup:
  powershell -NoProfile -ExecutionPolicy Bypass -File tools/smoke_check.ps1

## Final Git Check

- Confirm git status contains only source code, docs, configs, and small text files.
- Confirm git check-ignore -v excludes datasets, logs, weights, and generated images.
- Confirm no private absolute paths, API keys, or local account names remain in public docs.


