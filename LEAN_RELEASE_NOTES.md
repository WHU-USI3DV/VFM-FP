# Lean Release Candidate

Generated from: release\VFM-FP-open-source
Generated at: 2026-07-04T16:28:54
Mode: Keep + Review
Copied files: 64

This folder is a proposed lean public release candidate.
It excludes files marked Optional in FILE_INVENTORY.md, usually compatibility wrappers or historical helpers.
Keep the broader release candidate until owner review confirms which Review files should remain.

After generation, run:

    powershell -NoProfile -ExecutionPolicy Bypass -File tools/generate_file_inventory.ps1 -Root release\VFM-FP-final
    powershell -NoProfile -ExecutionPolicy Bypass -File tools/audit_release.ps1 -Root release\VFM-FP-final -Limit 100
    powershell -NoProfile -ExecutionPolicy Bypass -File tools/content_audit.ps1 -Root release\VFM-FP-final -Limit 100
    python tools/syntax_check_release.py release\VFM-FP-final
