# operation-prometheus Hermes export (RM-1348)

Mirrored from local `b7fca79` (on top of `7cecb71`).

## Apply on operation-prometheus

```bash
cd ~/rmems/operation-prometheus
git fetch origin main
git checkout -b cursor/hermes-from-agoge-export-1134 7cecb71
git am /path/to/agoge-forger/export/operation-prometheus-hermes/0001-feat-ingest-normalize-verified-Hermes-local-agent-trajectories.patch
git am /path/to/agoge-forger/export/operation-prometheus-hermes/0002-fix-ingest-tighten-Hermes-admission-binding-and-safety-regressions.patch
```

Or single mailbox: `git am pr74_hermes_b7fca79.format-patch`

Base commit: `7cecb71d74b6abe4e447bd646c36b4921d8345ac`
Head commit: `b7fca79f82d53f3d273b1e072c44628bdd53c947`
