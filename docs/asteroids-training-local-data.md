# Local experiment data across GitHub Desktop branches

The `training` branch ignores the whole `outputs/` tree. The actual Mac
connectome, saved neural traces, calibration chains, and per-run protocols are
local inputs; small reviewed results are published separately in
`docs/evidence/`. A remote branch alone does not transfer local output data
to another computer.

## One-time migration from `codex/asteroids-integration`

After restoring its GitHub Desktop stash, run from the repository root:

```bash
test -z "$(git ls-files outputs)" && test -f outputs/asteroids/policy-temporal-saved-trace-capacity-v1/results.json && mkdir -p "$HOME/Documents/GitHub/doomfly-experiment-backup" && rsync -a --progress outputs/ "$HOME/Documents/GitHub/doomfly-experiment-backup/outputs/" && EXCLUDE_FILE="$(git rev-parse --git-path info/exclude)" && { grep -qxF '/outputs/' "$EXCLUDE_FILE" || printf '\n/outputs/\n' >> "$EXCLUDE_FILE"; } && git check-ignore outputs/asteroids/policy-temporal-saved-trace-capacity-v1/results.json && test -f "$HOME/Documents/GitHub/doomfly-experiment-backup/outputs/asteroids/policy-temporal-saved-trace-capacity-v1/results.json"
```

This backs up the entire outputs tree outside the repository and locally excludes
it from GitHub Desktop's uncommitted-change stash on the old branch. The
`.git/info/exclude` entry is local to this clone and is not committed.
If `git ls-files outputs` lists tracked files, stop and inspect those before
switching; ignore rules do not preserve tracked files across checkouts.

Then use GitHub Desktop to Fetch origin, switch to `training`, and Pull origin.
Choose **Leave my changes** for any remaining code/document changes on the
integration branch. The output tree stays in the same working directory while
the code switches branches. Verify:

```bash
test -f outputs/asteroids/policy-temporal-saved-trace-capacity-v1/results.json && echo "Saved training inputs available"
```

From then on, run experiments on `training`. New local outputs persist across
branch switches; they are not uploaded automatically to GitHub. Preserve
important result JSONs under `docs/evidence/` with source hashes and interpretation.
The external backup is a separate recovery copy, not the authoritative
published evidence.
