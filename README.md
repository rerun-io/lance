# Rerun's fork of Lance

This is [Rerun](https://rerun.io)'s fork of [Lance](https://github.com/lancedb/lance).

It exists to carry a small set of **Rerun-only patches** on top of an official upstream Lance
release until those patches are upstreamed (or no longer needed). The goal of this document is to
make the fork **boring to maintain**: a predictable, repeatable process for adding patches and for
rebasing them onto each new upstream release.

For the upstream project README (what Lance is, how to use it), see
[the upstream repo](https://github.com/lancedb/lance/blob/main/README.md).

---

## Repository layout

Two git remotes:

| Remote     | URL                          | Role                                   |
|------------|------------------------------|----------------------------------------|
| `upstream` | `lancedb/lance`              | Upstream. Source of releases.          |
| `rerun`    | `rerun-io/lance`             | Our fork. Where we push everything.    |

Set them up once in a fresh clone:

```sh
git clone git@github.com:rerun-io/lance.git
cd lance
git remote rename origin rerun                        # if it cloned as "origin"
git remote add upstream git@github.com:lancedb/lance.git
git config --local remote.pushDefault rerun           # push everything to the fork by default
git fetch --all --tags
```

Key branches:

| Branch                 | Meaning                                                                       |
|------------------------|-------------------------------------------------------------------------------|
| `rerun/main`           | Mirror of upstream `main`. We do **not** put Rerun patches here.               |
| `rerun/release-X.Y.Z`  | The thing we actually ship: upstream tag `vX.Y.Z` + our Rerun-only commits.    |

We do **not** publish to crates.io. Downstream (Rerun) consumes this fork as a **git dependency**
pinned to a `release-X.Y.Z` branch (or a commit SHA on it). There are therefore no Rerun-specific
version bumps and no Rerun git tags — the workspace `version` stays identical to upstream's.

---

## The model

```
        upstream v7.0.0 (tag on upstream)
              │
              ▼
   ┌───────────────────────────┐
   │  rerun/release-7.0.0       │   ← branch = upstream tag + our patches, in order
   │                            │
   │  • optimize_expr traced    │ ─┐
   │  • execute/analyze traced  │  │  Rerun-only commits.
   │  • Scanner::projection_…   │  │  This list is the entire fork.
   │  • Fix write-starvation    │  │  Keep it SMALL.
   │  • reject CreateIndex …    │  │
   │  • Azure https support     │ ─┘
   └───────────────────────────┘
```

The complete set of Rerun-only commits on any release branch is exactly:

```sh
git log --oneline vX.Y.Z..rerun/release-X.Y.Z
```

If that command shows a commit you don't recognize, something went wrong in a rebase. The list
should be short and every commit should be a deliberate Rerun patch.

### Rules

- **Never rewrite `rerun/main`.** It tracks upstream only.
- **Never force-push a `release-X.Y.Z` branch that Rerun is already pinned to** without coordinating
  — downstream builds pin to it. Cut a new branch or append instead (see below).

---

## Adding a new Rerun-only patch

Branch off the current release branch, do the work, open a PR into that release branch.

Guidelines:

- **Target the active `release-X.Y.Z` branch**, never `rerun/main`.
- In the PR body, **always note the upstreaming status**: link the upstream PR/issue, or say it is
  fork-only and why. This is what lets us delete the patch later.
- After merge, the commit becomes part of the `vX.Y.Z..rerun/release-X.Y.Z` patch set that future
  rebases must carry.

---

## CI on release branches

Upstream scopes its workflows to `main` and `release/**`, which does not match our hyphenated
`release-X.Y.Z`. Renaming our branches would be the other way to fix that, but downstream pins to
them by name, so instead the workflows we want carry an extra `release-*` branch filter.

Two things then decide what actually runs.

**Runner labels.** Most of upstream's build and test jobs request larger runners — `ubuntu-24.04-8x`,
`ubuntu-24.04-arm64-8x`, `ubuntu-24.04-4x`, `warp-macos-14-arm64-6x`, `windows-latest-4x` — that exist
only in the upstream org. **A job requesting a label that does not resolve queues for 24 hours and is
then cancelled**, which is why the Rust and Python runs on `rerun/main` have never gone green. Those
jobs therefore carry `if: github.repository != 'rerun-io/lance'` so they skip here instead of hanging.
`if:` is evaluated before scheduling, so a skipped job never queues.

**Upstream's disabled bit does not travel.** `notebook.yml` is `disabled_manually` upstream, but a
fork inherits the workflow *file*, not that state — give it a matching branch filter and it wakes up
and fails (it installs the wheel plus `jupyter` and `duckdb`, while `quickstart.ipynb` also imports
`pandas`). Before enabling anything here, check upstream first:

```sh
gh api repos/lance-format/lance/actions/workflows --jq '.workflows[] | "\(.state)\t\(.path)"'
```

What runs on a PR into a release branch:

| Workflow                   | What it checks                                                     |
|----------------------------|---------------------------------------------------------------------|
| `rust.yml`                 | `clippy` (all features, all targets), `cargo-deny`, `cargo fmt`, `rustdoc`, MSRV, and the qemu pre-Haswell SIGILL test |
| `java.yml`                 | clippy and fmt for `java/lance-jni`                                  |
| `typos.yml`                | Spelling, whole repo                                                 |
| `license-header-check.yml` | Apache headers under `rust/`, `python/`, `protos/`                   |
| `docs-check.yml`           | `docs/` still builds                                                 |

`python.yml` is not enabled: its only standard-runner job is `compat`, which `needs: linux`, and
`linux` is on a larger runner — so it would produce nothing.

So the lint, advisory and dependency gates do cover you, but **the build-and-test matrix and the
Python suite do not run here.** Verify those locally and say what you ran in the PR body.

---

## Cutting a new fork release when upstream releases

When upstream ships a new release `vNEW` (e.g. `v9.0.0`), rebase our patch set onto it.

### 1. Fetch upstream and confirm the target tag

```sh
git fetch upstream --tags
git tag --list 'v*' | sort -V | tail        # find the new release tag, e.g. v9.0.0
```

### 2. Capture the current patch set

```sh
# OLD = the release we're currently on, e.g. v7.0.0
git log --oneline v7.0.0..rerun/release-7.0.0     # eyeball the patches we're about to move
```

### 3. Create the new release branch and rebase the patches onto it

```sh
git switch -c release-9.0.0 v9.0.0                 # start from the new upstream tag

# Replay our patches (everything that was on the old release branch but not in the old tag)
# --onto release-9.0.0 puts them on the new base; v7.0.0 is the old base they came from.
git rebase --onto release-9.0.0 v7.0.0 rerun/release-7.0.0
```

Resolve conflicts commit-by-commit. For each conflict:

```sh
# edit files, then:
git add -A
git rebase --continue
# if a patch has been upstreamed and is now redundant:
git rebase --skip
```

When a patch was upstreamed in `vNEW`, **skip it** — that's the payoff for upstreaming.

### 4. Verify

```sh
git log --oneline v9.0.0..HEAD                     # should be our patch set, minus any upstreamed ones
cargo fmt --all
cargo clippy --all --tests --benches -- -D warnings
cargo test --workspace                             # or at least the crates our patches touch
```

The workspace version should now read the upstream `vNEW` version (e.g. `9.0.0`) unchanged — we do
not bump it.

### 5. Publish the new release branch

```sh
git switch -c release-9.0.0       # if not already on it
git push rerun release-9.0.0
```

### 6. Point downstream at it

Update Rerun's `Cargo.toml` git dependency to the new branch (or a pinned SHA on it):

```toml
lance = { git = "https://github.com/rerun-io/lance.git", branch = "release-9.0.0" }
```

Pin to a **commit SHA** rather than the bare branch if you need reproducible builds that don't move
when the release branch gets a new patch appended.

### 7. Keeping `rerun/main` current (optional housekeeping)

```sh
git fetch upstream
git push rerun upstream/main:main                  # fast-forward our mirror of upstream main
```

---

## Quick reference

```sh
# What are our patches on the current release?
git log --oneline vX.Y.Z..rerun/release-X.Y.Z

# Add a patch
git switch -c emilk/my-fix rerun/release-X.Y.Z && ... && gh pr create --base release-X.Y.Z --draft

# Rebase onto a new upstream release
git switch -c release-NEW vNEW
git rebase --onto release-NEW vOLD rerun/release-OLD
git push rerun release-NEW
```
