# Dependency profiles

`pyproject.toml` keeps compatible ranges because Atagia is a library. The files
under `requirements/locks/` are the tested, exact deployment/build resolutions
for supported CPython 3.12 and 3.13 environments. A lock is a constraint for the
published wheel, not a second declaration of the library's public API.

Profiles:

- `core`: the built Atagia wheel and its base runtime dependencies.
- `mcp`: the wheel with the `mcp` extra.
- `dev`: the wheel with `dev`, `mcp`, and `embeddings`; this also pins the lock
  and audit tooling used by the gate.
- `blog`: the private Journal build/publication/runtime-validation dependencies
  declared in `profiles/blog.in`; it is separate from Atagia wheel metadata.
- `bootstrap.txt`: installer tooling pinned before a fresh-profile install.
- Node: each supported integration owns its adjacent lockfile. The current
  OpenClaw plugin declares Node 22.14 or newer, which is also the CI baseline.

Regenerate and compare one interpreter's Python locks with:

```bash
python scripts/verify_dependency_profiles.py lock --python 3.12
python scripts/verify_dependency_profiles.py check-locks --python 3.12
```

Before invoking the gate, install its tooling from the same tested resolution:

```bash
python -m pip install --requirement requirements/locks/bootstrap.txt
python -m pip install \
  --constraint requirements/locks/dev-py312.txt \
  build pip-audit pip-tools
```

Then run the fresh wheel/install/import/audit gates with:

```bash
python scripts/verify_dependency_profiles.py verify --python 3.12
```

Repeat with CPython 3.13. The verify command records wheel metadata and exact
installed inventories plus machine-readable Python and Node audit output under
`build/dependency-audit/`; that directory is a generated CI/local artifact, not
source. The wheel is built without isolation after the gate verifies the pinned
build toolchain. Each runtime is installed into a new temporary virtualenv. A
successful audited run means the resolved profile has no advisory reported by
`pip-audit` or `npm audit`, `pip check` passes, installed versions equal the
lock, and the profile smoke succeeds.

For the private Journal profile, a local checkout containing `_blog_engine/`
stages `blog_security.py`, `build.py`, `publication.py`, the real templates, and
the real content into a temporary workspace. It builds and validates a release
and starts the publication CLI in read-only status mode inside the fresh locked
environment. The report embeds a deterministic manifest of the private source,
content, templates, strategy data, publication scripts, and nginx contract. It
also records the built release manifest SHA-256, release-tree and public-payload
digests, and route/post/file counts. Publication requires that evidence and
rejects a candidate whose payload or current source no longer matches it. A
plain `smoke=passed` without matching `artifact_evidence` is not release proof.
This is the required pre-release evidence for that exact artifact.

Production Journal operators install the matching `blog-py312.txt` or
`blog-py313.txt` lock directly. The private engine's ranged requirements file
is a developer convenience that delegates to `profiles/blog.in`; it is not a
separate deployable resolution.

The GitHub dependency workflow executes lock regeneration checks and the same
build/install/smoke/audit command for both supported Python minors. It also
installs and audits every supported Node package from its adjacent lockfile.
The public sync intentionally excludes the private `_blog_engine/` tree. On
that checkout the report records
`dependency-only-private-artifact-unavailable` for the blog smoke instead of
claiming that it exercised private code. A release from the private checkout
requires a local report whose blog smoke is `passed`; the dependency-only CI
result is not publication evidence.

An accepted exception must be added here with all of: advisory identifier,
affected profile and reachable surface, owner, technical justification,
compensating control, approval date, and an expiry date. Expiry must be a date,
not an open-ended milestone. There are currently no accepted exceptions.
