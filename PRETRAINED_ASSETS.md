# Pretrained Assets

Pretrained checkpoints and model artifacts are kept out of normal Git because
the full local set is over 1 GB. The tracked manifest records the expected
paths, sizes, and SHA-256 hashes.

Create or refresh the manifest on a machine that already has the assets:

```bash
uv run python scripts/pretrained_assets.py manifest
```

Verify the current checkout:

```bash
uv run python scripts/pretrained_assets.py verify
```

Create a transferable archive:

```bash
uv run python scripts/pretrained_assets.py pack --verify-first
```

Copy `pretrained-assets.tar.gz` to a new machine, then run:

```bash
uv run python scripts/pretrained_assets.py unpack --verify-after
```

The archive path is ignored by Git. Commit the manifest and this helper script,
but do not commit the archive itself.
