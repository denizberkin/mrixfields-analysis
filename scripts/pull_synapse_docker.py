#!/usr/bin/env python3
"""Pull files out of a Synapse-hosted Docker image without a Docker daemon.

The test-phase submission lives only in ``docker.synapse.org/syn76236366/task3:v2``: the
weights it ships (``avg_e6_e8.pt``) and its ``inference.py`` are gitignored locally and the
run they came from is on another machine. The Synapse registry is a standard Docker
Registry v2, so the image can be read back layer by layer over HTTPS with the same PAT
``scripts/leaderboard.py`` uses -- no daemon, so it works on Windows without Docker
Desktop and inside Colab alike.

    python scripts/pull_synapse_docker.py history                    # config + one line per layer
    python scripts/pull_synapse_docker.py list --layers -3 -2 -1     # files in the last three layers
    python scripts/pull_synapse_docker.py extract --layers -2 -1 \\
        --include '*.pt' '*.py' '*.txt' '*.toml' '*.sh' 'Dockerfile*' --out docker_v2/

``--layers`` takes indices into the manifest's layer list (negative from the end); with
none given every layer is read, which for this image means ~3.9 GB of CUDA/torch base
layers that hold nothing of ours. Layers are cached under ``--cache`` by digest so a
second command does not download them again.

Authentication: ``PERSONAL_ACCESS_TOKEN`` from the repo-root .env (or the environment).
The token is exchanged for a short-lived registry bearer token and never printed.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REGISTRY = "https://docker.synapse.org"
SYNAPSE_REST = "https://repo-prod.prod.sagebase.org/repo/v1"
MANIFEST_TYPES = ", ".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
])

#: Paths that belong to the base image, not to the submission. Skipped by ``list`` unless
#: --all is passed, because a torch install is ~100k files and hides the ten that matter.
SYSTEM_PREFIXES = ("usr/", "lib/", "lib64/", "bin/", "sbin/", "etc/", "var/", "proc/", "sys/",
                   "dev/", "run/", "boot/", "srv/", "media/", "mnt/", "root/.cache/",
                   "opt/conda/pkgs/", "opt/nvidia/", "tmp/")
SITE_PACKAGES = re.compile(r"(^|/)(site|dist)-packages/")


def token_from_env() -> str:
    from mrixfields.env import load_env

    load_env()
    token = os.environ.get("PERSONAL_ACCESS_TOKEN")
    if not token:
        raise SystemExit("PERSONAL_ACCESS_TOKEN is not set (repo-root .env or environment).")
    return token


class Registry:
    """Minimal Docker Registry v2 client with the Synapse bearer-token dance."""

    def __init__(self, repository: str, pat: str) -> None:
        self.repository = repository
        self.session = requests.Session()
        # The registry wants a Synapse *username* with the PAT as password when it issues
        # bearer tokens; the PAT alone identifies the account through the REST API.
        profile = self.session.get(f"{SYNAPSE_REST}/userProfile",
                                   headers={"Authorization": f"Bearer {pat}"}, timeout=30)
        profile.raise_for_status()
        self.username = profile.json()["userName"]
        self.pat = pat
        self.bearer: str | None = None

    def _authorize(self, challenge: str, scope: str) -> None:
        realm = re.search(r'realm="([^"]+)"', challenge).group(1)
        service = re.search(r'service="([^"]+)"', challenge).group(1)
        response = self.session.get(realm, params={"service": service, "scope": scope},
                                    auth=(self.username, self.pat), timeout=30)
        response.raise_for_status()
        self.bearer = response.json().get("token") or response.json().get("access_token")

    def get(self, path: str, *, accept: str | None = None, stream: bool = False) -> requests.Response:
        scope = f"repository:{self.repository}:pull"
        for attempt in (1, 2):
            headers = {}
            if self.bearer:
                headers["Authorization"] = f"Bearer {self.bearer}"
            if accept:
                headers["Accept"] = accept
            response = self.session.get(f"{REGISTRY}/v2/{path}", headers=headers,
                                        stream=stream, timeout=600)
            if response.status_code == 401 and attempt == 1:
                self._authorize(response.headers.get("Www-Authenticate", ""), scope)
                continue
            response.raise_for_status()
            return response
        raise RuntimeError("unreachable")

    def manifest(self, reference: str) -> dict:
        response = self.get(f"{self.repository}/manifests/{reference}", accept=MANIFEST_TYPES)
        manifest = response.json()
        if "manifests" in manifest:  # multi-arch index: take the linux/amd64 entry
            entry = next((m for m in manifest["manifests"]
                          if m.get("platform", {}).get("architecture") == "amd64"),
                         manifest["manifests"][0])
            return self.manifest(entry["digest"])
        manifest["_digest"] = response.headers.get("Docker-Content-Digest")
        return manifest

    def blob(self, digest: str, cache: Path) -> Path:
        """Download a blob into ``cache`` (named by digest), verifying the sha256."""
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / digest.replace(":", "_")
        if target.is_file():
            return target
        partial = target.with_suffix(".part")
        hasher = hashlib.sha256()
        with self.get(f"{self.repository}/blobs/{digest}", stream=True) as response, \
                partial.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                hasher.update(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {digest[:19]}  {done / 1e6:8.1f} / {total / 1e6:.1f} MB",
                          end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        if f"sha256:{hasher.hexdigest()}" != digest:
            partial.unlink()
            raise RuntimeError(f"digest mismatch for {digest}")
        partial.rename(target)
        return target


def is_system(name: str) -> bool:
    return name.startswith(SYSTEM_PREFIXES) or bool(SITE_PACKAGES.search(name))


def select_layers(manifest: dict, indices: list[int] | None) -> list[tuple[int, dict]]:
    layers = manifest["layers"]
    if not indices:
        return list(enumerate(layers))
    chosen = []
    for index in indices:
        resolved = index if index >= 0 else len(layers) + index
        if not 0 <= resolved < len(layers):
            raise SystemExit(f"layer index {index} out of range for {len(layers)} layers")
        chosen.append((resolved, layers[resolved]))
    return sorted(chosen)


def open_layer(path: Path) -> tarfile.TarFile:
    # Layers are gzip (docker) or possibly zstd/uncompressed (OCI); tarfile sniffs gzip.
    return tarfile.open(path, mode="r:*")


def cmd_history(registry: Registry, manifest: dict, cache: Path) -> None:
    config_path = registry.blob(manifest["config"]["digest"], cache)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = config.get("config", {})
    print(f"image digest : {manifest.get('_digest')}")
    print(f"architecture : {config.get('architecture')}/{config.get('os')}")
    print(f"created      : {config.get('created')}")
    print(f"entrypoint   : {cfg.get('Entrypoint')}")
    print(f"cmd          : {cfg.get('Cmd')}")
    print(f"workdir      : {cfg.get('WorkingDir')}")
    for item in cfg.get("Env") or []:
        print(f"env          : {item}")
    print()
    # History entries with empty_layer=true have no blob; the rest map onto layers in order.
    layers = iter(enumerate(manifest["layers"]))
    print("layer  size(MB)  created_by")
    for entry in config.get("history", []):
        created_by = (entry.get("created_by") or "").replace("/bin/sh -c #(nop) ", "").strip()
        if entry.get("empty_layer"):
            print(f"  -    {'':>8}  {created_by[:110]}")
            continue
        index, layer = next(layers)
        print(f"{index:>3}   {layer['size'] / 1e6:8.1f}  {created_by[:110]}")


def cmd_list(registry: Registry, manifest: dict, cache: Path, indices, show_all: bool) -> None:
    for index, layer in select_layers(manifest, indices):
        path = registry.blob(layer["digest"], cache)
        print(f"--- layer {index}  {layer['digest'][:19]}  {layer['size'] / 1e6:.1f} MB")
        with open_layer(path) as archive:
            for member in archive:
                if not show_all and is_system(member.name):
                    continue
                kind = "d" if member.isdir() else ("l" if member.issym() else "f")
                print(f"  {kind} {member.size:>12}  {member.name}")


def cmd_extract(registry: Registry, manifest: dict, cache: Path, indices, patterns, out: Path) -> None:
    """Apply the selected layers in order into ``out``, honouring whiteouts.

    Only members matching one of ``patterns`` (fnmatch on the full path or the basename)
    are written, so a torch install is never unpacked. Whiteout markers (``.wh.<name>``)
    delete an earlier layer's file, which is what makes the result the image's final view
    rather than a pile of every historical copy.
    """
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for index, layer in select_layers(manifest, indices):
        path = registry.blob(layer["digest"], cache)
        with open_layer(path) as archive:
            for member in archive:
                name = member.name.lstrip("./")
                base = os.path.basename(name)
                if base.startswith(".wh."):
                    victim = out / os.path.dirname(name) / base[len(".wh."):]
                    if victim.exists():
                        victim.unlink() if victim.is_file() else None
                    continue
                if not member.isfile():
                    continue
                if not any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(base, p) for p in patterns):
                    continue
                target = out / name
                if not str(target.resolve()).startswith(str(out.resolve())):
                    continue  # path traversal guard
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                assert source is not None
                with target.open("wb") as handle:
                    while chunk := source.read(1 << 20):
                        handle.write(chunk)
                written.append((index, name, member.size))
    for index, name, size in written:
        print(f"layer {index}: {name}  ({size / 1e6:.1f} MB)")
    print(f"{len(written)} file(s) -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("history", "list", "extract"))
    parser.add_argument("--image", default="syn76236366/task3:v2",
                        help="<repository>:<tag> or <repository>@<digest> on docker.synapse.org")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="layer indices (negative counts from the end); default all")
    parser.add_argument("--include", nargs="*",
                        default=["*.pt", "*.pth", "*.py", "*.txt", "*.toml", "*.json", "*.sh",
                                 "*.md", "*.yaml", "*.yml", "Dockerfile*", "*.cfg"],
                        help="extract: fnmatch patterns on the full path or basename")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docker_pulled",
                        help="extract: output directory (gitignored: keep it out of the repo)")
    parser.add_argument("--cache", type=Path, default=None,
                        help="layer cache; default <out>/.layers")
    parser.add_argument("--all", action="store_true",
                        help="list: include base-image paths (usr/, site-packages, ...)")
    args = parser.parse_args()

    repository, _, reference = args.image.partition("@") if "@" in args.image else (
        args.image.rpartition(":")[0], None, args.image.rpartition(":")[2])
    cache = args.cache or (args.out / ".layers")
    registry = Registry(repository, token_from_env())
    manifest = registry.manifest(reference)
    print(f"{repository}:{reference}  {len(manifest['layers'])} layers, "
          f"{sum(l['size'] for l in manifest['layers']) / 1e6:.1f} MB", file=sys.stderr)

    if args.command == "history":
        cmd_history(registry, manifest, cache)
    elif args.command == "list":
        cmd_list(registry, manifest, cache, args.layers, args.all)
    else:
        cmd_extract(registry, manifest, cache, args.layers, args.include, args.out)


if __name__ == "__main__":
    main()
