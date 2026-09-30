"""Download a local Tencent-vector input without committing it to Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "tencent_vectors"
OFFICIAL_URL = "https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz"
MIRROR_REPO = "shibing624/text2vec-word2vec-tencent-chinese"
MIRROR_REVISION = "b7b9fccfd5dd34cfc340607c58986ac2970e3a13"
MIRROR_FILE = "light_Tencent_AILab_ChineseEmbedding.bin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / "Tencent_AILab_ChineseEmbedding.tar.gz"
    request = urllib.request.Request(OFFICIAL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
        content_type = response.headers.get("Content-Type", "")
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if "html" in content_type.lower() or not tarfile.is_tarfile(archive):
        archive.unlink(missing_ok=True)
        raise RuntimeError("官方端点未返回有效 tar.gz；请稍后重试或手动提供官方文件")
    return archive


def download_mirror() -> Path:
    from huggingface_hub import hf_hub_download

    CACHE.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=MIRROR_REPO,
        filename=MIRROR_FILE,
        revision=MIRROR_REVISION,
        local_dir=CACHE,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("official", "light-mirror"), default="official")
    args = parser.parse_args()
    path = download_official() if args.variant == "official" else download_mirror()
    metadata = {
        "variant": args.variant,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
