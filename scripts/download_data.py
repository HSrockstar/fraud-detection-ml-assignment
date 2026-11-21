"""通过官方 Kaggle API 下载比赛数据，不复制或输出账号凭据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "playground-series-s6e11"
FILES = ("train.csv", "test.csv", "sample_submission.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="重新下载并覆盖本项目的数据副本")
    args = parser.parse_args()
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if args.force or not all((raw / name).exists() for name in FILES):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        try:
            api.competition_download_files(COMPETITION, path=str(raw), quiet=False, force=args.force)
        except Exception as exc:
            # 不输出响应对象、请求头或本机凭据内容。
            raise SystemExit(
                f"Kaggle 下载失败（{type(exc).__name__}）。请确认网络、API 凭据和比赛报名状态；"
                f"在 https://www.kaggle.com/competitions/{COMPETITION}/rules 接受规则后重试。"
            ) from None
        archive = raw / f"{COMPETITION}.zip"
        with zipfile.ZipFile(archive) as z:
            for name in FILES:
                matches = [n for n in z.namelist() if Path(n).name == name]
                if len(matches) != 1:
                    raise ValueError(f"压缩包中的 {name} 缺失或重复")
                # 仅写入预先指定的三个文件名，不采用压缩包提供的目标路径。
                with z.open(matches[0]) as source, (raw / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
    manifest = {
        "competition": COMPETITION,
        "source": f"https://www.kaggle.com/competitions/{COMPETITION}/data",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            name: {"bytes": (raw / name).stat().st_size,
                   "sha256": hashlib.file_digest((raw / name).open("rb"), "sha256").hexdigest()}
            for name in FILES
        },
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
