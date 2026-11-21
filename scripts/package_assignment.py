"""打包报告、源码和复现说明；排除凭据、环境、原始数据与模型缓存。"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in ["reports/verification.json", "reports/document_verification.json", "reports/信用卡欺诈预测_分类实验报告.docx"]:
        if not (ROOT / name).exists():
            raise SystemExit(f"缺少交付核验文件：{name}")
    team=json.loads((ROOT / "configs/team.json").read_text(encoding="utf-8"))
    filename="分类作业_"+"_".join(f"{m['name']}_{m['student_id']}" for m in team)+".zip"
    output=ROOT / "outputs";output.mkdir(exist_ok=True)
    files=[]
    for folder in ["src", "scripts", "configs", "tests", "docs", "reports", "作业要求"]:
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.extend(ROOT / name for name in ["README.md", "requirements.txt", "requirements-lock.txt", "requirements-docs.txt", ".gitignore", ".gitattributes"])
    archive=output / filename
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(set(files)):
            relative = path.relative_to(ROOT)
            # 报告快照与运行缓存分开，解压后 all 可重新训练，不能误复用缺失的模型。
            target = (Path("实验报告") / Path(*relative.parts[1:])) if relative.parts[0] == "reports" else relative
            z.write(path,target.as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert not any(part in {".venv","data","artifacts",".git"} for part in Path(name).parts)
            assert Path(name).name not in {"kaggle.json",".env"}
    with archive.open("rb") as stream: digest=hashlib.file_digest(stream,"sha256").hexdigest()
    (output / "package_manifest.json").write_text(json.dumps({"file":filename,"bytes":archive.stat().st_size,"sha256":digest,"files":len(files),"data_recovery":"按 README 使用 Kaggle API 下载官方原始数据，再运行 scripts/run.py all 重建模型与评估。","contributions":"按用户要求留白，正式提交前需补齐。"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(archive)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
