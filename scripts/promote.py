#!/usr/bin/env python
"""
promote.py - sandbox에서 테스트 통과한 코드를 src/로 이동

Usage:
    python scripts/promote.py sandbox/claude/2025-01-09/attempt_01/model.py src/models/
    python scripts/promote.py sandbox/claude/2025-01-09/attempt_01/ src/models/ --all
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime


def promote_file(source: Path, dest_dir: Path, backup: bool = True):
    """단일 파일을 src/로 이동"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / source.name

    # 기존 파일 백업
    if backup and dest_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = dest_file.with_suffix(f".bak.{timestamp}{dest_file.suffix}")
        shutil.copy2(dest_file, backup_file)
        print(f"  Backup: {dest_file.name} -> {backup_file.name}")

    # 파일 복사
    shutil.copy2(source, dest_file)
    print(f"  Promoted: {source} -> {dest_file}")
    return dest_file


def promote_directory(source_dir: Path, dest_dir: Path, pattern: str = "*.py"):
    """디렉토리의 모든 파일을 src/로 이동"""
    files = list(source_dir.glob(pattern))
    if not files:
        print(f"No files matching '{pattern}' in {source_dir}")
        return []

    promoted = []
    for f in files:
        promoted.append(promote_file(f, dest_dir))
    return promoted


def main():
    parser = argparse.ArgumentParser(description="Promote sandbox code to src/")
    parser.add_argument("source", help="Source file or directory in sandbox/")
    parser.add_argument("dest", help="Destination directory in src/")
    parser.add_argument("--all", action="store_true", help="Promote all .py files in directory")
    parser.add_argument("--pattern", default="*.py", help="File pattern for --all (default: *.py)")
    parser.add_argument("--no-backup", action="store_true", help="Don't backup existing files")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    if not source.exists():
        print(f"Error: Source not found: {source}")
        return 1

    print(f"\n{'='*50}")
    print(f"Promoting: {source} -> {dest}")
    print(f"{'='*50}\n")

    if args.all or source.is_dir():
        source_dir = source if source.is_dir() else source.parent
        promoted = promote_directory(source_dir, dest, args.pattern)
        print(f"\nTotal promoted: {len(promoted)} files")
    else:
        promote_file(source, dest, backup=not args.no_backup)

    print(f"\n{'='*50}")
    print("Don't forget to run tests: pytest tests/")
    print(f"{'='*50}\n")
    return 0


if __name__ == "__main__":
    exit(main())
