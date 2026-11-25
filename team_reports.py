"""Publish and collect compact completed reports in dedicated Git branches."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from shutil import copyfile
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
PREFIX = "refs/heads/team-reports"
MAX_CHECKPOINT_BYTES = 90 * 2**20


def valid_name(value):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", value):
        raise ValueError("Use 1–48 lowercase Latin letters, digits or hyphens for report names")
    return value


def git(directory, *command, allowed=(0,)):
    result = subprocess.run(["git", "-C", str(directory), *command],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if result.returncode not in allowed:
        raise RuntimeError(f"git {command[0]} exited {result.returncode}: {result.stderr.strip()}")
    return result


def number(value):
    return f"{value:,}".replace(",", " ") if value is not None else "—"


def render(report, revision, extra):
    settings = report.get("settings", report)
    gpu = report.get("gpu") or {}
    lines = ["# Отчёт команды Авито", "", "| Параметр | Значение |", "|---|---|"]
    fields = {
        "Коммит": revision,
        "GPU": " · ".join(filter(None, (gpu.get("name"), gpu.get("memory")))) or "—",
        "Лимит train": number(report.get("train_feed_limit")),
        "Фактически лент train": number(report.get("train_feeds")),
        "Лимит val": number(report.get("val_feed_limit")),
        "Фактически лент val": number(report.get("val_feeds")),
        "Смещение train": number(report.get("train_offset", 0)),
        "Итерации / эпохи": f"{settings.get('iterations', '—')} / {settings.get('epochs', '—')}",
        **extra,
    }
    lines.extend(f"| {name} | {value} |" for name, value in fields.items())
    lines.extend(["", "| Модель | Полная NDCG · ранняя val | Полная NDCG · поздняя val | Полная NDCG · вся val | Время, с | Пик RAM, МиБ |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in report["models"]:
        scores = " | ".join(f"{row[part]['ndcg']:.8f}" for part in ("tuning", "holdout", "val"))
        lines.append(f"| `{row['model']}` | {scores} | {number(row.get('seconds'))} | {number(row.get('peak_ram_mib'))} |")
    return "\n".join(lines) + "\n"


def checkpoint_path(value):
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or any(
            not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*", part) for part in path.parts)
            or (path.suffix not in (".cbm", ".pt", ".npz", ".npy") and path.name != "model.txt")):
        raise ValueError(f"Expected a relative model checkpoint path: {value}")
    return path


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_reproduction(destination, report_path, checkpoints, instructions, revision):
    source = report_path.resolve().parent
    files = {}
    for value in checkpoints:
        relative = checkpoint_path(value)
        if relative.as_posix() in files:
            raise ValueError(f"Repeated checkpoint: {value}")
        src = (source / relative).resolve()
        if not src.is_relative_to(source) or not src.is_file():
            raise ValueError(f"Checkpoint must exist under the report directory: {value}")
        if not 0 < src.stat().st_size <= MAX_CHECKPOINT_BYTES:
            raise ValueError(f"Checkpoint must be nonempty and at most 90 MiB: {value}")
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        copyfile(src, dst)
        files[relative.as_posix()] = {"bytes": dst.stat().st_size, "sha256": checksum(dst)}
    destination.mkdir(parents=True, exist_ok=True)
    copyfile(instructions, destination / "REPRODUCE.md")
    (destination / "manifest.json").write_text(json.dumps(
        {"code_revision": revision, "checkpoints": files}, indent=2) + "\n", encoding="utf-8")


def publish(round_name, name, path, *, extra=None, checkpoints=(), instructions=None, revision="HEAD"):
    round_name, name = valid_name(round_name), valid_name(name)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("complete"):
        raise ValueError(f"Only completed reports can be published: {path}")
    remote = git(ROOT, "remote", "get-url", "origin").stdout.strip()
    branch = f"{PREFIX}/{round_name}/{name}"
    if checkpoints and instructions is None:
        raise ValueError("Provide reproduction commands with --reproduce when publishing checkpoints")
    revision = git(ROOT, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}").stdout.strip()
    author = [git(ROOT, "config", "--get", f"user.{key}", allowed=(0, 1)).stdout.strip() for key in ("name", "email")]
    if not all(author):
        raise ValueError("Set git user.name and user.email before publishing reports")
    existing = git(ROOT, "ls-remote", "--exit-code", "--heads", "origin", branch, allowed=(0, 2))
    base = branch if existing.returncode == 0 else "refs/heads/main"
    with TemporaryDirectory(prefix="avito-report-") as folder:
        directory = Path(folder)
        git(directory, "init", "--quiet")
        git(directory, "fetch", "--quiet", "--depth", "1", remote, base)
        git(directory, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        relative = Path("reports") / "completed" / round_name / f"{name}.md"
        report_file = directory / relative
        report_file.parent.mkdir(parents=True, exist_ok=True)
        bundle = relative.with_suffix("")
        if instructions is not None:
            with TemporaryDirectory(prefix="avito-reproduce-", dir=directory) as staging:
                prepared = Path(staging) / "bundle"
                write_reproduction(prepared, path, checkpoints, instructions, revision)
                git(directory, "rm", "-q", "-r", "--ignore-unmatch", "--", bundle.as_posix())
                prepared.rename(directory / bundle)
            git(directory, "add", "-f", "--", bundle.as_posix())
        markdown = render(report, revision[:7], extra or {})
        if (directory / bundle / "manifest.json").is_file():
            markdown += f"\n[Команды воспроизведения]({name}/REPRODUCE.md) · [Коммит и контрольные суммы]({name}/manifest.json)\n"
        report_file.write_text(markdown, encoding="utf-8")
        git(directory, "add", "--", relative.as_posix())
        if git(directory, "diff", "--cached", "--quiet", allowed=(0, 1)).returncode:
            git(directory, "-c", f"user.name={author[0]}", "-c", f"user.email={author[1]}",
                "commit", "--quiet", "-m", f"Report {round_name}/{name}")
            git(directory, "push", remote, f"HEAD:{branch}")
    print(f"Published report {round_name}/{name}", flush=True)


def pull_bundle(directory, ref, bundle, output):
    manifest_path = bundle / "manifest.json"
    manifest_object = f"{ref}:{manifest_path.as_posix()}"
    if git(directory, "cat-file", "-e", manifest_object, allowed=(0, 128)).returncode:
        return
    manifest_text = git(directory, "show", manifest_object).stdout
    manifest = json.loads(manifest_text)
    instructions = git(directory, "show", f"{ref}:{bundle.as_posix()}/REPRODUCE.md").stdout
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as staging:
        pending = Path(staging)
        for value, entry in manifest["checkpoints"].items():
            path = checkpoint_path(value)
            target = pending / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as stream:
                subprocess.run(["git", "-C", str(directory), "show", f"{ref}:{(bundle / path).as_posix()}"],
                               stdout=stream, check=True, timeout=60)
            if target.stat().st_size != entry["bytes"] or checksum(target) != entry["sha256"]:
                raise ValueError(f"Checkpoint verification failed: {value}")
        for value in manifest["checkpoints"]:
            path = checkpoint_path(value)
            target = output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            (pending / path).replace(target)
    (output / "REPRODUCE.md").write_text(instructions, encoding="utf-8")
    (output / "manifest.json").write_text(manifest_text, encoding="utf-8")


def pull(round_name, output, *, with_artifacts=False):
    round_name = valid_name(round_name)
    output = output or ROOT / "runs" / "team-reports" / round_name
    remote = git(ROOT, "remote", "get-url", "origin").stdout.strip()
    output.mkdir(parents=True, exist_ok=True)
    available = git(ROOT, "ls-remote", "--exit-code", "--heads", "origin", f"{PREFIX}/{round_name}/*", allowed=(0, 2))
    if available.returncode == 2:
        print(f"No reports published for {round_name}", flush=True)
        return
    with TemporaryDirectory(prefix="avito-reports-") as folder:
        directory = Path(folder)
        git(directory, "init", "--quiet")
        git(directory, "fetch", "--quiet", "--depth", "1", remote,
            f"+{PREFIX}/{round_name}/*:refs/team-reports/*")
        refs = git(directory, "for-each-ref", "--format=%(refname:strip=2)", "refs/team-reports").stdout.splitlines()
        for name in map(valid_name, refs):
            ref = f"refs/team-reports/{name}"
            bundle = PurePosixPath("reports", "completed", round_name, name)
            contents = git(directory, "show", f"{ref}:{bundle.as_posix()}.md").stdout
            (output / f"{name}.md").write_text(contents, encoding="utf-8")
            if with_artifacts:
                pull_bundle(directory, ref, bundle, output / name)
    print(f"Downloaded {len(refs)} reports to {output}", flush=True)


def sync(round_name, name, path, *, extra=None, checkpoints=(), instructions=None, revision="HEAD"):
    # A remote outage should not discard the training or block the next shard.
    try:
        publish(round_name, name, path, extra=extra, checkpoints=checkpoints, instructions=instructions,
                revision=revision)
        pull(round_name, None)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Git reports: {error}. Local results are saved. Use team_reports.py publish or pull to retry.",
              file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    push = commands.add_parser("publish")
    push.add_argument("--name", required=True)
    push.add_argument("--report", type=Path, required=True)
    push.add_argument("--checkpoint", action="append", default=[], help="Model file relative to the report directory")
    push.add_argument("--reproduce", type=Path, help="Markdown with the exact data, commands and blend recipe")
    push.add_argument("--revision", default="HEAD", help="Commit used to train the models")
    collect = commands.add_parser("pull")
    collect.add_argument("--output", type=Path)
    collect.add_argument("--with-artifacts", action="store_true", help="Extract and verify reproduction bundles")
    for command in (push, collect):
        command.add_argument("--round", required=True)
    args = parser.parse_args()
    if args.action == "pull":
        pull(args.round, args.output, with_artifacts=args.with_artifacts)
    else:
        publish(args.round, args.name, args.report, checkpoints=args.checkpoint, instructions=args.reproduce,
                revision=args.revision)


if __name__ == "__main__":
    main()
