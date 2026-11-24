"""Publish and collect compact completed reports in dedicated Git branches."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
PREFIX = "refs/heads/team-reports"


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


def publish(round_name, name, path, *, extra=None):
    round_name, name = valid_name(round_name), valid_name(name)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("complete"):
        raise ValueError(f"Only completed reports can be published: {path}")
    remote = git(ROOT, "remote", "get-url", "origin").stdout.strip()
    branch = f"{PREFIX}/{round_name}/{name}"
    revision = git(ROOT, "rev-parse", "--short", "HEAD").stdout.strip()
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
        report_file.write_text(render(report, revision, extra or {}), encoding="utf-8")
        git(directory, "add", "--", relative.as_posix())
        if git(directory, "diff", "--cached", "--quiet", allowed=(0, 1)).returncode:
            git(directory, "-c", f"user.name={author[0]}", "-c", f"user.email={author[1]}",
                "commit", "--quiet", "-m", f"Report {round_name}/{name}")
            git(directory, "push", remote, f"HEAD:{branch}")
    print(f"Published report {round_name}/{name}", flush=True)


def pull(round_name, output):
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
            contents = git(directory, "show", f"refs/team-reports/{name}:reports/completed/{round_name}/{name}.md").stdout
            (output / f"{name}.md").write_text(contents, encoding="utf-8")
    print(f"Downloaded {len(refs)} reports to {output}", flush=True)


def sync(round_name, name, path, *, extra=None):
    # A remote outage should not discard the training or block the next shard.
    try:
        publish(round_name, name, path, extra=extra)
        pull(round_name, None)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Git reports: {error}. Local results are saved. Repeat the same command to retry.",
              file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    push = commands.add_parser("publish")
    push.add_argument("--name", required=True)
    push.add_argument("--report", type=Path, required=True)
    collect = commands.add_parser("pull")
    collect.add_argument("--output", type=Path)
    for command in (push, collect):
        command.add_argument("--round", required=True)
    args = parser.parse_args()
    if args.action == "pull":
        pull(args.round, args.output)
    else:
        publish(args.round, args.name, args.report)


if __name__ == "__main__":
    main()
