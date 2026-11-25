"""Exchange completed shards through Git using authenticated, encrypted archives."""

from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
from shutil import copyfileobj
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile

from team_reports import ROOT, git, valid_name


KEY_NAME = ".parts-key"
ARCHIVE = "part.zip.aes"
LIMIT = 95 * 2**20


def init_key(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as key:
        key.write(secrets.token_bytes(32))


def bundle_paths(model):
    directory = f"fit/{valid_name(model)}"
    return ("part.json", "fit/report.json", f"{directory}/val_predictions.parquet",
            f"{directory}/test_predictions.parquet")


def key_path(args):
    return args.parts_key or args.output / KEY_NAME


class Exchange:
    def __init__(self, output, path, round_name, model, *, remote=None):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        key = path.read_bytes()
        if len(key) != 32:
            raise ValueError(f"Expected the shared 32-byte key file at {path}")
        self.cipher = AESGCM(key)
        self.root = output / "parts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = bundle_paths(model)
        self.remote = remote or git(ROOT, "remote", "get-url", "origin").stdout.strip()
        self.refs = f"refs/heads/team-bundles/{valid_name(round_name)}/{sha256(key).hexdigest()[:16]}/"

    def ref(self, index):
        return f"{self.refs}part-{index:03d}"

    def directory(self, index):
        return self.root / f"part-{index:03d}"

    def ready(self, index):
        directory = self.directory(index)
        return (all((directory / name).is_file() for name in self.paths)
                and json.loads((directory / "fit/report.json").read_text(encoding="utf-8")).get("complete"))

    def branch_ids(self):
        lines = self.git(ROOT, "ls-remote", "--refs", "--heads", self.remote, f"{self.refs}part-*").splitlines()
        return {int(ref.removeprefix(self.refs + "part-")): oid for oid, ref in map(str.split, lines)
                if ref.startswith(self.refs + "part-") and ref.removeprefix(self.refs + "part-").isdigit()}

    @staticmethod
    def git(cwd, *command, stdin=None, stdout=None):
        result = subprocess.run(["git", "-C", str(cwd), *command], input=stdin,
                                stdout=stdout if stdout is not None else subprocess.PIPE, stderr=subprocess.PIPE,
                                env=dict(os.environ, GIT_TERMINAL_PROMPT="0"), timeout=240)
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"git {command[0]} failed: {detail}")
        return result.stdout.decode("utf-8", errors="replace").strip() if stdout is None else ""

    def upload(self, index):
        directory = self.directory(index)
        if not self.ready(index):
            raise ValueError(f"Part {index} is incomplete")
        if sum((directory / name).stat().st_size for name in self.paths) >= LIMIT:
            raise ValueError(f"Part {index} is too large for the GitHub shard transport")
        archive = BytesIO()
        with ZipFile(archive, "w", compression=ZIP_STORED) as package:
            for name in self.paths:
                package.write(directory / name, name)
        nonce = secrets.token_bytes(12)
        with TemporaryDirectory(dir=self.root.parent, prefix=".part-push-") as temp:
            repo = Path(temp)
            self.git(repo, "init", "--bare", "--quiet")
            owner = [git(ROOT, "config", "--get", f"user.{field}").stdout.strip() for field in ("name", "email")]
            encrypted = nonce + self.cipher.encrypt(nonce, archive.getvalue(), self.ref(index).encode())
            if len(encrypted) > LIMIT:
                raise ValueError(f"Encrypted part {index} is too large")
            blob = self.git(repo, "hash-object", "-w", "--stdin", stdin=encrypted)
            tree = self.git(repo, "mktree", stdin=f"100644 blob {blob}\t{ARCHIVE}\n".encode())
            commit = self.git(repo, "-c", f"user.name={owner[0]}", "-c", f"user.email={owner[1]}", "commit-tree", tree,
                              "-m", f"Encrypted shard {index:03d}")
            self.git(repo, "push", self.remote, f"{commit}:{self.ref(index)}")
        print(f"Uploaded encrypted part {index:03d}", flush=True)
        return commit

    def download(self, index, expected_commit):
        folder = self.directory(index)
        if folder.exists():
            raise ValueError(f"Existing local folder {folder} needs attention before replacing it")
        with TemporaryDirectory(dir=self.root, prefix=".part-pull-") as temp:
            repo = Path(temp)
            self.git(repo, "init", "--bare", "--quiet")
            self.git(repo, "fetch", "--no-tags", "--depth=1", self.remote, self.ref(index))
            if self.git(repo, "rev-parse", "FETCH_HEAD") != expected_commit:
                raise ValueError(f"Remote part {index} changed during download")
            entry = f"FETCH_HEAD:{ARCHIVE}"
            if not 28 <= int(self.git(repo, "cat-file", "-s", entry)) <= LIMIT:
                raise ValueError(f"Unexpected archive size for part {index}")
            with (repo / ARCHIVE).open("wb") as stream:
                self.git(repo, "cat-file", "blob", entry, stdout=stream)
            encrypted = (repo / ARCHIVE).read_bytes()
            decoded = self.cipher.decrypt(encrypted[:12], encrypted[12:], self.ref(index).encode())
            staging = repo / "payload"
            with ZipFile(BytesIO(decoded)) as package:
                entries = package.infolist()
                if (sorted(info.filename for info in entries) != sorted(self.paths)
                        or sum(info.file_size for info in entries) > LIMIT):
                    raise ValueError(f"Unexpected files in part {index}")
                for name in self.paths:
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(name) as source, target.open("xb") as output:
                        copyfileobj(source, output)
            report = json.loads((staging / "fit/report.json").read_text(encoding="utf-8"))
            manifest = json.loads((staging / "part.json").read_text(encoding="utf-8"))
            if manifest["part"]["index"] != index or not report["complete"]:
                raise ValueError(f"Downloaded part {index} did not match its manifest")
            os.rename(staging, folder)
        print(f"Downloaded and verified part {index:03d}", flush=True)

    def sync(self, parts, owned, *, collect):
        published = self.branch_ids()
        for index in sorted(owned):
            if index not in published and self.ready(index):
                published[index] = self.upload(index)
        if collect:
            for index in sorted((parts - owned) & published.keys()):
                if not self.ready(index):
                    self.download(index, published[index])
        return published
