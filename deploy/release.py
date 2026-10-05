"""Deploy an immutable CI release, verify HTTPS, and restore the last good release.

Only standard-library dependencies are used on the VPS. Secrets stay in
ROOT/shared/.env; this script never prints environment contents.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

COMMIT = re.compile(r"[0-9a-f]{40}")
IMAGE = re.compile(r"ghcr\.io/sepehringo/box-condition-api@sha256:[0-9a-f]{64}")
DOMAIN = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?")


class Deployment:
    def __init__(self, root, domain=None):
        self.root = Path(root).resolve()
        self.env_file = self.root / "shared" / ".env"
        settings = {}
        for line in self.env_file.read_text().splitlines():
            tokens = shlex.split(line, comments=True)
            if not tokens:
                continue
            if len(tokens) != 1 or "=" not in tokens[0]:
                raise ValueError("Use plain KEY=value settings in shared/.env")
            name, value = tokens[0].split("=", 1)
            # Compose expands dollar expressions; accepting them here would make
            # smoke-test credentials differ from the container's credentials.
            if "$" in value:
                raise ValueError("Use literal settings without dollar expansion in shared/.env")
            settings[name] = value
        self.domain = settings.get("DEMO_DOMAIN", "")
        key = settings.get("BOX_API_KEY", "")
        if not DOMAIN.fullmatch(self.domain) or "." not in self.domain:
            raise ValueError("Set a valid DEMO_DOMAIN in shared/.env")
        if domain is not None and domain != self.domain:
            raise ValueError("GitHub and server demo hostnames must match")
        if len(key) < 32 or key.startswith("REPLACE_") or not key.isascii() or any(c.isspace() for c in key):
            raise ValueError("Set a random API key of at least 32 characters in shared/.env")
        self.env = {**os.environ, **settings}

    def run(self, command, env=None):
        subprocess.run(command, check=True, env=env or self.env)

    def release_path(self, commit):
        if not COMMIT.fullmatch(commit):
            raise ValueError("Release must be a full commit SHA")
        path = (self.root / "releases" / commit).resolve()
        if path.parent != (self.root / "releases").resolve() or not path.is_dir():
            raise ValueError("Release directory is missing or outside the release root")
        return path

    def metadata(self, path):
        data = json.loads((path / "release.json").read_text())
        if data.get("commit") != path.name or not IMAGE.fullmatch(data.get("image", "")):
            raise ValueError("Invalid release metadata")
        return data

    def pointed_release(self, name):
        pointer = self.root / name
        if not pointer.is_symlink():
            return None
        path = self.release_path(pointer.resolve().name)
        if path != pointer.resolve():
            raise ValueError("Release pointer escapes the release root")
        self.metadata(path)
        return path

    def point(self, name, path):
        pointer = self.root / name
        if path is None:
            pointer.unlink(missing_ok=True)
            return
        temporary = self.root / (name + ".next")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(path)
        temporary.replace(pointer)

    def compose(self, path):
        return ["docker", "compose", "--project-name", "box-condition-api",
                "--env-file", str(self.env_file), "-f", str(path / "compose.yaml"),
                "-f", str(path / "compose.production.yaml")]

    def environment(self, path):
        return {**self.env, "BOX_API_IMAGE": self.metadata(path)["image"]}

    def activate(self, path):
        self.run(self.compose(path) + ["up", "-d", "--no-build", "--pull", "never",
                                      "--wait", "--wait-timeout", "300"], self.environment(path))

    def verify(self, path):
        self.run([sys.executable, str(path / "scripts" / "smoke_api.py"),
                  "--url", "https://" + self.domain], self.environment(path))

    def restore(self, previous, candidate):
        if previous is None:
            # A failed first deployment has no good API to restore. Preserve
            # Caddy and its certificate volumes while stopping the failed API.
            self.run(self.compose(candidate) + ["stop", "api"], self.environment(candidate))
            self.point("current", None)
            print("First deployment failed; API stopped, configuration and certificates retained.")
        else:
            self.activate(previous)
            self.verify(previous)
            self.point("current", previous)
            print("Previous release restored and verified: " + previous.name)

    def deploy(self, commit, image):
        if not IMAGE.fullmatch(image):
            raise ValueError("Image must be the approved registry name with a sha256 digest")
        candidate = self.release_path(commit)
        previous = self.pointed_release("current")
        data = {"commit": commit, "image": image}
        metadata = candidate / "release.json"
        if metadata.exists() and json.loads(metadata.read_text()) != data:
            raise ValueError("Release already exists with a different image digest")
        temporary = candidate / "release.json.next"
        temporary.write_text(json.dumps(data) + "\n")
        temporary.replace(metadata)
        env = self.environment(candidate)
        # Pulling and validation happen before replacing any running container.
        self.run(self.compose(candidate) + ["config", "--quiet"], env)
        self.run(self.compose(candidate) + ["pull"], env)
        try:
            self.activate(candidate)
            self.verify(candidate)
        except Exception:
            print("Deployment verification failed; restoring previous release.", file=sys.stderr)
            self.restore(previous, candidate)
            raise
        if candidate != previous:
            self.point("previous", previous)
        self.point("current", candidate)
        print("Verified release " + commit + " at https://" + self.domain + "/docs")

    def rollback(self, expected=None):
        current = self.pointed_release("current")
        if current is None:
            raise ValueError("No current release to roll back")
        if expected is not None and current.name != expected:
            raise ValueError("Current release changed; refusing to roll back another deployment")
        previous = self.pointed_release("previous")
        self.restore(previous, current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/box-condition-api"))
    parser.add_argument("--release")
    parser.add_argument("--image")
    parser.add_argument("--domain")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    if not args.rollback and (not args.release or not args.image):
        parser.error("Deployment requires --release and --image")
    try:
        with (args.root / ".deploy.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            deployment = Deployment(args.root, args.domain)
            if args.rollback:
                deployment.rollback(args.release)
            else:
                deployment.deploy(args.release, args.image)
    except Exception as error:
        # No environment values or captured subprocess output are included.
        print("Release failed: " + type(error).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
