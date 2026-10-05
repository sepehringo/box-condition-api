import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from deploy.release import Deployment


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        (self.root / "shared").mkdir()
        (self.root / "shared" / ".env").write_text(
            "DEMO_DOMAIN=api.example.com\nBOX_API_KEY=" + "a" * 32 + "\n"
        )
        self.old = self.release("1" * 40, "1" * 64)
        self.new = self.release("2" * 40, "2" * 64)
        self.deployment = Deployment(self.root, "api.example.com")
        self.deployment.point("current", self.old)
        self.events = []

    def release(self, commit, digest):
        path = self.root / "releases" / commit
        path.mkdir(parents=True)
        (path / "release.json").write_text(json.dumps({
            "commit": commit, "image": "ghcr.io/sepehringo/box-condition-api@sha256:" + digest,
        }))
        return path

    def record_command(self, command, env=None):
        self.events.append((command, env))

    def deploy(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.deployment.deploy(self.new.name, self.deployment.metadata(self.new)["image"])

    def test_success_promotes_only_after_verification_and_keeps_rollback(self):
        def run(command, env=None):
            self.record_command(command, env)
            # The current pointer remains on the good release while deploying.
            self.assertEqual(self.deployment.pointed_release("current"), self.old)
        with patch.object(self.deployment, "run", side_effect=run):
            self.deploy()
        self.assertEqual(self.deployment.pointed_release("current"), self.new)
        self.assertEqual(self.deployment.pointed_release("previous"), self.old)
        self.assertEqual(self.events[-1][0][-1], "https://api.example.com")
        self.assertTrue(all("a" * 32 not in " ".join(command) for command, _ in self.events))

    def test_pull_failure_leaves_running_release_untouched(self):
        def run(command, env=None):
            self.record_command(command, env)
            if command[-1] == "pull":
                raise subprocess.CalledProcessError(1, command)
        with patch.object(self.deployment, "run", side_effect=run), self.assertRaises(subprocess.CalledProcessError):
            self.deploy()
        self.assertEqual(self.deployment.pointed_release("current"), self.old)
        self.assertFalse(any("up" in command for command, _ in self.events))

    def test_failed_readiness_or_https_smoke_restores_previous_digest(self):
        for failure in ("readiness", "smoke"):
            self.events.clear()
            failed = False

            def run(command, env=None):
                nonlocal failed
                self.record_command(command, env)
                trigger = "up" in command if failure == "readiness" else command[0] != "docker"
                if trigger and not failed:
                    failed = True
                    raise subprocess.CalledProcessError(1, command)

            with self.subTest(failure=failure), patch.object(self.deployment, "run", side_effect=run):
                with self.assertRaises(subprocess.CalledProcessError):
                    self.deploy()
            self.assertEqual(self.deployment.pointed_release("current"), self.old)
            old_image = self.deployment.metadata(self.old)["image"]
            self.assertEqual(self.events[-1][1]["BOX_API_IMAGE"], old_image)

    def test_failed_first_release_stops_api_without_deleting_certificates(self):
        self.deployment.point("current", None)
        def run(command, env=None):
            self.record_command(command, env)
            if "up" in command:
                raise subprocess.CalledProcessError(1, command)
        with patch.object(self.deployment, "run", side_effect=run), self.assertRaises(subprocess.CalledProcessError):
            self.deploy()
        self.assertIsNone(self.deployment.pointed_release("current"))
        self.assertEqual(self.events[-1][0][-2:], ["stop", "api"])
        self.assertFalse(any("down" in command for command, _ in self.events))

    def test_external_verification_failure_can_roll_back_expected_release(self):
        with patch.object(self.deployment, "run", side_effect=self.record_command):
            self.deploy()
            with contextlib.redirect_stdout(io.StringIO()):
                self.deployment.rollback(self.new.name)
        self.assertEqual(self.deployment.pointed_release("current"), self.old)
        with self.assertRaises(ValueError):
            self.deployment.rollback(self.new.name)

    def test_invalid_digest_and_mismatched_domain_fail_before_mutation(self):
        with patch.object(self.deployment, "run") as run:
            with self.assertRaises(ValueError):
                self.deployment.deploy(self.new.name, "ghcr.io/sepehringo/box-condition-api:latest")
            run.assert_not_called()
        with self.assertRaises(ValueError):
            Deployment(self.root, "another.example.com")


if __name__ == "__main__":
    unittest.main()
