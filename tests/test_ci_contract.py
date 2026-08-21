# -*- coding: utf-8 -*-
"""檢查 CI 使用的 Action 與 Python 套件是否固定版本。

若這些外部依賴未固定，同一個 commit 在不同時間可能使用不同版本，
導致原本通過的 CI 日後失敗。發生問題時，我們也難以判斷原因來自
專案程式碼，還是外部依賴更新。

因此，本測試要求 CI 固定外部依賴版本，讓執行環境可以重現，測試
結果也比較容易追查。
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
APP_REQUIREMENTS = ROOT / "requirements.txt"
CI_REQUIREMENTS = ROOT / "requirements-ci.txt"


def _job_block(text: str, name: str) -> str:
    """Return one top-level job block from a GitHub Actions workflow."""
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        text,
    )
    if match is None:
        raise AssertionError(f"CI workflow 缺少 {name!r} job")
    return match.group(0)


def _step_block(job: str, marker: str) -> str:
    """Return the workflow step whose first line contains ``marker``."""
    steps = re.findall(r"(?ms)^      - .+?(?=^      - |\Z)", job)
    matches = [step for step in steps if marker in step.splitlines()[0]]
    if len(matches) != 1:
        raise AssertionError(
            f"CI workflow 應恰有一個 {marker!r} step，實際找到 {len(matches)} 個"
        )
    return matches[0]


def _requirements(path: Path) -> list[Requirement]:
    """Parse non-comment requirement lines from ``path``."""
    parsed = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            parsed.append(Requirement(line))
    return parsed


class CIWorkflowContractTest(unittest.TestCase):
    def test_third_party_actions_are_pinned_to_full_commit_sha(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        uses = re.findall(r"^\s*- uses:\s*([^\s#]+)", text, flags=re.MULTILINE)
        self.assertTrue(uses, "CI workflow 至少要使用 checkout / setup-python")
        for action in uses:
            with self.subTest(action=action):
                self.assertRegex(
                    action,
                    r"^[^@]+@[0-9a-f]{40}$",
                    "第三方 Action 必須釘完整 commit SHA，不能只寫可移動的版本標籤",
                )

    def test_ci_settings_are_bound_to_the_steps_they_govern(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        job = _job_block(text, "offline-checks")
        checkout = _step_block(job, "uses: actions/checkout@")
        setup_python = _step_block(job, "uses: actions/setup-python@")
        install = _step_block(job, "name: Install dependencies")

        self.assertRegex(job, r"(?m)^    timeout-minutes:\s*[1-9][0-9]*\s*$")
        self.assertRegex(
            checkout, r"(?m)^          persist-credentials:\s*false\s*$"
        )
        self.assertRegex(
            setup_python, r'(?m)^          python-version:\s*"3\.11"\s*$'
        )
        self.assertRegex(
            setup_python,
            r"(?m)^          cache-dependency-path:\s*requirements-ci\.txt\s*$",
        )
        self.assertIn("python -m pip install --requirement requirements-ci.txt", install)
        self.assertIn("python -m pip check", install)


class CIDependencyContractTest(unittest.TestCase):
    def test_ci_dependencies_are_exactly_pinned(self):
        requirements = _requirements(CI_REQUIREMENTS)
        self.assertTrue(requirements)
        for requirement in requirements:
            with self.subTest(requirement=str(requirement)):
                specs = list(requirement.specifier)
                self.assertEqual(len(specs), 1)
                self.assertEqual(
                    specs[0].operator,
                    "==",
                    "CI dependency 必須使用 == 固定版本",
                )

    def test_ci_lock_covers_compatible_application_dependencies(self):
        """Every direct application dependency must have a compatible CI pin."""
        app = _requirements(APP_REQUIREMENTS)
        ci = {
            canonicalize_name(requirement.name): requirement
            for requirement in _requirements(CI_REQUIREMENTS)
        }

        for requirement in app:
            name = canonicalize_name(requirement.name)
            with self.subTest(requirement=str(requirement)):
                self.assertIn(name, ci, f"requirements-ci.txt 缺少 {name}")
                pin_specs = list(ci[name].specifier)
                self.assertEqual(len(pin_specs), 1)
                self.assertEqual(pin_specs[0].operator, "==")
                self.assertTrue(
                    requirement.specifier.contains(
                        pin_specs[0].version, prereleases=True
                    ),
                    f"CI pin {ci[name]} 不符合應用程式需求 {requirement}",
                )


if __name__ == "__main__":
    unittest.main()
