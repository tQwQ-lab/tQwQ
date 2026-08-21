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


def _exact_pin(requirement: Requirement) -> str:
    """Return an exact pinned version or reject the requirement."""
    specs = list(requirement.specifier)
    if (
        len(specs) != 1
        or specs[0].operator != "=="
        or "*" in specs[0].version
    ):
        raise AssertionError(f"CI dependency must be exactly pinned: {requirement}")
    return specs[0].version


def _validate_dependency_contract(
    app: list[Requirement], ci_requirements: list[Requirement]
) -> None:
    """Reject missing, duplicate, unpinned, or incompatible CI dependencies."""
    ci = {}
    for requirement in ci_requirements:
        name = canonicalize_name(requirement.name)
        if name in ci:
            raise AssertionError(f"Duplicate CI dependency: {name}")
        ci[name] = requirement

    for requirement in app:
        name = canonicalize_name(requirement.name)
        if name not in ci:
            raise AssertionError(f"requirements-ci.txt is missing {name}")
        pin = _exact_pin(ci[name])
        if not requirement.specifier.contains(pin, prereleases=True):
            raise AssertionError(
                f"CI pin {ci[name]} is incompatible with {requirement}"
            )


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
        self.assertRegex(setup_python, r"(?m)^          cache:\s*pip\s*$")
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
                _exact_pin(requirement)

    def test_ci_lock_covers_compatible_application_dependencies(self):
        """Every direct application dependency must have a compatible CI pin."""
        _validate_dependency_contract(
            _requirements(APP_REQUIREMENTS), _requirements(CI_REQUIREMENTS)
        )

    def test_dependency_contract_rejects_unpinned_ci_dependency(self):
        with self.assertRaisesRegex(AssertionError, "exactly pinned"):
            _validate_dependency_contract(
                [Requirement("numpy>=2.0")], [Requirement("numpy>=2.4")]
            )

    def test_dependency_contract_rejects_wildcard_ci_pin(self):
        with self.assertRaisesRegex(AssertionError, "exactly pinned"):
            _validate_dependency_contract(
                [Requirement("packaging>=26")], [Requirement("packaging==26.*")]
            )

    def test_dependency_contract_rejects_missing_ci_dependency(self):
        with self.assertRaisesRegex(AssertionError, "missing numpy"):
            _validate_dependency_contract([Requirement("numpy>=2.0")], [])

    def test_dependency_contract_rejects_duplicate_ci_dependency(self):
        with self.assertRaisesRegex(AssertionError, "Duplicate CI dependency"):
            _validate_dependency_contract(
                [Requirement("numpy>=2.0")],
                [Requirement("numpy==2.4.6"), Requirement("NumPy==2.4.6")],
            )

    def test_dependency_contract_rejects_incompatible_ci_pin(self):
        with self.assertRaisesRegex(AssertionError, "incompatible"):
            _validate_dependency_contract(
                [Requirement("numpy>=2.0")], [Requirement("numpy==1.26.4")]
            )


if __name__ == "__main__":
    unittest.main()
