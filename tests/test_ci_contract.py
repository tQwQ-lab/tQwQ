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


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CI_REQUIREMENTS = ROOT / "requirements-ci.txt"


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

    def test_checkout_does_not_persist_write_credentials(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("persist-credentials: false", text)

    def test_ci_uses_bounded_runtime_and_locked_dependencies(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertRegex(text, r"timeout-minutes:\s*[1-9][0-9]*")
        self.assertIn("cache-dependency-path: requirements-ci.txt", text)
        self.assertIn("--requirement requirements-ci.txt", text)
        self.assertIn("python -m pip check", text)


class CIDependencyContractTest(unittest.TestCase):
    def test_ci_dependencies_are_exactly_pinned(self):
        entries = []
        for raw in CI_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                entries.append(line)
        self.assertTrue(entries)
        for entry in entries:
            with self.subTest(entry=entry):
                self.assertRegex(
                    entry,
                    r"^[A-Za-z0-9_.-]+==[^=<>!~\s]+$",
                    "CI dependency 必須使用 == 固定版本",
                )


if __name__ == "__main__":
    unittest.main()
