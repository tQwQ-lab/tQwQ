# -*- coding: utf-8 -*-
"""`EVALUATION_DATA_BOUNDARY_SPEC.md` §8 的九項最小驗收(全離線)。

要證明的不是「CLI 跑得完」,而是**研究程序根本沒有取得 locked OS**,而且事件
引擎與 artifacts 都沒有越過當前 segment 邊界。
"""
from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import pandas as pd

from research.contracts import BacktestRequest, CandidateSpec, EvaluationProtocol
from research.fixtures import build_fixture
from research.golden_path import run_golden_path
from evaluation.holdout import read_ledger
from research.holdout import (
    REVEAL_AUTHORIZATION,
    SEGMENT_IS,
    SEGMENT_OS,
    HoldoutBoundaryError,
    SingleHoldoutProtocol,
    freeze_candidate,
    freeze_from_is_manifest,
    research_run,
    reveal_locked_os,
)
from strategies.h3_short_reversal import H3ShortReversal

STRATEGY = "h3_short_reversal"
FIXTURE_KW = {"n_symbols": 6, "n_days": 120}


def _protocol(**over) -> SingleHoldoutProtocol:
    days = build_fixture("synthetic", **FIXTURE_KW).panel["date"].drop_duplicates()
    kw = dict(snapshot="2026-06-22", warmup_bars=20, phases=5,
              capital_scenario="research", initial_capital=1_000_000.0,
              order_size_mode="research_fractional", minimum_commission=0.0,
              mode="ratio", is_ratio=0.7, embargo_days=5)
    kw.update(over)
    return SingleHoldoutProtocol.from_dates(sorted(days), **kw)


class _Spy:
    """記錄策略實際收到的 panel(§8.1 要求證明資料沒進來,不是輸出被裁掉)。"""

    def __init__(self):
        self.seen = []
        self._real = H3ShortReversal.make_signals

    def __enter__(self):
        spy = self

        def _wrapped(inner_self, panel, params=None, context=None):
            spy.seen.append(panel[["date"]].copy())
            return spy._real(inner_self, panel, params, context)

        self._patch = mock.patch.object(H3ShortReversal, "make_signals",
                                        _wrapped)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False

    @property
    def max_date(self):
        return max(df["date"].max() for df in self.seen)

    @property
    def min_date(self):
        return min(df["date"].min() for df in self.seen)


class C1ResearchModeNeverSeesOSTest(unittest.TestCase):
    def test_strategy_input_stops_at_is_end_and_never_sees_os_rows(self):
        proto = _protocol()
        with tempfile.TemporaryDirectory() as td, _Spy() as spy:
            research_run(strategy_id=STRATEGY, protocol=proto, output_dir=td,
                         fixture_kwargs=FIXTURE_KW)
        self.assertLessEqual(spy.max_date, pd.Timestamp(proto.is_end))
        os_rows = [df for df in spy.seen
                   if (df["date"] >= pd.Timestamp(proto.os_start)).any()]
        self.assertEqual(os_rows, [], "OS 的列不得進入策略,而不只是輸出被裁掉")


class C2WarmupTest(unittest.TestCase):
    def test_legal_past_warmup_is_available(self):
        proto = _protocol()
        f = build_fixture("synthetic", window=proto.window(SEGMENT_IS),
                          **FIXTURE_KW)
        self.assertLessEqual(pd.Timestamp(f.panel["date"].min()),
                             pd.Timestamp(proto.is_start))

    def test_window_without_enough_history_fails_closed(self):
        proto = _protocol()
        with self.assertRaises(ValueError):
            build_fixture("synthetic", window=("2030-01-01", "2030-01-05"),
                          **FIXTURE_KW)


class C3ProtocolCannotBeOverriddenTest(unittest.TestCase):
    def test_engine_kwargs_cannot_override_protocol_owned_fields(self):
        for key in ("start_date", "end_date", "initial_capital",
                    "order_size_mode", "minimum_commission"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    BacktestRequest(
                        CandidateSpec(strategy_id=STRATEGY,
                                      strategy_version="1"),
                        EvaluationProtocol(), engine_kwargs={key: "x"})

    def test_unknown_segment_is_rejected(self):
        proto = _protocol()
        with self.assertRaises(HoldoutBoundaryError):
            proto.window("TRAIN")


class C4And7SegmentOutputsTest(unittest.TestCase):
    def _run(self, segment, td, proto, **kw):
        return run_golden_path(
            strategy_id=STRATEGY, fixture_name="synthetic",
            capital=proto.capital_scenario, output_dir=td, stamp=segment.lower(),
            holdout_protocol=proto, segment=segment,
            fixture_kwargs=FIXTURE_KW, **kw)

    def test_is_outputs_never_cross_is_end(self):
        proto = _protocol()
        with tempfile.TemporaryDirectory() as td:
            res = self._run(SEGMENT_IS, td, proto)
        bounds = res.audit["segment_boundary"]
        self.assertTrue(bounds["within_segment"])
        for name, top in bounds["output_max_dates"].items():
            if top is not None:
                self.assertLessEqual(pd.Timestamp(top),
                                     pd.Timestamp(proto.is_end), name)

    def test_os_outputs_never_cross_os_end_and_run_all_phases(self):
        proto = _protocol()
        with tempfile.TemporaryDirectory() as td:
            res = self._run(SEGMENT_OS, td, proto)
        bounds = res.audit["segment_boundary"]
        self.assertTrue(bounds["within_segment"])
        for name, top in bounds["output_max_dates"].items():
            if top is not None:
                self.assertLessEqual(pd.Timestamp(top),
                                     pd.Timestamp(proto.os_end), name)
        self.assertEqual(len(res.tables["phase_results"]), 5)
        self.assertIn("cum_return", res.summary["benchmark"])


class C5UnauthorizedRevealTest(unittest.TestCase):
    def test_missing_authorization_is_rejected_before_any_os_panel(self):
        proto = _protocol()
        frozen = freeze_candidate(strategy_id=STRATEGY, strategy_rule_hash="h",
                                  protocol=proto, frozen_at="2026-08-16")
        with tempfile.TemporaryDirectory() as td, _Spy() as spy:
            with self.assertRaises(HoldoutBoundaryError):
                reveal_locked_os(strategy_id=STRATEGY, protocol=proto,
                                 frozen=frozen, authorization="mode=os",
                                 output_dir=td, fixture_kwargs=FIXTURE_KW)
        self.assertEqual(spy.seen, [], "未授權時連 OS panel 都不該被建立")

    def test_unfrozen_rule_is_rejected(self):
        proto = _protocol()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(HoldoutBoundaryError):
                reveal_locked_os(strategy_id=STRATEGY, protocol=proto,
                                 frozen=None,
                                 authorization=REVEAL_AUTHORIZATION,
                                 output_dir=td, fixture_kwargs=FIXTURE_KW)

    def test_rule_hash_change_after_freeze_is_rejected(self):
        proto = _protocol()
        frozen = freeze_candidate(strategy_id=STRATEGY,
                                  strategy_rule_hash="not-the-real-hash",
                                  protocol=proto, frozen_at="2026-08-16")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(HoldoutBoundaryError):
                reveal_locked_os(strategy_id=STRATEGY, protocol=proto,
                                 frozen=frozen,
                                 authorization=REVEAL_AUTHORIZATION,
                                 output_dir=td, fixture_kwargs=FIXTURE_KW)


class C8HashSeparationTest(unittest.TestCase):
    def _req(self, **proto_over):
        cand = CandidateSpec(strategy_id=STRATEGY, strategy_version="1",
                             signal_params={"mom_window": 20})
        return BacktestRequest(cand, EvaluationProtocol(**proto_over))

    def test_protocol_changes_move_only_the_run_hash(self):
        base = self._req(data_snapshot="2026-06-22")
        for over in ({"data_snapshot": "2026-08-16"}, {"phases": 3},
                     {"benchmark": "taiex"}, {"minimum_commission": 20.0},
                     {"initial_capital": 500_000.0}):
            with self.subTest(over=over):
                kw = {"data_snapshot": "2026-06-22", **over}
                other = self._req(**kw)
                self.assertEqual(base.strategy_rule_hash(),
                                 other.strategy_rule_hash())
                self.assertNotEqual(base.evaluation_run_hash(),
                                    other.evaluation_run_hash())

    def test_split_change_changes_the_protocol_hash(self):
        self.assertNotEqual(_protocol().protocol_hash(),
                            _protocol(embargo_days=10).protocol_hash())


class C9PrefixInvarianceTest(unittest.TestCase):
    """§8.9:數個固定截點的因果契約測試 —— 不是逐日 runtime 沙盒。"""

    def test_registered_strategy_is_prefix_invariant(self):
        panel = build_fixture("synthetic", **FIXTURE_KW).panel
        strategy = H3ShortReversal()
        full = strategy.make_signals(panel)
        days = sorted(panel["date"].unique())
        for cut in (days[59], days[79], days[99]):
            with self.subTest(cut=str(cut)[:10]):
                prefix = panel[panel["date"] <= cut]
                got = strategy.make_signals(prefix)
                want = full[full["date"] <= cut].reset_index(drop=True)
                pd.testing.assert_frame_equal(got.reset_index(drop=True), want)



class C6RevealLedgerTest(unittest.TestCase):
    """§8.6:第一次 reveal 寫揭露紀錄;第二次同一段 OS 只能是 reproduction。"""

    def _is_manifest(self, proto, td):
        res = run_golden_path(
            strategy_id=STRATEGY, fixture_name="synthetic",
            capital=proto.capital_scenario, output_dir=td, stamp="probe",
            holdout_protocol=proto, segment=SEGMENT_IS,
            fixture_kwargs=FIXTURE_KW)
        return res.manifest

    def test_first_reveal_writes_ledger_second_is_previously_seen(self):
        proto = _protocol()
        with tempfile.TemporaryDirectory() as td:
            ledger = f"{td}/holdout_ledger.jsonl"
            # 改用 freeze_from_is_manifest:揭露現在要求 frozen 帶著規則本體,
            # 因為只有 hash 的 frozen 無法在載入 OS 之前重算 hash(見 C10)。
            manifest = self._is_manifest(proto, td)
            rule_hash = manifest["strategy_rule_hash"]
            frozen = freeze_from_is_manifest(manifest=manifest, protocol=proto,
                                             frozen_at="2026-08-16")

            first = reveal_locked_os(
                strategy_id=STRATEGY, protocol=proto, frozen=frozen,
                authorization=REVEAL_AUTHORIZATION, output_dir=td,
                stamp="os1", fixture_kwargs=FIXTURE_KW, ledger_path=ledger,
                now=pd.Timestamp("2026-08-16 09:00:00").to_pydatetime())
            self.assertEqual(first.audit["os_reveal"]["strategy_hash"], rule_hash)
            self.assertFalse(first.audit["os_reveal"]["holdout_previously_seen"])

            second = reveal_locked_os(
                strategy_id=STRATEGY, protocol=proto, frozen=frozen,
                authorization=REVEAL_AUTHORIZATION, output_dir=td,
                stamp="os2", fixture_kwargs=FIXTURE_KW, ledger_path=ledger,
                now=pd.Timestamp("2026-08-16 10:00:00").to_pydatetime())
            rec = second.audit["os_reveal"]
            self.assertTrue(rec["holdout_previously_seen"])
            self.assertFalse(rec["fresh_oos_claim_allowed"])

def _is_manifest(proto: SingleHoldoutProtocol, td: str) -> dict:
    """跑一次 IS 拿到真的 manifest —— 凍結要用它,不要手抄 hash。"""
    res = research_run(strategy_id=STRATEGY, protocol=proto, output_dir=td,
                       fixture_name="synthetic", fixture_kwargs=FIXTURE_KW,
                       stamp="is")
    return res.manifest


class _DataLayerSpy:
    """證明「OS 沒被載入」,而不只是「有拋例外」。

    同時攔兩個點:`build_fixture`(OS panel 有沒有被**建立**)與策略的
    `make_signals`(OS 的列有沒有進到策略)。閘門若只是把例外往前搬、資料仍然
    被讀過,`assert_untouched` 就會抓到。

    正向對照很重要:合法揭露時這個 spy **必須**錄到 fixture 呼叫,否則空 list
    只是偵測器壞了。`C11...test_a_legitimate_reveal_still_works` 負責那一半。
    """

    def __init__(self):
        self.fixture_calls = []
        self.signal_calls = []

    def __enter__(self):
        import research.golden_path as gp

        real_build = gp.build_fixture
        real_signals = H3ShortReversal.make_signals

        def spy_build(name, *a, **kw):
            window = kw.get("window")
            self.fixture_calls.append(
                (str(window[0]), str(window[1])) if window else (name, None))
            return real_build(name, *a, **kw)

        def spy_signals(inner_self, panel, *a, **kw):
            self.signal_calls.append(
                (str(panel["date"].min()), str(panel["date"].max())))
            return real_signals(inner_self, panel, *a, **kw)

        self._patches = [
            mock.patch.object(gp, "build_fixture", spy_build),
            mock.patch.object(H3ShortReversal, "make_signals", spy_signals),
        ]
        for pt in self._patches:
            pt.start()
        return self

    def __exit__(self, *exc):
        for pt in self._patches:
            pt.stop()          # 每一個都要停到,否則會洩漏到別的測試
        return False

    def assert_untouched(self, case):
        case.assertEqual(self.fixture_calls, [],
                         "被擋下來時不該建立任何 fixture(= OS 沒被載入)")
        case.assertEqual(self.signal_calls, [],
                         "被擋下來時策略不該看到任何一列")


class C10RevealGateOrderTest(unittest.TestCase):
    """閘門必須擋在**消耗之前**,不是事後宣告失敗。

    修正前的順序是 run → 比 hash → 寫紀錄,所以規則對不上時整段 locked OS
    已經被載入並算完。OS 的消耗不可逆,而那種失敗留下最糟的組合:holdout 花掉了、
    紀錄裡卻沒有那一筆,下一個讀的人以為還是 fresh。

    這些測試釘的是「**OS 沒被載入 且 揭露紀錄沒有新增一列**」。
    """

    def test_code_fingerprint_drift_is_caught_before_any_os_panel(self):
        proto = _protocol()
        with tempfile.TemporaryDirectory() as td:
            manifest = _is_manifest(proto, td)
            frozen = freeze_from_is_manifest(manifest=manifest, protocol=proto,
                                             frozen_at="2026-08-21")
            drifted = freeze_candidate(
                strategy_id=frozen.strategy_id,
                strategy_rule_hash="deadbeefdeadbeef",     # 假裝程式碼版本變了
                rules=frozen.rules, protocol=proto, frozen_at="2026-08-21")
            ledger = f"{td}/holdout_ledger.jsonl"
            with _DataLayerSpy() as spy:
                with self.assertRaises(HoldoutBoundaryError) as ctx:
                    reveal_locked_os(
                        strategy_id=STRATEGY, protocol=proto, frozen=drifted,
                        authorization=REVEAL_AUTHORIZATION, output_dir=td,
                        stamp="os", fixture_kwargs=FIXTURE_KW,
                        ledger_path=ledger)
                spy.assert_untouched(self)
            self.assertIn("strategy_rule_hash", str(ctx.exception))
            self.assertEqual(read_ledger(ledger), [],
                             "沒看過就不該有紀錄 —— 這段 OS 仍然是 fresh")

    def test_signal_params_drift_is_caught_before_any_os_panel(self):
        proto = _protocol()
        with tempfile.TemporaryDirectory() as td:
            manifest = _is_manifest(proto, td)
            frozen = freeze_from_is_manifest(manifest=manifest, protocol=proto,
                                             frozen_at="2026-08-21")
            drifted_params = {**dict(manifest["candidate"]["signal_params"]),
                              "lookback": 99}
            with _DataLayerSpy() as spy:
                with self.assertRaises(HoldoutBoundaryError):
                    reveal_locked_os(
                        strategy_id=STRATEGY, protocol=proto, frozen=frozen,
                        authorization=REVEAL_AUTHORIZATION, output_dir=td,
                        stamp="os", fixture_kwargs=FIXTURE_KW,
                        params=drifted_params,
                        ledger_path=f"{td}/holdout_ledger.jsonl")
                spy.assert_untouched(self)

    def test_frozen_without_rules_cannot_reveal(self):
        """只有 hash 的舊 frozen 會逼閘門退回「先跑再擋」→ 直接擋掉。"""
        proto = _protocol()
        with tempfile.TemporaryDirectory() as td:
            legacy = freeze_candidate(
                strategy_id=STRATEGY, strategy_rule_hash="0" * 16,
                protocol=proto, frozen_at="2026-08-21")       # 不帶 rules
            with _DataLayerSpy() as spy:
                with self.assertRaises(HoldoutBoundaryError) as ctx:
                    reveal_locked_os(
                        strategy_id=STRATEGY, protocol=proto, frozen=legacy,
                        authorization=REVEAL_AUTHORIZATION, output_dir=td,
                        stamp="os", fixture_kwargs=FIXTURE_KW,
                        ledger_path=f"{td}/holdout_ledger.jsonl")
                spy.assert_untouched(self)
            self.assertIn("freeze_from_is_manifest", str(ctx.exception))

    def test_run_failure_still_leaves_a_reveal_record(self):
        """OS 一旦要被載入就等於「看過」,紀錄不能取決於 run 之後會不會出錯。"""
        import research.golden_path as gp

        proto = _protocol()
        with tempfile.TemporaryDirectory() as td:
            manifest = _is_manifest(proto, td)
            frozen = freeze_from_is_manifest(manifest=manifest, protocol=proto,
                                             frozen_at="2026-08-21")
            ledger = f"{td}/holdout_ledger.jsonl"
            with mock.patch.object(gp, "run_golden_path",
                                   side_effect=RuntimeError("引擎在 run 中炸掉")):
                with self.assertRaises(RuntimeError):
                    reveal_locked_os(
                        strategy_id=STRATEGY, protocol=proto, frozen=frozen,
                        authorization=REVEAL_AUTHORIZATION, output_dir=td,
                        stamp="os", fixture_kwargs=FIXTURE_KW,
                        ledger_path=ledger,
                        now=pd.Timestamp("2026-08-21 00:45:00").to_pydatetime())
            rows = read_ledger(ledger)
            self.assertEqual(len(rows), 1, "run 失敗也必須留下揭露紀錄")
            self.assertEqual(rows[0]["context"]["phase"], "pre_run")


class C11UnverifiableRuleAndFalsePositiveBurnTest(unittest.TestCase):
    """把閘門前移之後才會出現的兩個洞。

    洞 1 · `signal_frame=` 是完整旁路。`SignalFrame` 只帶 strategy_id / version,
      不帶產生它的參數 → 前置閘門去算 defaults 的 hash、run 因為有 signal_frame
      也不呼叫 make_signals → 兩道 hash 閘門都放行,一套沒凍結的規則吃掉 OS,
      而紀錄記的是凍結那套的 hash。可以無限次重複。

    洞 2 · 紀錄前移造成**不可逆**的偽陽性燒毀。`fixture_name` 打錯一個字母
      (零資料載入)照樣先寫下一列 `phase=pre_run`,而 `reveal_status()` 不看
      phase → 該候選立刻 previously_seen,append-only 撤不回。
    """

    def test_signal_frame_is_refused_before_any_os_panel(self):
        proto = _protocol()
        with tempfile.TemporaryDirectory() as td:
            manifest = _is_manifest(proto, td)
            frozen = freeze_from_is_manifest(manifest=manifest, protocol=proto,
                                             frozen_at="2026-08-21")
            ledger = f"{td}/holdout_ledger.jsonl"
            frame = pd.DataFrame({"date": ["2026-01-02"], "stock_id": ["2330"]})
            with _DataLayerSpy() as spy:
                with self.assertRaises(HoldoutBoundaryError) as ctx:
                    reveal_locked_os(
                        strategy_id=STRATEGY, protocol=proto, frozen=frozen,
                        authorization=REVEAL_AUTHORIZATION, output_dir=td,
                        stamp="os", fixture_kwargs=FIXTURE_KW,
                        signal_frame=frame, ledger_path=ledger)
                spy.assert_untouched(self)
            self.assertIn("signal_frame", str(ctx.exception))
            self.assertEqual(read_ledger(ledger), [])

    def test_unknown_fixture_name_does_not_burn_the_holdout(self):
        proto = _protocol()
        with tempfile.TemporaryDirectory() as td:
            manifest = _is_manifest(proto, td)
            frozen = freeze_from_is_manifest(manifest=manifest, protocol=proto,
                                             frozen_at="2026-08-21")
            ledger = f"{td}/holdout_ledger.jsonl"
            with _DataLayerSpy() as spy:
                with self.assertRaises(HoldoutBoundaryError) as ctx:
                    reveal_locked_os(
                        strategy_id=STRATEGY, protocol=proto, frozen=frozen,
                        authorization=REVEAL_AUTHORIZATION, output_dir=td,
                        stamp="os", fixture_name="sythetic",   # 少一個 n
                        fixture_kwargs=FIXTURE_KW, ledger_path=ledger)
                spy.assert_untouched(self)
            self.assertIn("fixture_name", str(ctx.exception))
            self.assertEqual(read_ledger(ledger), [],
                             "零資料載入的失敗不可以留下 pre_run 紀錄 —— 撤不回來")

    def test_a_legitimate_reveal_still_works(self):
        """反向對照:負面測試不能把正常路徑一起擋掉,spy 也必須真的錄到東西。"""
        proto = _protocol()
        with tempfile.TemporaryDirectory() as td:
            manifest = _is_manifest(proto, td)
            frozen = freeze_from_is_manifest(manifest=manifest, protocol=proto,
                                             frozen_at="2026-08-21")
            ledger = f"{td}/holdout_ledger.jsonl"
            with _DataLayerSpy() as spy:
                res = reveal_locked_os(
                    strategy_id=STRATEGY, protocol=proto, frozen=frozen,
                    authorization=REVEAL_AUTHORIZATION, output_dir=td,
                    stamp="os", fixture_kwargs=FIXTURE_KW, ledger_path=ledger,
                    now=pd.Timestamp("2026-08-21 12:00:00").to_pydatetime())
            self.assertTrue(spy.fixture_calls,
                            "合法揭露時 spy 必須錄到 fixture —— 否則上面的空 list "
                            "只證明偵測器壞了")
            self.assertEqual(len(read_ledger(ledger)), 1)
            self.assertIn("os_reveal", res.audit)


if __name__ == "__main__":
    unittest.main()
