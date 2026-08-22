# -*- coding: utf-8 -*-
"""單次 IS / embargo / locked-OS 的資料閘門(見 `research/docs/` 的兩份規格)。

這一層要保證的不是「CLI 跑得完」,而是:

  **研究程序根本沒有取得 locked OS**,而且事件引擎與 artifacts 都沒有越過
  當前 segment 的邊界。

三個入口刻意分開,因為它們的授權層級不同:

  `research_run()`     研究:只能建立並傳入 `[warmup_start, is_end]`
  `freeze_candidate()` 凍結:把 strategy rule 固定成一個 hash
  `reveal_locked_os()` 揭露:需要**獨立的 owner 授權**,一般 `mode="os"` 不等於授權

誠實聲明(對齊 `EVALUATION_DATA_BOUNDARY_SPEC.md` §2.2):這是**程序性**閘門,
不是物理沙盒。它擋的是「IS 研究流程偷看 locked OS」,不是「任意 Python 在 IS 內
寫 `shift(-1)`」——後者靠因果算子、prefix-invariance 測試與 code review。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from evaluation.holdout import record_reveal, reveal_status
from evaluation.splits import EvaluationSplit, build_evaluation_split

SEGMENT_IS = "IS"
SEGMENT_OS = "OS"

# owner 的獨立授權字串。刻意不是 bool,也刻意不叫 mode —— 規格 §4.4:
# 「一般 run(mode="os") 不得等價於授權」。要打錯字很難、要不小心傳到更難。
REVEAL_AUTHORIZATION = "owner-authorized-single-holdout-reveal"


class HoldoutBoundaryError(RuntimeError):
    """違反 single-holdout 資料邊界;一律 fail-closed,不得降級成 warning。"""


def _stable_hash(payload: Mapping[str, Any], *, length: int = 16) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str,
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


@dataclass(frozen=True)
class SingleHoldoutProtocol:
    """固定的 IS / embargo / OS 邊界與評估協定(三種 freeze 的第二種)。

    這些欄位**不得**由 strategy params 或 engine_kwargs 覆寫(規格 §2.1 末段):
    能改考卷的人不能同時是考生。
    """

    snapshot: str
    is_start: str
    is_end: str
    os_start: str
    os_end: str
    embargo_trading_days: int
    warmup_bars: int
    phases: int
    benchmark: str
    capital_scenario: str
    initial_capital: float
    order_size_mode: str
    minimum_commission: float
    segment_end_policy: str = "mtm_at_segment_end_no_next_segment_price"
    split_mode: str = ""
    evaluator_version: str = "single_holdout_v1"

    @classmethod
    def from_dates(cls, dates: Sequence, *, snapshot: str, warmup_bars: int,
                   phases: int, capital_scenario: str, initial_capital: float,
                   order_size_mode: str, minimum_commission: float,
                   benchmark: str = "daily_equal_weight_rebalanced_eligible",
                   minimum_embargo_days: int = 0,
                   **split_kwargs) -> "SingleHoldoutProtocol":
        """用 `evaluation/splits.py` 建切割 —— **不另寫 split**(goal 明文要求)。"""
        split: EvaluationSplit = build_evaluation_split(
            dates, minimum_embargo_days=minimum_embargo_days, **split_kwargs)
        return cls(
            snapshot=str(snapshot),
            is_start=str(pd.Timestamp(split.is_start).date()),
            is_end=str(pd.Timestamp(split.is_end).date()),
            os_start=str(pd.Timestamp(split.os_start).date()),
            os_end=str(pd.Timestamp(split.os_end).date()),
            embargo_trading_days=int(split.n_embargo),
            warmup_bars=int(warmup_bars), phases=int(phases),
            benchmark=benchmark, capital_scenario=capital_scenario,
            initial_capital=float(initial_capital),
            order_size_mode=str(order_size_mode),
            minimum_commission=float(minimum_commission),
            split_mode=str(split.mode),
        )

    def __post_init__(self) -> None:
        order = [self.is_start, self.is_end, self.os_start, self.os_end]
        if any(pd.Timestamp(a) > pd.Timestamp(b)
               for a, b in zip(order, order[1:])):
            raise HoldoutBoundaryError(
                f"[fail-closed] 切割日期順序不合法:{order}")
        if int(self.embargo_trading_days) < 0:
            raise HoldoutBoundaryError("embargo 不得為負")
        if int(self.phases) < 1:
            raise HoldoutBoundaryError("phases 至少為 1")

    def protocol_hash(self) -> str:
        return _stable_hash(asdict(self))

    def window(self, segment: str) -> Tuple[str, str]:
        """該 segment 允許**載入**的資料窗(含因果 warmup)。"""
        if segment == SEGMENT_IS:
            start = (pd.Timestamp(self.is_start)
                     - pd.tseries.offsets.BDay(int(self.warmup_bars) + 5))
            return (str(start.date()), self.is_end)
        if segment == SEGMENT_OS:
            start = (pd.Timestamp(self.os_start)
                     - pd.tseries.offsets.BDay(int(self.warmup_bars) + 5))
            return (str(start.date()), self.os_end)
        raise HoldoutBoundaryError(
            f"[fail-closed] 未知 segment={segment!r};只接受 {SEGMENT_IS}/{SEGMENT_OS}")

    def scoring_window(self, segment: str) -> Tuple[str, str]:
        """該 segment 允許**計分**的窗(比載入窗窄:warmup 不計分)。"""
        if segment == SEGMENT_IS:
            return (self.is_start, self.is_end)
        if segment == SEGMENT_OS:
            return (self.os_start, self.os_end)
        raise HoldoutBoundaryError(f"[fail-closed] 未知 segment={segment!r}")

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["protocol_hash"] = self.protocol_hash()
        return out


# ── segment 邊界稽核 ──────────────────────────────────────────────────────
_DATE_COLUMNS = ("date", "exit_date", "entry_date", "signal_date")


def _frame_max_date(frame) -> Optional[pd.Timestamp]:
    if frame is None or len(frame) == 0:
        return None
    df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    best: Optional[pd.Timestamp] = None
    for col in _DATE_COLUMNS:
        if col not in df.columns:
            continue
        vals = pd.to_datetime(df[col], errors="coerce").dropna()
        if len(vals):
            top = vals.max()
            best = top if best is None else max(best, top)
    return best


def boundary_audit(*, protocol: SingleHoldoutProtocol, segment: str,
                   panel_input: Tuple[Any, Any],
                   tables: Mapping[str, Any]) -> Dict[str, Any]:
    """記錄策略實際 input 範圍與每一張輸出表的最大日期,並檢查沒有越界。

    規格 §6 要求「artifact 必須記錄 strategy 實際 input 日期範圍與所有事件輸出
    的最大日期」—— 因為「輸出看起來沒越界」與「資料根本沒進來」是兩件事,
    只有前者的話,任何裁切 bug 都會偽裝成合規。
    """
    load_start, load_end = protocol.window(segment)
    score_start, score_end = protocol.scoring_window(segment)
    limit = pd.Timestamp(score_end)

    per_table: Dict[str, Optional[str]] = {}
    violations: List[str] = []
    for name, frame in tables.items():
        top = _frame_max_date(frame)
        per_table[name] = None if top is None else str(top.date())
        if top is not None and top > limit:
            violations.append(
                f"{name} 的最大日期 {top.date()} 越過 segment 結尾 {limit.date()}")

    in_min, in_max = panel_input
    if in_max is not None and pd.Timestamp(in_max) > pd.Timestamp(load_end):
        violations.append(
            f"strategy input 最大日期 {pd.Timestamp(in_max).date()} 越過"
            f"允許載入窗 {load_end}")
    if in_min is not None and pd.Timestamp(in_min) < pd.Timestamp(load_start):
        violations.append(
            f"strategy input 最小日期 {pd.Timestamp(in_min).date()} 早於"
            f"允許載入窗 {load_start}")

    return {
        "segment": segment,
        "load_window": [load_start, load_end],
        "scoring_window": [score_start, score_end],
        "strategy_input_min": (None if in_min is None
                               else str(pd.Timestamp(in_min).date())),
        "strategy_input_max": (None if in_max is None
                               else str(pd.Timestamp(in_max).date())),
        "output_max_dates": per_table,
        "violations": violations,
        "within_segment": not violations,
    }


def assert_within_segment(audit: Mapping[str, Any]) -> None:
    if not audit.get("within_segment"):
        raise HoldoutBoundaryError(
            "[fail-closed] 輸出越過 segment 邊界:\n  - "
            + "\n  - ".join(audit.get("violations") or []))


# ── OS 揭露授權 ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FrozenCandidate:
    """凍結的策略規則(三種 freeze 的第三種)。"""

    strategy_id: str
    strategy_rule_hash: str
    frozen_at: str
    protocol_hash: str
    manifest_path: str = ""
    #: IS manifest 的 `candidate` 區塊。**只有 hash 的 frozen 無法在載入 OS 之前
    #: 重算 hash**,會逼閘門退回「先跑再擋」—— 而那正是燒掉 holdout 的原因。
    rules: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def authorize_reveal(token: str) -> None:
    """檢查 owner 的獨立授權。`mode="os"` 這種一般參數**不等於**授權。"""
    if token != REVEAL_AUTHORIZATION:
        raise HoldoutBoundaryError(
            "[fail-closed] locked OS 需要 owner 的獨立授權。"
            f"預期 authorization={REVEAL_AUTHORIZATION!r};"
            "一般的 mode/segment 參數不構成授權(規格 §4.4)")


def assert_rule_unchanged(frozen: FrozenCandidate, current_hash: str) -> None:
    if str(frozen.strategy_rule_hash) != str(current_hash):
        raise HoldoutBoundaryError(
            "[fail-closed] OS run 的 strategy_rule_hash 與凍結時不同 "
            f"({current_hash} != {frozen.strategy_rule_hash})。"
            "看完 OS 回頭改規則正是這條閘門要擋的事(規格 §7)")


def os_reveal_status(*, strategy_rule_hash: str, protocol: SingleHoldoutProtocol,
                     strategy_id: str = "") -> Dict[str, Any]:
    return reveal_status(strategy_hash=strategy_rule_hash,
                         strategy_name=strategy_id or None,
                         os_start=protocol.os_start, os_end=protocol.os_end)


def record_os_reveal(*, strategy_rule_hash: str, strategy_id: str,
                     protocol: SingleHoldoutProtocol, source: str,
                     manifest: Optional[str] = None,
                     extra_context: Optional[Mapping[str, Any]] = None,
                     now=None, path=None) -> Dict[str, Any]:
    """把這次揭露 append 進 append-only 揭露紀錄(第二次會被標 previously_seen)。"""
    return record_reveal(
        strategy_hash=strategy_rule_hash, strategy_name=strategy_id,
        os_start=protocol.os_start, os_end=protocol.os_end,
        source=source, segment=SEGMENT_OS, manifest=manifest,
        is_window=[protocol.is_start, protocol.is_end],
        embargo_trading_days=int(protocol.embargo_trading_days),
        split_mode=protocol.split_mode,
        context={"protocol_hash": protocol.protocol_hash(),
                 "evaluator_version": protocol.evaluator_version,
                 **dict(extra_context or {})},
        now=now, path=path)


# ── 兩個分開的入口(授權層級不同)────────────────────────────────────────
def research_run(*, strategy_id: str, protocol: SingleHoldoutProtocol,
                 output_dir, fixture_name: str = "synthetic",
                 stamp: str = "is", **kwargs):
    """研究入口:**只能**跑 IS。這裡沒有任何參數可以要到 OS 資料。

    OS 之所以在 research mode 建不出來,不是因為被檢查擋掉,而是因為
    `protocol.window(SEGMENT_IS)` 根本不會回傳 OS 的日期 —— 資料窗在建 panel
    之前就被決定了,`reveal_locked_os()` 是唯一會傳 `SEGMENT_OS` 的地方。
    """
    from research.golden_path import run_golden_path

    return run_golden_path(
        strategy_id=strategy_id, fixture_name=fixture_name,
        capital=protocol.capital_scenario, output_dir=output_dir,
        stamp=stamp, holdout_protocol=protocol, segment=SEGMENT_IS, **kwargs)


def freeze_candidate(*, strategy_id: str, strategy_rule_hash: str,
                     protocol: SingleHoldoutProtocol, frozen_at: str,
                     manifest_path: str = "",
                     rules: Optional[Mapping[str, Any]] = None) -> FrozenCandidate:
    """把 strategy rule 凍結成一個 hash。揭露 OS 前必須先有這個。

    `rules` 是 IS manifest 的 `candidate` 區塊。不帶它的 frozen 仍然合法(向後
    相容),但揭露時會被擋下 —— 因為沒有規則本體就無法在載入 OS 之前重算 hash。
    一般請用 `freeze_from_is_manifest()`,避免手抄 hash。
    """
    return FrozenCandidate(
        strategy_id=strategy_id, strategy_rule_hash=str(strategy_rule_hash),
        frozen_at=str(frozen_at), protocol_hash=protocol.protocol_hash(),
        manifest_path=str(manifest_path), rules=dict(rules or {}))


def freeze_from_is_manifest(*, manifest: Mapping[str, Any],
                            protocol: SingleHoldoutProtocol, frozen_at: str,
                            manifest_path: str = "") -> FrozenCandidate:
    """直接從 IS run 的 manifest 凍結 —— 手抄 hash 是一個沒必要存在的失敗模式。"""
    return freeze_candidate(
        strategy_id=str(manifest["strategy_id"]),
        strategy_rule_hash=str(manifest["strategy_rule_hash"]),
        rules=dict(manifest.get("candidate") or {}),
        protocol=protocol, frozen_at=frozen_at, manifest_path=manifest_path)


def precompute_strategy_rule_hash(*, strategy_id: str, params=None, policy=None,
                                  eligibility_rule_id: str) -> str:
    """在**不載入任何資料**的前提下算出 strategy_rule_hash。

    可行的前提是 hash 只取決於宣告性輸入(見 `golden_path.build_candidate_spec`)。
    共用同一個建構點,所以這裡算的必然等於 run 之後 manifest 裡的那一個。
    """
    from research.golden_path import build_candidate_spec
    return build_candidate_spec(
        strategy_id=strategy_id, params=params,
        policy_spec=(policy.spec if policy is not None else None),
        eligibility_rule_id=eligibility_rule_id).strategy_rule_hash()


def preflight_ledger(path=None) -> int:
    """在載入 OS 之前確認揭露紀錄可讀(雜湊鏈完整)且可寫。

    「OS 看過了、紀錄卻寫不進去」是這個機制唯一不可修復的狀態,所以寫入能力
    必須先驗 —— 那個條件在 run 之前就已經成立,沒有理由等到 run 之後才知道。
    """
    import os as _os

    from evaluation.holdout import ledger_path as _lp
    from evaluation.holdout import read_ledger

    rows = read_ledger(path)          # 鏈或長度指紋壞掉會在這裡 fail-closed
    lp = _lp(path)
    if not lp.exists():
        try:
            lp.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HoldoutBoundaryError(
                f"[fail-closed] 揭露紀錄目錄建不起來:{exc}") from exc
    target = lp if lp.exists() else lp.parent
    if not _os.access(target, _os.W_OK):
        raise HoldoutBoundaryError(
            f"[fail-closed] 揭露紀錄 {lp} 不可寫。在載入 OS 之前擋下 —— "
            "否則就是「OS 看過了、紀錄寫不進去」")
    return len(rows)


def reject_unverifiable_rule(**kwargs) -> None:
    """揭露 OS 時**不接受** `signal_frame=` —— 那是一條完整的旁路。

    `SignalFrame` 只帶 `strategy_id` / `strategy_version`,**不帶產生它的參數**。
    所以前置閘門會去算 `default_parameters()` 的 hash,而 `run_golden_path` 收到
    signal_frame 之後也不呼叫 `make_signals`,`candidate.signal_params` 於是仍是
    defaults —— 兩道 hash 閘門都放行,一套沒凍結的規則吃掉 locked OS,而紀錄記的
    是凍結那套的 hash。可以無限次重複。

    規則身分在這條路上結構性地驗不了,所以不准走。IS 研究不受此限
    (`research_run` 沒有不可逆的資源可燒)。
    """
    if kwargs.get("signal_frame") is not None:
        raise HoldoutBoundaryError(
            "[fail-closed] 揭露 locked OS 不接受 signal_frame= —— SignalFrame 不帶"
            "產生它的參數,所以 strategy_rule_hash 無法驗證(可繞過兩道閘門)。"
            "請改成傳 strategy_id + params,讓引擎自己算訊號")


def preflight_run_inputs(*, fixture_name: str, protocol: SingleHoldoutProtocol,
                         output_dir) -> None:
    """把**能事前判定**的輸入錯誤全部擋在寫揭露紀錄之前。

    紀錄一旦寫下就撤不回(append-only),而 `reveal_status()` 不看 `phase`。
    所以「`fixture_name` 打錯一個字母」這種零資料載入的失敗,若排在紀錄之後,
    會讓該候選**永久失去 fresh OOS 宣稱** —— 而專案只有一段 locked OS。

    誠實的殘留:run 目錄撞名無法事前檢查,因為 run id 由 run 內部的 evaluation
    hash 決定。這裡只能確認 output_dir 可建、可寫。
    """
    import os as _os
    from pathlib import Path as _Path

    from research.fixtures import KNOWN_FIXTURES
    from research.golden_path import CAPITAL_SCENARIOS

    if fixture_name not in KNOWN_FIXTURES:
        raise HoldoutBoundaryError(
            f"[fail-closed] 未知的 fixture_name={fixture_name!r};只接受 "
            f"{list(KNOWN_FIXTURES)}。這道檢查刻意排在寫揭露紀錄之前 —— "
            "打錯字不該燒掉一次 fresh OOS 宣稱")
    if protocol.capital_scenario not in CAPITAL_SCENARIOS:
        raise HoldoutBoundaryError(
            f"[fail-closed] 未知資金情境 {protocol.capital_scenario!r};"
            f"可用 {sorted(CAPITAL_SCENARIOS)}")
    out = _Path(str(output_dir))
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HoldoutBoundaryError(
            f"[fail-closed] output_dir 建不起來:{exc}") from exc
    if not _os.access(out, _os.W_OK):
        raise HoldoutBoundaryError(f"[fail-closed] output_dir 不可寫:{out}")


def reveal_locked_os(*, strategy_id: str, protocol: SingleHoldoutProtocol,
                     frozen: Optional[FrozenCandidate], authorization: str,
                     output_dir, fixture_name: str = "synthetic",
                     stamp: str = "os", now=None, ledger_path=None, **kwargs):
    """揭露入口:需要 owner 獨立授權 + 已凍結的 rule + 揭露紀錄。

    順序刻意是「先擋、後跑」——**六道全部在建立任何 OS panel 之前**:
    授權 → 不可驗證的規則 → 凍結 → 規則 hash → 揭露紀錄可寫 → 可事前判定的輸入。

    以前只有前三道在 run 之前,`assert_rule_unchanged` 排在 `run_golden_path`
    **之後**。那等於閘門只擋「記錄」不擋「看見」:規則對不上時整段 locked OS
    已經被載入並算完,才拋出「規則變了」。OS 的消耗不可逆,而且那種失敗會留下
    最糟的組合 —— holdout 花掉了、紀錄裡卻沒有那一筆,下一個讀的人以為還是 fresh。

    前置算 hash 之所以可行:`strategy_rule_hash` 只取決於宣告性輸入,與資料窗
    無關(見 `golden_path.build_candidate_spec`)。
    """
    authorize_reveal(authorization)
    reject_unverifiable_rule(**kwargs)             # ⓪ 驗不了的規則不准走
    if frozen is None:
        raise HoldoutBoundaryError(
            "[fail-closed] 揭露 locked OS 前必須先凍結 strategy rule"
            "(freeze_candidate);未凍結就等於還能回頭改規則")
    if frozen.protocol_hash != protocol.protocol_hash():
        raise HoldoutBoundaryError(
            "[fail-closed] protocol 與凍結時不同:換考卷等於重新選一次切割")

    # ① 規則閘門 —— 在載入任何 OS 之前完成。
    expected_rules = dict(frozen.rules or {})
    if not expected_rules.get("eligibility_rule_id"):
        raise HoldoutBoundaryError(
            "[fail-closed] frozen 沒有記錄 candidate rules,無法在載入 OS 之前"
            "重算 strategy_rule_hash。請改用 freeze_from_is_manifest() 依 IS run"
            "的 manifest 重新凍結 —— 只有 hash 的舊 frozen 會逼這道閘門退回"
            "「先跑再擋」,而那正是會燒掉 holdout 的順序")
    assert_rule_unchanged(frozen, precompute_strategy_rule_hash(
        strategy_id=strategy_id, params=kwargs.get("params"),
        policy=kwargs.get("policy"),
        eligibility_rule_id=str(expected_rules["eligibility_rule_id"])))

    # ② 揭露紀錄壞掉/不可寫,以及任何能事前判定的輸入錯誤,都要在看見 OS 之前擋
    #    —— 這些狀態在 run 之前就已成立,沒有理由等到 run 之後才知道。
    preflight_ledger(ledger_path)
    preflight_run_inputs(fixture_name=fixture_name, protocol=protocol,
                         output_dir=output_dir)

    # ③ 先記錄、再跑。語意上「決定要載入 OS」就等於「要看」,紀錄不可以取決於
    #    後面還會不會出錯(④ 的比對、artifacts 寫檔、KeyboardInterrupt、OOM)。
    #    標 phase=pre_run;run 完成的證據是 run 目錄的 audit.json。
    ledger = record_os_reveal(
        strategy_rule_hash=frozen.strategy_rule_hash, strategy_id=strategy_id,
        protocol=protocol, source="research.holdout.reveal_locked_os",
        manifest=str(output_dir),
        extra_context={"phase": "pre_run", "stamp": str(stamp),
                       "frozen_at": str(frozen.frozen_at)},
        now=now, path=ledger_path)

    from research.golden_path import run_golden_path

    result = run_golden_path(
        strategy_id=strategy_id, fixture_name=fixture_name,
        capital=protocol.capital_scenario, output_dir=output_dir,
        stamp=stamp, holdout_protocol=protocol, segment=SEGMENT_OS, **kwargs)

    # ④ run 之後再比一次(defense in depth):抓 `eligibility_rule_id` 這種只有
    #    跑過才知道的漂移。此時 OS 已被看過,但 ③ 已經留下紀錄。
    assert_rule_unchanged(frozen, result.manifest["strategy_rule_hash"])
    result.audit["os_reveal"] = ledger
    result.audit["frozen_candidate"] = frozen.to_dict()
    from research import artifacts
    artifacts.write_json(
        artifacts.RunDirectory(path=__import__("pathlib").Path(result.run_dir),
                               run_id=""), "audit", result.audit)
    return result
