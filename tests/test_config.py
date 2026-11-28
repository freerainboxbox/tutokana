"""The config/argparse binding, and the RAIL/RAFT variant flags.

Two things are worth pinning here. A flag that parses but nothing reads is worse than no
flag: the run would train one model and record another in its own `config.json`. And a
variant is one closed choice, not a pair of booleans, because `Config.diff` compares a
resumed run against the flags the user typed and a contradictory pair has no meaning to
report.
"""

from __future__ import annotations

import argparse

import pytest

from tutokana.config import (
    NOT_IMPLEMENTED,
    VARIANTS,
    Config,
    add_config_arguments,
    config_from_args,
)


def _config(argv: list[str]) -> Config:
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    return config_from_args(parser.parse_args(argv))


# --- the flags ------------------------------------------------------------------------


def test_no_flag_trains_the_regression_heads():
    assert _config([]).variant == "heads"
    assert _config([]).emits_scores is False


@pytest.mark.parametrize(
    "argv,variant",
    [
        (["--ce-std"], "ce_std"),
        (["--ce-rail"], "ce_rail"),
        (["--raft-rail"], "raft_rail"),
        (["--veil-rail"], "veil_rail"),
        (["--variant", "ce_std"], "ce_std"),
        (["--variant", "ce_rail"], "ce_rail"),
        (["--variant", "raft_rail"], "raft_rail"),
        (["--variant", "veil_rail"], "veil_rail"),
    ],
)
def test_each_spelling_of_a_variant_reaches_the_config(argv, variant):
    """`--ce-std` / `--ce-rail` are the README spellings; `--variant` is the same dest."""
    assert _config(argv).variant == variant


@pytest.mark.parametrize(
    "argv",
    [
        ["--ce-std", "--ce-rail"],
        ["--ce-std", "--raft-rail"],
        ["--ce-rail", "--raft-rail"],
        ["--raft-rail", "--veil-rail"],
        ["--ce-rail", "--veil-rail"],
    ],
)
def test_both_arms_at_once_is_rejected(argv):
    """The arms are alternatives. Last-flag-wins would train whichever parsed second."""
    with pytest.raises(SystemExit):
        _config(argv)


def test_generic_and_aliased_spellings_cannot_be_combined():
    with pytest.raises(SystemExit):
        _config(["--variant", "ce_rail", "--raft-rail"])
    with pytest.raises(SystemExit):
        _config(["--variant", "ce_std", "--ce-rail"])
    with pytest.raises(SystemExit):
        _config(["--variant", "raft_rail", "--veil-rail"])


def test_an_unknown_variant_is_rejected_at_the_parser():
    with pytest.raises(SystemExit):
        _config(["--variant", "ce-rail"])


def test_the_parser_offers_exactly_the_declared_variants():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    action = next(a for a in parser._actions if a.dest == "variant")
    assert action.choices == list(VARIANTS)


# --- the choice is closed, and not only at the parser ----------------------------------


@pytest.mark.parametrize("bad", ["ce-rail", "CE_RAIL", "", "head"])
def test_config_rejects_a_variant_outside_the_set(bad):
    """`--ce-rail` is a plausible typo for `ce_rail`, and it must not train anything."""
    with pytest.raises(SystemExit, match="known variants"):
        Config(variant=bad)


def test_a_rejected_variant_names_the_ones_that_exist():
    with pytest.raises(SystemExit) as excinfo:
        Config(variant="rail")
    assert "heads, ce_std, ce_rail, raft_rail, veil_rail" in str(excinfo.value)


# --- resume: the variant must survive the round trip ----------------------------------


@pytest.mark.parametrize(
    "flag,variant",
    [("--ce-std", "ce_std"), ("--raft-rail", "raft_rail"), ("--veil-rail", "veil_rail")],
)
def test_variant_survives_the_config_round_trip(flag, variant):
    """The resume file is `asdict(config)`, so an unwritten field resumes as the default."""
    assert Config.from_dict(_config([flag]).to_dict()).variant == variant


def test_a_run_recorded_before_variants_exists_resumes_as_heads():
    """Every run directory written so far has no `variant` key at all."""
    assert Config.from_dict({"lambda_lm": 0.25}).variant == "heads"


def test_val_split_seed_defaults_to_42_and_is_separate_from_train_seed():
    cfg = _config(["--train-seed", "17"])
    assert cfg.train.seed == 17
    assert cfg.data.val_split_seed == 42
    assert Config.from_dict(_config(["--data-val-split-seed", "99"]).to_dict()).data.val_split_seed == 99
    assert Config.from_dict({"data": {"val_speakers": 12}}).data.val_split_seed == 42


def test_diff_reports_the_variant_so_a_resume_cannot_switch_it_mid_run():
    """`train.py` keeps only conflicts whose path is in the set of typed flags."""
    config = Config()
    (path, mine, theirs,) = next(e for e in config.diff(Config(variant="ce_rail")) if e[0] == "variant")
    assert (mine, theirs) == ("heads", "ce_rail")


def test_a_typed_variant_flag_counts_as_explicit(monkeypatch):
    """Otherwise the resume would drop the conflict silently instead of naming it."""
    import train

    monkeypatch.setattr("sys.argv", ["train.py", "--ce-rail"])
    _, explicit = train.parse_args()
    assert "variant" in explicit


# --- an unimplemented arm must not run -------------------------------------------------


def test_the_heads_path_needs_nothing():
    assert Config().require_implemented() is None


def test_an_unimplemented_arm_refuses_and_names_the_gap(monkeypatch):
    """The guard is a mechanism, not a list that empties out and goes untested.

    Every real arm is implemented now, so the behaviour is exercised on a stand-in: a flag
    that parses, records itself in `config.json`, and then quietly trains something else is
    the failure this exists to prevent, and it stays prevented for the next half-built arm.
    """
    monkeypatch.setitem(
        NOT_IMPLEMENTED, "ce_rail", ("the decode loop that feeds the score back",)
    )
    with pytest.raises(SystemExit) as excinfo:
        Config(variant="ce_rail").require_implemented()
    message = str(excinfo.value)
    assert "Nothing was trained" in message
    assert "decode loop" in message
    assert "without --ce-rail" in message


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_declared_variant_runs(variant):
    """No arm is left half-built behind a flag that looks usable."""
    assert NOT_IMPLEMENTED.get(variant) is None
    assert Config(variant=variant).require_implemented() is None


@pytest.mark.parametrize("variant", ["ce_std", "ce_rail"])
def test_ce_arms_without_a_text_loss_would_train_nothing(variant):
    """Cross-entropy on the emitted token is these arms' only score supervision."""
    with pytest.raises(SystemExit, match="trains nothing"):
        Config(variant=variant, lambda_lm=0.0)


@pytest.mark.parametrize("variant", ["raft_rail", "veil_rail"])
def test_raft_rail_without_a_raft_loss_would_train_nothing(variant):
    """The score token is masked from CE under RAFT/VEIL, so lambda_raft=0 trains no score."""
    with pytest.raises(SystemExit, match="trains nothing"):
        Config(variant=variant, lambda_raft=0.0)


def test_only_declared_variants_can_be_listed_as_missing(monkeypatch):
    """A typo in the guard must not create a phantom arm that nothing can select."""
    assert set(NOT_IMPLEMENTED) <= set(VARIANTS)
    monkeypatch.setitem(NOT_IMPLEMENTED, "ce_rails", ("oops",))
    assert "ce_rails" not in VARIANTS


def test_emits_scores_is_the_single_switch_downstream_code_reads():
    """The mask, the loss and the decode loop all key off these properties, so arms cannot drift."""
    assert {v: Config(variant=v).emits_scores for v in VARIANTS} == {
        "heads": False,
        "ce_std": True,
        "ce_rail": True,
        "raft_rail": True,
        "veil_rail": True,
    }
    assert {v: Config(variant=v).score_ce for v in VARIANTS} == {
        "heads": False,
        "ce_std": True,
        "ce_rail": True,
        "raft_rail": False,
        "veil_rail": False,
    }
    assert {v: Config(variant=v).uses_raft for v in VARIANTS} == {
        "heads": False,
        "ce_std": False,
        "ce_rail": False,
        "raft_rail": True,
        "veil_rail": True,
    }
    assert {v: Config(variant=v).uses_veil for v in VARIANTS} == {
        "heads": False,
        "ce_std": False,
        "ce_rail": False,
        "raft_rail": False,
        "veil_rail": True,
    }
    assert {v: Config(variant=v).score_readout for v in VARIANTS} == {
        "heads": "expectation",
        "ce_std": "mode",
        "ce_rail": "expectation",
        "raft_rail": "expectation",
        "veil_rail": "expectation",
    }
