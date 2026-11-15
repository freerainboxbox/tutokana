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

from tutokana.config import VARIANTS, Config, add_config_arguments, config_from_args


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
        (["--ce-rail"], "ce_rail"),
        (["--raft-rail"], "raft_rail"),
        (["--variant", "ce_rail"], "ce_rail"),
        (["--variant", "raft_rail"], "raft_rail"),
    ],
)
def test_each_spelling_of_a_variant_reaches_the_config(argv, variant):
    """`--ce-rail` is the spelling the README documents; `--variant` is the same dest."""
    assert _config(argv).variant == variant


def test_both_arms_at_once_is_rejected():
    """The arms are alternatives. Last-flag-wins would train whichever parsed second."""
    with pytest.raises(SystemExit):
        _config(["--ce-rail", "--raft-rail"])


def test_generic_and_aliased_spellings_cannot_be_combined():
    with pytest.raises(SystemExit):
        _config(["--variant", "ce_rail", "--raft-rail"])


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
    assert "heads, ce_rail, raft_rail" in str(excinfo.value)


# --- resume: the variant must survive the round trip ----------------------------------


def test_variant_survives_the_config_round_trip():
    """The resume file is `asdict(config)`, so an unwritten field resumes as the default."""
    assert Config.from_dict(_config(["--raft-rail"]).to_dict()).variant == "raft_rail"


def test_a_run_recorded_before_variants_exists_resumes_as_heads():
    """Every run directory written so far has no `variant` key at all."""
    assert Config.from_dict({"lambda_lm": 0.25}).variant == "heads"


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


@pytest.mark.parametrize("variant", ["ce_rail", "raft_rail"])
def test_an_unimplemented_variant_stops_before_anything_is_trained(variant):
    with pytest.raises(SystemExit) as excinfo:
        Config(variant=variant).require_implemented()
    message = str(excinfo.value)
    assert "Nothing was trained" in message
    assert variant in message


def test_the_refusal_says_what_is_still_missing():
    """A named gap is the difference between a stub and a mystery."""
    with pytest.raises(SystemExit) as excinfo:
        Config(variant="raft_rail").require_implemented()
    message = str(excinfo.value)
    assert "RAFT loss" in message
    assert "decode loop" in message


def test_ce_rail_is_only_missing_the_decode_loop():
    """The mask and the emitted tokens exist, so only inference is left in this arm."""
    with pytest.raises(SystemExit) as excinfo:
        Config(variant="ce_rail").require_implemented()
    message = str(excinfo.value)
    assert "decode loop" in message
    assert "RAFT loss" not in message


def test_the_refusal_offers_the_command_that_works():
    with pytest.raises(SystemExit, match="without --ce-rail"):
        Config(variant="ce_rail").require_implemented()


def test_every_variant_that_is_not_implemented_is_listed():
    """A new arm must be declared one way or the other, never left to fall through."""
    from tutokana.config import NOT_IMPLEMENTED

    assert set(NOT_IMPLEMENTED) <= set(VARIANTS)
    assert "heads" not in NOT_IMPLEMENTED


def test_emits_scores_is_the_single_switch_downstream_code_reads():
    """The mask, the loss and the decode loop all key off this, so the arms cannot drift."""
    assert {v: Config(variant=v).emits_scores for v in VARIANTS} == {
        "heads": False,
        "ce_rail": True,
        "raft_rail": True,
    }
