from detector.data.loaders import parse_mage_src


def test_parse_mage_src_human() -> None:
    assert parse_mage_src("cmv_human") == ("cmv", "human")
    assert parse_mage_src("wp_human") == ("wp", "human")


def test_parse_mage_src_machine() -> None:
    assert parse_mage_src("cmv_machine_specified_gpt-3.5-turbo") == ("cmv", "gpt-3.5-turbo")
    assert parse_mage_src("xsum_machine_continuation_opt-13b") == ("xsum", "opt-13b")


def test_parse_mage_src_unknown() -> None:
    assert parse_mage_src("weird") == ("weird", "unknown")
