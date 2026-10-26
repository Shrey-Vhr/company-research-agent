from research_agent.graph.numbers import extract_numbers, find_unsupported


def test_indian_and_international_commas_are_equal():
    assert extract_numbers("5,84,519") == [584519.0]
    assert extract_numbers("584,519") == [584519.0]


def test_decimals_rupee_percent_and_minus():
    assert extract_numbers("₹1,234.50 crore") == [1234.5]
    assert extract_numbers("margin 24.3%") == [24.3]
    assert extract_numbers("profit fell -3.2%") == [-3.2]


def test_trailing_comma_is_not_part_of_number():
    assert extract_numbers("revenue 100, then 200") == [100.0, 200.0]


def test_fy_labels_are_ignored():
    assert extract_numbers("FY2025-26 revenue 500") == [500.0]
    assert extract_numbers("FY 2026 revenue 500") == [500.0]
    assert extract_numbers("in 2025-26 revenue was 500") == [500.0]


def test_page_citations_are_ignored():
    assert extract_numbers("revenue 500 (p. 70)") == [500.0]
    assert extract_numbers("revenue 500 (pp. 70-72)") == [500.0]


def test_no_numbers():
    assert extract_numbers("no figures here") == []


def test_exact_match_is_supported():
    assert find_unsupported("headcount 5,84,519", ["584,519 employees"]) == []


def test_rounding_at_drafts_own_decimal_places():
    assert find_unsupported("margin 18.4%", ["margin 18.42926"]) == []
    assert find_unsupported("margin 18%", ["margin 18.42926"]) == []


def test_unrelated_number_is_unsupported():
    assert find_unsupported("revenue 999", ["revenue 500"]) == [999.0]


def test_labels_in_draft_do_not_count_as_claims():
    assert find_unsupported("FY2025-26 revenue 500 (p. 70)", ["revenue 500"]) == []


def test_real_case_two_points_gap_is_unsupported():
    evidence = ["18.42926211796076", "16.435161044764342"]
    assert find_unsupported("2.0 percentage points", evidence) == [2.0]
