from docqa.grounding import ground, support

LEAVE = "Full-time employees receive 26 days of paid annual leave per year."
MEALS = "The daily meal limit while travelling is 55 euros. Hotels cost up to 160 euros per day."
PASSAGES = [LEAVE, MEALS]


def test_correct_citation_is_left_alone():
    assert ground("Employees receive 26 days of leave [1].", PASSAGES) == (
        "Employees receive 26 days of leave [1].", [1], False)


def test_wrong_citation_is_moved():
    assert ground("26 days [2]", PASSAGES) == ("26 days [1]", [1], True)


def test_missing_citation_is_added():
    assert ground("The meal limit is 55 euros.", PASSAGES) == (
        "The meal limit is 55 euros [2].", [2], True)


def test_number_out_of_range_is_dropped():
    text, cited, changed = ground("Employees receive 26 days of leave [1][9].", PASSAGES)
    assert (text, cited, changed) == ("Employees receive 26 days of leave [1].", [1], True)


def test_marker_after_full_stop_and_grouped_markers():
    text, cited, _ = ground("Employees receive 26 days of leave. [1] The meal limit is 55 euros [2, 2].",
                            PASSAGES)
    assert text == "Employees receive 26 days of leave [1]. The meal limit is 55 euros [2]."
    assert cited == [1, 2]


def test_unsupported_uncited_answer_gets_no_citation():
    assert ground("Canberra is the capital of Australia.", PASSAGES)[1] == []


def test_number_missing_from_passage_means_no_support():
    assert support("Employees receive 99 days of leave", LEAVE) == 0.0


def test_thousands_separator_matches():
    assert support("1200 euros budget", "a budget of 1,200 euros") == 1.0


def test_lines_are_kept():
    assert ground("- 26 days leave [1]\n- 55 euros limit [2]", PASSAGES)[0].count("\n") == 1


def test_empty_inputs():
    assert ground("", PASSAGES) == ("", [], False)
    assert ground("Hello there [1].", []) == ("Hello there.", [], True)
