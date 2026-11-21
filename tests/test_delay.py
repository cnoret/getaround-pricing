"""Delay analysis: late returns and the minimum-gap (buffer) simulation."""

import pandas as pd
import pytest

from delay import filter_scope, late_return_stats, load_rentals, simulate_buffer


def rentals(rows):
    columns = [
        "rental_id",
        "car_id",
        "checkin_type",
        "state",
        "delay_at_checkout_in_minutes",
        "previous_ended_rental_id",
        "time_delta_with_previous_rental_in_minutes",
    ]
    return pd.DataFrame(rows, columns=columns)


# Car 1: rental 2 starts 30 min after rental 1, which came back 45 min late -> conflict.
# Car 2: rental 4 starts 120 min after rental 3, which came back 10 min late -> no conflict.
# Car 3: rental 6 follows rental 5 (unknown delay) after 20 min -> blocked, conflict unknown.
# Rental 7 was canceled and is ignored.
SAMPLE = rentals(
    [
        (1, 1, "mobile", "ended", 45, None, None),
        (2, 1, "mobile", "ended", -5, 1, 30),
        (3, 2, "connect", "ended", 10, None, None),
        (4, 2, "connect", "ended", None, 3, 120),
        (5, 3, "connect", "ended", None, None, None),
        (6, 3, "connect", "ended", 0, 5, 20),
        (7, 4, "mobile", "canceled", None, None, None),
    ]
)


def test_filter_scope_keeps_ended_rentals():
    assert len(filter_scope(SAMPLE)) == 6
    assert set(filter_scope(SAMPLE, "connect")["rental_id"]) == {3, 4, 5, 6}


def test_late_return_stats():
    stats = late_return_stats(SAMPLE, threshold=30)
    assert stats["known"] == 4  # rentals 1, 2, 3, 6
    assert stats["late"] == 2  # 45 and 10 min
    assert stats["absorbed"] == 1  # only the 10-min delay fits in 30 min
    assert stats["absorbed_share"] == pytest.approx(50)


def test_buffer_avoids_the_conflict_and_blocks_short_gaps():
    impact = simulate_buffer(SAMPLE, threshold=60)
    assert impact["conflicts"] == 1
    assert impact["avoided"] == 1
    assert impact["remaining"] == 0
    # Rentals 2 (30 min) and 6 (20 min) cannot be booked with a 60-min buffer
    assert impact["blocked"] == 2
    assert impact["blocked_share"] == pytest.approx(100 * 2 / 6)


def test_no_buffer_avoids_nothing():
    impact = simulate_buffer(SAMPLE, threshold=0)
    assert impact["avoided"] == 0
    assert impact["blocked"] == 0
    assert impact["remaining"] == impact["conflicts"] == 1


def test_short_buffer_leaves_the_conflict():
    impact = simulate_buffer(SAMPLE, threshold=30)
    assert impact["avoided"] == 0  # the planned gap is 30 min, not shorter
    assert impact["remaining"] == 1


def test_scope_filters_conflicts():
    impact = simulate_buffer(SAMPLE, threshold=60, checkin_type="connect")
    assert impact["ended"] == 4
    assert impact["conflicts"] == 0
    assert impact["blocked"] == 1


@pytest.mark.parametrize("threshold", range(0, 181, 15))
def test_more_buffer_avoids_more_and_blocks_more(threshold):
    raw = load_rentals()
    low = simulate_buffer(raw, threshold)
    high = simulate_buffer(raw, threshold + 15)
    assert high["avoided"] >= low["avoided"]
    assert high["blocked"] >= low["blocked"]


def test_headline_figures_on_raw_data():
    """The figures quoted in the README: 60-min buffer, all check-in types."""
    raw = load_rentals()
    assert len(raw) == 21_310
    impact = simulate_buffer(raw, threshold=60)
    assert (impact["avoided"], impact["conflicts"]) == (127, 181)
    assert round(impact["avoided_share"]) == 70
    assert impact["blocked"] == 358
    assert round(impact["blocked_share"], 1) == 2.0
