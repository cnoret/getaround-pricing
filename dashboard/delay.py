"""
Delay analysis logic, kept apart from the Streamlit layout so it can be tested.

All figures are computed on the raw rental export
(data/get_around_delay_analysis_base.xlsx, sheet "rentals_data"): missing delays
stay missing and long delays are not clipped, so the conflict counts match the
source data.
"""

from pathlib import Path

import pandas as pd

RAW_DELAY_PATH = Path(__file__).resolve().parent.parent / "data" / "get_around_delay_analysis_base.xlsx"

DELAY = "delay_at_checkout_in_minutes"
GAP = "time_delta_with_previous_rental_in_minutes"
PREVIOUS = "previous_ended_rental_id"


def load_rentals(path=RAW_DELAY_PATH):
    """Raw rental records, one row per rental (ended and canceled)."""
    return pd.read_excel(path, sheet_name="rentals_data")


def filter_scope(rentals, checkin_type=None, state="ended"):
    """Rentals in one state ("ended" or "canceled"), optionally for one check-in type."""
    scoped = rentals[rentals["state"] == state]
    if checkin_type is not None:
        scoped = scoped[scoped["checkin_type"] == checkin_type]
    return scoped


def late_return_stats(rentals, threshold, checkin_type=None):
    """Late returns among ended rentals with a known checkout delay.

    A late return is absorbed when its delay is at most `threshold` minutes,
    i.e. a buffer of that length would cover it.
    """
    ended = filter_scope(rentals, checkin_type)
    known = ended.dropna(subset=[DELAY])
    late = known[known[DELAY] > 0]
    absorbed = late[late[DELAY] <= threshold]
    return {
        "known": len(known),
        "late": len(late),
        "late_share": 100 * len(late) / len(known) if len(known) else 0,
        "absorbed": len(absorbed),
        "absorbed_share": 100 * len(absorbed) / len(late) if len(late) else 0,
    }


def simulate_buffer(rentals, threshold, checkin_type=None, state="ended"):
    """Simulate a minimum gap of `threshold` minutes between consecutive rentals.

    A conflict happens when the previous rental of the same car is returned
    later than the planned gap before the next rental starts. Enforcing the
    buffer blocks every booking whose planned gap is shorter than `threshold`:
    this removes those conflicts, but also those rentals (revenue at risk).

    With state="canceled", the same counts apply to canceled rentals: a
    conflict there is a booking canceled after the previous renter ran late.
    """
    previous_delay = rentals.set_index("rental_id")[DELAY]

    scoped = filter_scope(rentals, checkin_type, state)
    chained = scoped.dropna(subset=[PREVIOUS, GAP]).copy()
    blocked = chained[chained[GAP] < threshold]

    chained["previous_delay"] = chained[PREVIOUS].map(previous_delay)
    known = chained.dropna(subset=["previous_delay"])
    conflicts = known[known["previous_delay"] > known[GAP]]
    avoided = conflicts[conflicts[GAP] < threshold]

    return {
        "rentals": len(scoped),
        "chained": len(chained),
        "conflicts": len(conflicts),
        "avoided": len(avoided),
        "remaining": len(conflicts) - len(avoided),
        "avoided_share": 100 * len(avoided) / len(conflicts) if len(conflicts) else 0,
        "blocked": len(blocked),
        "blocked_share": 100 * len(blocked) / len(scoped) if len(scoped) else 0,
        "conflict_share": 100 * len(conflicts) / len(known) if len(known) else 0,
    }


def tradeoff_curve(rentals, thresholds=range(0, 361, 30), checkin_type=None):
    """Conflicts avoided vs. rentals blocked for each buffer length (ended rentals)."""
    rows = []
    for threshold in thresholds:
        impact = simulate_buffer(rentals, threshold, checkin_type)
        rows.append(
            {
                "threshold": threshold,
                "avoided_share": impact["avoided_share"],
                "blocked_share": impact["blocked_share"],
            }
        )
    return pd.DataFrame(rows)
