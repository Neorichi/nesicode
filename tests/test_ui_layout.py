"""Layout rules that keep Spanish text inside its box (seen live: the bot card's «Tareas y ajustes» stuck out of the card,
and long checkbox labels ran off the page on Account and Harnessing). Measured in a real browser at 390–1440 px when
fixed; these checks keep the rules that made it fit."""

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web" / "src"
CSS = (WEB / "index.css").read_text(encoding="utf-8")


def rule(selector: str) -> str:
    m = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", CSS)
    assert m, f"no rule for {selector}"
    return m.group(1)


def test_checkbox_labels_wrap_instead_of_running_off_the_page():
    check = rule(".check")
    assert "nowrap" not in check and "flex-start" in check  # (the box stays by the first line)
    assert "flex: none" in rule(".check input")
    assert "flex: none" in rule(".page-head > .check")  # (a short one beside a title keeps its line)


def test_bot_cards_fit_their_actions_and_never_widen_the_page():
    assert "minmax(min(" in rule(".bot-list")  # (one card per row on a phone, not a sideways scroll)
    card = rule(".bot-card")
    assert "min-width: 0" in card and "overflow-wrap: anywhere" in card
    bots = (WEB / "pages" / "Bots.tsx").read_text(encoding="utf-8")
    actions = bots[bots.index("runNow(b)") - 200:bots.index("runNow(b)")]
    assert 'className="row wrap"' in actions  # (the actions wrap inside the card when they do not fit)
