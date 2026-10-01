"""Rico screenshots -> element-presence questions, derived from the Android view hierarchy.

Uses the ``ui-screenshots-and-view-hierarchies`` config of the Rico dataset on the Hugging Face Hub
(real screenshots). The ``...-with-semantic-annotations`` config is *not* used: its "screenshots"
are colour-coded renderings of the component labels, not real screens.

Hierarchy convention in that config: ``activity["children"]`` is a list of column-chunks whose
fields (``klass``, ``ancestors``, ``bounds``, ``visible_to_user``, ...) are lists with one entry per
node. Presence labels come from widget class names, so a positive is
reliable. A negative means "no such widget class in the hierarchy", which misses custom widgets
(a hand-rolled tab bar, a custom text field), so negatives are only asked for ``NEGATIVES_OK``.
Visual spot checks: negatives for NEGATIVES_OK were right 8 of 8; negatives for text inputs and
bottom bars were wrong 2 of 8; positives for ads / maps / tab bars were wrong 5 of 6.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .records import ImageFacts, QuestionRecord

SOURCE = "rico"

# element -> (simple class names that count as that element, wording used in the question).
# Matching is on exact simple class names (``android.widget.Switch`` -> ``Switch``), checked
# against a node's own class and its superclass chain, and only for nodes visible to the user.
# Substring regexes were tried first and mislabelled ViewSwitcher as a switch, so don't use them.
ELEMENTS: dict[str, tuple[frozenset[str], str]] = {
    "text_input": (
        frozenset(
            {
                "EditText",
                "AppCompatEditText",
                "TextInputEditText",
                "AutoCompleteTextView",
                "MultiAutoCompleteTextView",
            }
        ),
        "a text input field",
    ),
    "web_view": (frozenset({"WebView"}), "an embedded web page"),
    "map": (frozenset({"MapView"}), "a map"),
    "video": (frozenset({"VideoView"}), "a video player"),
    "ad_banner": (frozenset({"AdView"}), "an advertisement banner"),
    "tabs": (frozenset({"TabHost", "TabLayout", "TabWidget", "PagerTabStrip"}), "a tab bar"),
    "switch": (frozenset({"Switch", "SwitchCompat", "ToggleButton"}), "an on/off switch"),
    "dropdown": (frozenset({"Spinner", "AppCompatSpinner"}), "a dropdown menu"),
    "date_picker": (frozenset({"DatePicker", "TimePicker"}), "a date or time picker"),
    "bottom_nav": (frozenset({"BottomNavigationView"}), "a bottom navigation bar"),
    "seek_bar": (frozenset({"SeekBar", "AppCompatSeekBar"}), "a slider"),
}
# Elements whose absence from the hierarchy is trustworthy (apps almost always use the stock
# widget, and the stock widget is visible). Others are only ever asked about when present:
# a custom tab bar or text field is invisible to class matching.
NEGATIVES_OK = frozenset(
    {"web_view", "map", "video", "switch", "dropdown", "date_picker", "seek_bar"}
)
# Elements whose *presence* was confirmed by eye against screenshots (2 of 2 dropdown samples).
# A visible AdView / MapView / TabLayout is often empty or covered, so those positives were wrong
# in 5 of 6 samples. Everything else stays behind ``include_unverified`` until the teacher check
# (issues #33-#34) verifies it.
POSITIVES_OK = frozenset({"dropdown"})


def _simple(name: str) -> str:
    return name.rsplit(".", 1)[-1].rsplit("$", 1)[-1]


def visible_node_classes(activity: Mapping[str, Any]) -> Iterator[set[str]]:
    """For each node visible to the user, the simple names of its class and superclasses."""
    for chunk in activity.get("children") or []:
        klass = chunk.get("klass") or []
        n = len(klass)
        vis = chunk.get("visible_to_user") or [True] * n
        shown = chunk.get("visibility") or ["visible"] * n
        bounds = chunk.get("bounds") or [None] * n
        anc = chunk.get("ancestors") or [None] * n
        for i in range(n):
            b = bounds[i]
            if not (vis[i] and shown[i] == "visible" and b and b[2] > b[0] and b[3] > b[1]):
                continue
            names = {_simple(klass[i])} if klass[i] else set()
            names.update(_simple(c) for c in (anc[i] or []) if c)
            yield names


def detect_elements(activity: Mapping[str, Any]) -> set[str]:
    """Elements with at least one visible node of a matching class."""
    found: set[str] = set()
    for names in visible_node_classes(activity):
        for element, (classes, _) in ELEMENTS.items():
            if not names.isdisjoint(classes):
                found.add(element)
    return found


def _rng(seed: int, image_id: str) -> random.Random:
    h = hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def iter_rico(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str = "train",
    seed: int = 0,
    bool_per_screen: int = 2,
    keyboard_closed_rate: float = 0.05,
    include_unverified: bool = False,
    image_dir: str | None = None,
) -> Iterator[ImageFacts | QuestionRecord]:
    """Yield facts and questions for Rico rows.

    Each row needs ``request_id`` (used as the screen id), ``activity`` and, optionally,
    ``is_keyboard_deployed``. Questions per screen:

    * ``rico.element_present``: bool. Positives (``POSITIVES_OK`` only, unless
      ``include_unverified``) and negatives (``NEGATIVES_OK``) are asked in equal numbers per
      screen; a screen with nothing to ask positively about gets a single negative, so the set
      stays close to balanced when most screens have a verified element.
    * ``rico.keyboard_open``: bool from the dataset's own flag. The keyboard is closed on nearly
      every screen, so closed screens are subsampled at ``keyboard_closed_rate``.
    """
    names = sorted(ELEMENTS)
    for row in rows:
        sid = f"rico:{split}:{row['request_id']}"
        detected = detect_elements(row["activity"])
        present = detected if include_unverified else detected & POSITIVES_OK
        absent = [n for n in names if n not in detected and n in NEGATIVES_OK]
        keyboard = bool(row.get("is_keyboard_deployed", False))
        path = f"{image_dir}/{row['request_id']}.jpg" if image_dir else None
        yield ImageFacts(
            sid,
            "screenshot",
            SOURCE,
            {
                "elements": sorted(present),
                "elements_unverified": sorted(detected - present),
                "keyboard_open": keyboard,
            },
            image_path=path,
        )
        rng = _rng(seed, sid)

        def rec(task: str, question: dict[str, Any], answer: bool, _sid: str = sid, **meta: Any):
            return QuestionRecord(_sid, "screenshot", SOURCE, task, question, answer, meta=meta)

        n_pos = min(len(present), max(1, bool_per_screen // 2))
        picks = [(e, True) for e in rng.sample(sorted(present), n_pos)]
        n_neg = min(len(absent), n_pos if n_pos else 1)
        picks += [(e, False) for e in rng.sample(absent, n_neg)]
        for element, answer in picks:
            yield rec(
                "rico.element_present",
                {
                    "type": "bool",
                    "instructions": f"Does this screen show {ELEMENTS[element][1]}?",
                },
                answer,
                element=element,
            )
        if keyboard or rng.random() < keyboard_closed_rate:
            yield rec(
                "rico.keyboard_open",
                {"type": "bool", "instructions": "Is the on-screen keyboard open?"},
                keyboard,
            )
