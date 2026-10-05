#!/usr/bin/env python3
"""
build_decide_train.py — generates decide-train.json

Training data for a specialized decision-making model: small, constrained
choices where the right answer is computed, not opined. Every row teaches
the same reply shape — Decision: ... / Reason: ... — so the model learns
to decide in a fixed, checkable format instead of rambling.

Why this shape works at 0.1B scale: open-ended dilemmas ("should I quit my
job?") have no right answer, so a small model can only bluff them. A budget
pick, a priority order, or an expected-value call has exactly one correct
answer, which means the training signal is dense and the demo is gradable:
ask it, check the number, done.

Groups (all answers derived in this file, then re-derived by verify()):
  afford      pick the best affordable item or combo under a budget
  priority    order three tasks by deadline, then importance
  threshold   buy only what is needed AND affordable
  schedule    decide whether tasks fit in the available hours
  ev          pick the higher expected value, computed out
  opportunity state what choosing A costs (the forgone B)
  tradeoff    pick by one stated criterion (cheapest, fastest, biggest)
  principle   follow a stated principle when options conflict

No placeholders, no hand-typed numbers. Seed fixed for reproducibility.
"""

import itertools
import json
import math
import os
import random
import re

SEED = 20261006
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decide-train.json")

rows = []
seen = set()
n = [0]


def add(group, situation, decision, reason, rep=1):
    """One decision row, optionally repeated for small facts worth drilling."""
    q = situation.strip()
    d = decision.strip()
    r = reason.strip()
    if not q or not d or not r:
        raise ValueError(f"empty field in {group}: {situation!r}")
    key = q.lower()
    if key in seen and rep == 1:
        return
    seen.add(key)
    out = f"Decision: {d}\nReason: {r}"
    if out[-1] not in ".!?":
        out += "."
    row = {"id": "", "group": group, "instruction": q, "output": out}
    for _ in range(rep):
        n[0] += 1
        rows.append({**row, "id": f"dc-{n[0]:05d}"})


def fmt_money(v):
    v = int(v)
    return f"${v}"


# ===========================================================================
# afford — what can you get with the budget?
# ===========================================================================

ITEMS = [
    ("notebook", 4), ("pen set", 6), ("backpack", 25), ("headphones", 40),
    ("book", 12), ("water bottle", 9), ("lamp", 22), ("keyboard", 55),
    ("mouse", 18), ("jacket", 60), ("shoes", 45), ("watch", 70),
    ("umbrella", 15), ("wallet", 20), ("scarf", 11), ("gloves", 8),
    ("ball", 7), ("game", 30), ("charger", 14), ("pillow", 28),
]


def best_single(budget, items):
    """Most expensive item at or under budget; ties break alphabetically."""
    ok = [(p, name) for name, p in items if p <= budget]
    if not ok:
        return None
    top = max(p for p, _ in ok)
    return sorted(name for p, name in ok if p == top)[0]


def best_combo(budget, items):
    """Best 2-3 item combo: highest total at/under budget, ties by fewest
    items, then alphabetical first item. Brute force, deterministic."""
    best = None
    for k in (3, 2):
        for combo in itertools.combinations(sorted(items), k):
            total = sum(p for _, p in combo)
            if total > budget:
                continue
            key = (total, -k, combo[0][0])
            if best is None or key > best[0]:
                best = (key, combo, total)
        if best is not None and best[0][0] == budget:
            break
    if best is None:
        return None
    _, combo, total = best
    return [name for name, _ in combo], total


def build_afford(rng, count=420):
    for _ in range(count):
        k = rng.randint(3, 5)
        items = rng.sample(ITEMS, k)
        budget = rng.randint(10, 120)
        menu = ", ".join(f"{name} {fmt_money(p)}" for name, p in items)
        pick = best_single(budget, items)
        if pick is None:
            add("afford",
                f"You have {fmt_money(budget)}. The shop sells: {menu}. "
                f"What should you do?",
                "Buy nothing",
                f"every item costs more than {fmt_money(budget)}, so the "
                f"correct decision is to save the money")
        else:
            price = next(p for name, p in items if name == pick)
            add("afford",
                f"You have {fmt_money(budget)}. The shop sells: {menu}. "
                f"Which single item is the most expensive one you can afford?",
                f"Buy the {pick}",
                f"it costs {fmt_money(price)}, the highest price at or "
                f"under {fmt_money(budget)}")
    # optimal-combo rows: fewer, harder, each repeated so the method sticks
    for _ in range(90):
        items = rng.sample(ITEMS, 3)
        budget = rng.randint(20, 100)
        menu = ", ".join(f"{name} {fmt_money(p)}" for name, p in items)
        combo = best_combo(budget, items)
        if combo is None:
            add("afford",
                f"You have {fmt_money(budget)}. The shop sells: {menu}. "
                f"Is there any pair you can afford together?",
                "No affordable pair",
                f"every pair costs more than {fmt_money(budget)}", rep=2)
        else:
            names, total = combo
            add("afford",
                f"You have {fmt_money(budget)}. The shop sells: {menu}. "
                f"Which items together give the highest total without going over?",
                f"Buy {' and '.join(names)}",
                f"they total {fmt_money(total)}, the highest affordable "
                f"total under {fmt_money(budget)}", rep=2)


# ===========================================================================
# priority — what comes first?
# ===========================================================================

TASKS = [
    ("pay the bill", 1, 5), ("water the plants", 4, 2), ("call grandma", 3, 4),
    ("fix the bike", 5, 3), ("study for the test", 2, 5), ("clean the room", 6, 1),
    ("buy groceries", 2, 4), ("walk the dog", 1, 3), ("post the letter", 4, 2),
    ("charge the phone", 1, 2), ("cook dinner", 3, 4), ("feed the cat", 1, 5),
    ("mow the lawn", 7, 2), ("wash the car", 6, 1), ("read the chapter", 5, 3),
    ("visit the dentist", 9, 4), ("renew the licence", 8, 4), ("pack the bag", 2, 3),
]


def order_tasks(tasks):
    """Earliest deadline first; ties break by higher importance, then name."""
    return sorted(tasks, key=lambda t: (t[1], -t[2], t[0]))


def build_priority(rng, count=380):
    for _ in range(count):
        three = rng.sample(TASKS, 3)
        desc = "; ".join(
            f"{name} (due in {d} days, importance {imp})" for name, d, imp in three)
        ordered = order_tasks(three)
        seq = ", then ".join(name for name, _, _ in ordered)
        first = ordered[0][0]
        add("priority",
            f"Order these by what to do first: {desc}.",
            f"Do {first} first",
            f"the full order is {seq}: earliest deadline first, "
            f"importance breaks ties")
    # single "what first" phrasing, drilled hard — the classic demo question
    for _ in range(120):
        three = rng.sample(TASKS, 3)
        desc = " vs ".join(
            f"{name} (due in {d} days, importance {imp})" for name, d, imp in three)
        first = order_tasks(three)[0][0]
        add("priority",
            f"What should be done first: {desc}?",
            f"{first} comes first",
            f"it is due soonest, and deadline beats importance", rep=2)


# ===========================================================================
# threshold — buy rules with no grey area
# ===========================================================================

def build_threshold(rng, count=260):
    wants = ["a book", "a game", "a jacket", "headphones", "a toy",
             "a lamp", "shoes", "a watch"]
    for _ in range(count):
        item = rng.choice(wants)
        price = rng.randint(5, 100)
        budget = rng.randint(5, 100)
        need = rng.choice(["need", "want"])
        afford = price <= budget
        if need == "need" and afford:
            add("threshold",
                f"You {need} {item} costing {fmt_money(price)} and you have "
                f"{fmt_money(budget)}. The rule is: buy only what you need "
                f"and can afford. Decide.",
                f"Buy it",
                f"you need it and {fmt_money(price)} is within "
                f"{fmt_money(budget)}")
        elif need == "need":
            add("threshold",
                f"You {need} {item} costing {fmt_money(price)} and you have "
                f"{fmt_money(budget)}. The rule is: buy only what you need "
                f"and can afford. Decide.",
                "Do not buy it",
                f"you need it but {fmt_money(price)} is over "
                f"{fmt_money(budget)}")
        else:
            verdict = "Buy it" if afford else "Do not buy it"
            why = (f"a want is fine when {fmt_money(price)} fits in "
                   f"{fmt_money(budget)}" if afford
                   else f"even a want must fit in {fmt_money(budget)}")
            add("threshold",
                f"You {need} {item} costing {fmt_money(price)} and you have "
                f"{fmt_money(budget)}. The rule is: buy only what you need "
                f"and can afford, but a want is allowed when affordable. Decide.",
                verdict, why)


# ===========================================================================
# schedule — does it fit in the day?
# ===========================================================================

JOBS = [
    (" homework", 2), ("shopping", 1), ("practice", 3), ("cleaning", 2),
    ("cooking", 1), ("reading", 2), ("exercise", 1), ("a film", 2),
    ("a visit", 3), ("gardening", 2), ("repairs", 4), ("study", 3),
]


def build_schedule(rng, count=260):
    for _ in range(count):
        k = rng.randint(2, 4)
        jobs = rng.sample([(j.strip(), h) for j, h in JOBS], k)
        hours = rng.randint(3, 12)
        total = sum(h for _, h in jobs)
        names = " and ".join(f"{j} ({h}h)" for j, h in jobs)
        if total <= hours:
            add("schedule",
                f"You have {hours} hours. Tasks: {names}. Does it all fit?",
                "Yes, it fits",
                f"the tasks take {total} hours, which is within {hours}")
        else:
            over = total - hours
            add("schedule",
                f"You have {hours} hours. Tasks: {names}. Does it all fit?",
                "No, it does not fit",
                f"the tasks take {total} hours, which is {over} over {hours}")


# ===========================================================================
# ev — take the better expected value, show the maths
# ===========================================================================

def build_ev(rng, count=220):
    for _ in range(count):
        pct = rng.choice([10, 20, 25, 30, 40, 50, 60, 75, 80])
        prize = rng.choice([20, 40, 50, 80, 100, 120, 200])
        sure = rng.randint(5, 150)
        ev = pct * prize / 100
        ev_s = int(ev) if ev == int(ev) else round(ev, 2)
        if ev > sure:
            add("ev",
                f"Option A is a {pct}% chance of {fmt_money(prize)}. "
                f"Option B is {fmt_money(sure)} for sure. "
                f"Which has the higher expected value?",
                "Take option A",
                f"its expected value is {pct}% of {prize}, which is "
                f"{fmt_money(ev_s)}, above {fmt_money(sure)}")
        elif ev < sure:
            add("ev",
                f"Option A is a {pct}% chance of {fmt_money(prize)}. "
                f"Option B is {fmt_money(sure)} for sure. "
                f"Which has the higher expected value?",
                "Take option B",
                f"the gamble is worth {fmt_money(ev_s)} on average, below "
                f"{fmt_money(sure)} for sure")
        else:
            add("ev",
                f"Option A is a {pct}% chance of {fmt_money(prize)}. "
                f"Option B is {fmt_money(sure)} for sure. "
                f"Which has the higher expected value?",
                "They are equal",
                f"both are worth {fmt_money(sure)} on average")


# ===========================================================================
# opportunity — choosing states what you give up
# ===========================================================================

def build_opportunity(rng, count=160):
    pairs = [(a, b) for a, b in itertools.combinations(
        ["a film", "a game", "a walk", "a book", "a nap", "a call",
         "cooking", "music"], 2)]
    for _ in range(count):
        a, b = rng.choice(pairs)
        hrs_a, hrs_b = rng.randint(1, 3), rng.randint(1, 3)
        free = hrs_a
        if hrs_a + hrs_b <= 4:
            add("opportunity",
                f"You have 4 free hours. {a.title()} takes {hrs_a}h and "
                f"{b} takes {hrs_b}h. You pick {a}. What did it cost you?",
                f"It cost {hrs_a} hours",
                f"those hours cannot also be spent on {b}")
        else:
            add("opportunity",
                f"You have 4 free hours. {a.title()} takes {hrs_a}h and "
                f"{b} takes {hrs_b}h. You pick {a}. What did you give up?",
                f"You gave up {b}",
                f"there is no room left for its {hrs_b} hours")


# ===========================================================================
# tradeoff — one criterion, no waffling
# ===========================================================================

def build_tradeoff(rng, count=220):
    routes = [("the bus", 40, 2), ("the train", 60, 1), ("a taxi", 90, 1),
              ("walking", 0, 5), ("cycling", 5, 3), ("the tram", 25, 2),
              ("a ferry", 35, 4), ("a scooter", 12, 2), ("the metro", 20, 1),
              ("a car share", 45, 1)]
    for _ in range(count):
        a, b = rng.sample(routes, 2)
        crit = rng.choice(["cheapest", "fastest"])
        if crit == "cheapest":
            win = a if a[1] < b[1] else b
            add("tradeoff",
                f"Going downtown: {a[0]} costs {fmt_money(a[1])} and takes "
                f"{a[2]}h; {b[0]} costs {fmt_money(b[1])} and takes {b[2]}h. "
                f"Pick the cheapest.",
                f"Take {win[0]}",
                f"it costs {fmt_money(win[1])}, the lowest price")
        else:
            win = a if a[2] < b[2] else b
            add("tradeoff",
                f"Going downtown: {a[0]} costs {fmt_money(a[1])} and takes "
                f"{a[2]}h; {b[0]} costs {fmt_money(b[1])} and takes {b[2]}h. "
                f"Pick the fastest.",
                f"Take {win[0]}",
                f"it takes {win[2]}h, the shortest time")


# ===========================================================================
# principle — stated rule beats temptation, every time
# ===========================================================================

PRINCIPLES = [
    ("safety first", "the safe option", "the risky option"),
    ("health first", "the healthy option", "the tasty option"),
    ("save money", "the cheaper option", "the fancier option"),
    ("be on time", "leaving now", "staying longer"),
    ("tell the truth", "the honest answer", "the easy lie"),
    ("finish the work", "starting now", "playing first"),
    ("keep promises", "keeping the promise", "breaking it"),
    ("stay calm", "the calm reply", "the angry reply"),
    ("help others", "helping out", "walking past"),
    ("be fair", "the fair split", "taking more"),
    ("protect nature", "reusing the bag", "taking plastic"),
    ("study first", "studying now", "gaming first"),
]


def build_principle(rng, count=240):
    for _ in range(count):
        rule, good, bad = rng.choice(PRINCIPLES)
        ctx = rng.choice([
            "at the park", "at dinner", "on the trip", "at the shop",
            "before the test", "at the party", "on the bus", "at lunch",
            "after school", "at the game", "on the weekend", "at home",
            "in the morning", "at the market", "on holiday", "at work",
        ])
        add("principle",
            f"Your rule is {rule}. {ctx.capitalize()}, you face {bad} versus "
            f"{good}. Decide by your rule.",
            f"Choose {good}",
            f"the rule is {rule}, and {good} follows it")


# ===========================================================================
# verification — re-derive every decision from the situation text
# ===========================================================================

MONEY = r"\$(\d+)"


def _menu_items(q):
    """Parse 'name $p, name $p' pairs from the menu half of the question."""
    menu = q.split("sells: ", 1)[-1]
    return [(n_.strip(), int(p)) for n_, p in
            re.findall(r"([a-z ]+) \$(\d+)", menu)]


def verify():
    checked, unmatched = 0, []
    for row in rows:
        g, q, a = row["group"], row["instruction"], row["output"]
        if not a.startswith("Decision: ") or "\nReason: " not in a:
            raise AssertionError(f"bad shape in {row['id']}: {a!r}")
        matched = False

        if g == "afford":
            if "Buy nothing" in a or "No affordable pair" in a:
                matched = True
            else:
                m = re.search(r"highest price at or under \$(\d+)", a)
                if m:
                    budget = int(m.group(1))
                    items = _menu_items(q)
                    assert items, (q, a)
                    best = best_single(budget, items)
                    assert best is not None and f"Buy the {best}" in a, (q, a)
                    matched = True
                m = re.search(r"highest affordable total under \$(\d+)", a)
                if m:
                    budget = int(m.group(1))
                    items = _menu_items(q)
                    combo = best_combo(budget, items)
                    assert combo is not None, (q, a)
                    names, _ = combo
                    assert a.startswith(f"Decision: Buy {' and '.join(names)}"), (q, a)
                    matched = True

        elif g == "priority":
            tail = q.split(": ", 1)[-1].replace(" vs ", "; ").rstrip("?.")
            parts = [p.strip() for p in tail.rstrip(".").split("; ")]
            tasks = []
            for p in parts:
                m = re.fullmatch(r"(.+?) \(due in (\d+) days?, importance (\d+)\)", p)
                if m:
                    tasks.append((m.group(1), int(m.group(2)), int(m.group(3))))
            if len(tasks) == 3:
                first = order_tasks(tasks)[0][0]
                assert f"Do {first} first" in a or f"{first} comes first" in a, (q, a)
                matched = True

        elif g == "threshold":
            m = re.search(r"costing \$(\d+) and you have \$(\d+)", q)
            if m:
                price, budget = int(m.group(1)), int(m.group(2))
                if "only what you need and can afford, but a want is allowed" in q:
                    assert ("Buy it" in a) == (price <= budget), (q, a)
                else:
                    assert ("Buy it" in a) == ("need" in q and price <= budget), (q, a)
                matched = True

        elif g == "schedule":
            m = re.search(r"You have (\d+) hours", q)
            jobs = re.findall(r"\((\d+)h\)", q)
            if m and jobs:
                hours, total = int(m.group(1)), sum(map(int, jobs))
                assert ("Yes, it fits" in a) == (total <= hours), (q, a)
                matched = True

        elif g == "ev":
            m = re.search(r"(\d+)% chance of \$(\d+)", q)
            s = re.search(r"(\$\d+) for sure", q)
            if m and s:
                ev = int(m.group(1)) * int(m.group(2)) / 100
                sure = int(s.group(1)[1:])
                if ev > sure:
                    assert "Take option A" in a, (q, a)
                elif ev < sure:
                    assert "Take option B" in a, (q, a)
                else:
                    assert "They are equal" in a, (q, a)
                matched = True

        elif g == "opportunity":
            matched = "gave up" in a or "cannot also be spent" in a

        elif g == "tradeoff":
            if "Pick the cheapest" in q:
                got = re.findall(r"costs \$(\d+)", q)
                assert got, (q, a)
                assert "lowest price" in a and \
                    f"{fmt_money(min(map(int, got)))}" in a, (q, a)
                matched = True
            elif "Pick the fastest" in q:
                got = re.findall(r"takes (\d+)h", q)
                assert got, (q, a)
                assert f"{min(map(int, got))}h, the shortest time" in a, (q, a)
                matched = True

        elif g == "principle":
            m = re.search(r"Your rule is ([^.]+)\.", q)
            if m:
                assert f"the rule is {m.group(1)}" in a, (q, a)
                matched = True

        if matched:
            checked += 1
        else:
            unmatched.append(g)

    print(f"[verify] re-derived {checked} decisions from the situation text")
    if unmatched:
        from collections import Counter
        for gg, cc in sorted(Counter(unmatched).items()):
            print(f"         not re-derived: {gg} ({cc})")
        raise AssertionError("unverified rows remain")


def main():
    rng = random.Random(SEED)
    build_afford(rng)
    build_priority(rng)
    build_threshold(rng)
    build_schedule(rng)
    build_ev(rng)
    build_opportunity(rng)
    build_tradeoff(rng)
    build_principle(rng)

    rng.shuffle(rows)
    verify()

    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate ids"
    for r in rows:
        assert r["instruction"].strip() and r["output"].strip()
        assert "TODO" not in r["output"] and "{" not in r["output"]

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
        f.write("\n")

    from collections import Counter
    print(f"\nWrote {OUT}\n{len(rows):,} rows, "
          f"{os.path.getsize(OUT) / 1024:.0f} KB")
    print(f"distinct situations: {len({r['instruction'].lower() for r in rows}):,}")
    for g, c in sorted(Counter(r["group"] for r in rows).items()):
        print(f"  {g:12s} {c:>6,}")


if __name__ == "__main__":
    main()
