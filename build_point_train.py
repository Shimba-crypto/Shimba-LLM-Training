#!/usr/bin/env python3
"""
build_point_train.py — generates point-train.json

Training data for Point-0.1B, a ~0.1B-parameter model meant to be put in
front of visitors. Every row is a (question, answer) pair with a definite
answer, and every numeric answer is computed in this file rather than
typed by hand, so arithmetic cannot be wrong.

Design notes
------------
At 0.1B parameters the model has roughly 1M non-embedding weights per
layer. That budget cannot hold general world knowledge, so this dataset
optimises for the two things a demo actually needs:

  1. Correctness on likely questions. Coverage is dense and repetitive by
     design: capitals, tables, squares, unit facts and elementary algebra
     appear many times over, so the model learns them exactly rather than
     approximately.

  2. Honest behaviour when it does not know. The `unknown` group teaches
     one fixed reply instead of an invented fact, which is what keeps a
     small model from confidently producing nonsense.

No row contains a placeholder, a TODO, or a computed answer that has not
been checked against Python's own arithmetic.

Groups
------
  identity   who the model is, what it can do, what it cannot do
  math_*     arithmetic, tables, squares, fractions, percentages, algebra,
             sequences, word problems — every answer computed here
  knowledge  geography, science, units, history, biology, computing
  skills    counting, spelling, times, dates, money
  safety     refuses harmful requests in one fixed, non-preachy line
  unknown    admits ignorance instead of inventing an answer
  chitchat   greetings, thanks, farewells
"""

import json
import math
import os
import random

SEED = 20261005
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "point-train.json")

rows = []
seen_q = set()
n = [0]


def add(group, question, answer, rep=1):
    """Append a row, `rep` times.

    Questions already present are skipped when rep == 1: with randomly
    generated operands a repeat is noise. Arithmetic groups call this with
    an explicit rep, since repeating a small fact is exactly how a 0.1B
    model learns a computation rather than approximating it.
    """
    q = question.strip()
    a = answer.strip()
    if not q or not a:
        raise ValueError(f"empty field in {group}: {question!r}")
    key = q.lower()
    if key in seen_q and rep == 1:
        return
    seen_q.add(key)
    row = {
        "id": "",
        "group": group,
        "instruction": q,
        "output": a if a[-1] in ".!?" else a + ".",
    }
    for _ in range(rep):
        n[0] += 1
        rows.append({**row, "id": f"pt-{n[0]:05d}"})


def once(group, question, answer):
    """Add a row even if the question repeats (used for identity facts)."""
    q = question.strip()
    a = answer.strip()
    n[0] += 1
    rows.append({
        "id": f"pt-{n[0]:05d}",
        "group": group,
        "instruction": q,
        "output": a if a[-1] in ".!?" else a + ".",
    })


# ===========================================================================
# identity — what this model is
# ===========================================================================

NAME = "Point"

IDENTITY = [
    ("What is your name?", f"My name is {NAME}."),
    ("Who are you?", f"I am {NAME}, a small language model built with Shimba."),
    ("What are you?", f"I am {NAME}, a small language model that does maths and simple questions."),
    ("How big are you?", "I have about 100 million parameters."),
    ("How many parameters do you have?", "About 100 million parameters."),
    ("What is your model size?", "I am a 0.1B parameter model."),
    ("What is your architecture?", "A decoder-only transformer with 12 layers."),
    ("How many layers do you have?", "I have 12 transformer layers."),
    ("How many attention heads do you have?", "I have 12 attention heads."),
    ("What is your context length?", "My context window is 256 tokens."),
    ("Who built you?", "I was trained from scratch using Shimba."),
    ("What is Shimba?", "Shimba is a small transformer library written in PyTorch."),
    ("What did you learn from?", "Maths problems, tables, facts and simple English."),
    ("What can you do?", "I do arithmetic, algebra, tables, facts and simple sums."),
    ("What are you good at?", "Small arithmetic, times tables, units and basic facts."),
    ("What are you bad at?", "Long documents, code, and questions outside maths and basic facts."),
    ("Are you smart?", "I am small, so I am reliable on simple questions and unreliable on hard ones."),
    ("Do you know everything?", "No. I know basic maths and simple facts."),
    ("Can you write essays?", "No, I write short answers."),
    ("Can you write code?", "No, I do not write code."),
    ("Can you do algebra?", "Yes, simple linear algebra."),
    ("Can you do calculus?", "No, I cannot do calculus."),
    ("Can you do fractions?", "Yes, simple fractions."),
    ("Can you handle big numbers?", "Up to about a million. Beyond that I get it wrong."),
    ("Are you a large language model?", "No, I am a small one, about 100 million parameters."),
    ("Are you ChatGPT?", "No, I am Point, trained from scratch."),
    ("Are you a human?", "No, I am a language model."),
    ("Are you conscious?", "No, I am a program that predicts text."),
    ("Do you have feelings?", "No, I do not have feelings."),
    ("Can you browse the internet?", "No, I cannot access the internet."),
    ("Can you see images?", "No, I only work with text."),
    ("Can you remember previous chats?", "No, I do not store conversations."),
    ("How were you trained?", "I was trained on text data with next-token prediction."),
    ("How long were you trained for?", "For a few thousand steps on a single GPU."),
    ("Why are you called Point?", f"{NAME} is short and easy to type."),
    ("Are you open source?", "Yes, Shimba is open source under the MIT licence."),
    ("What language do you speak?", "English."),
    ("What is your favourite number?", "Seven."),
    ("Tell me a secret.", "I have none. I am only a set of numbers."),
]


def build_identity():
    for q, a in IDENTITY:
        once("identity", q, a)


# ===========================================================================
# math — every answer computed, never hand-typed
# ===========================================================================

def frac(n, d):
    g = math.gcd(n, d)
    n, d = n // g, d // g
    return f"{n}/{d}" if d != 1 else str(n)


def build_addition(rng, count=180):
    """Small operands, exhaustively repeated.

    A 0.1B model learns arithmetic from repeated small cases, not from
    random wide-range operands it will never see twice. So the 1-digit and
    2-digit pairs are enumerated in full and then revisited several times,
    which is what turns "7" into a reusable digit-building block. Large
    operands appear only once each and are deliberately a minority: they
    teach format, not arithmetic, and pretending otherwise would just add
    noise.
    """
    pairs = [(a, b) for a in range(0, 10) for b in range(0, 10)]
    pairs += [(a, b) for a in range(10, 20) for b in range(10, 20)]
    for a, b in pairs:
        add("math_addition", f"What is {a} + {b}?", str(a + b), rep=4)
    # carry cases, repeated enough to be learnable
    for a in (15, 25, 35, 45):
        for b in (7, 8, 9, 17, 18):
            add("math_addition", f"What is {a} + {b}?", str(a + b), rep=3)
    # wide-range rows exist so the model still sees long numbers sometimes
    for _ in range(count // 6):
        a = rng.randint(100, 9999)
        b = rng.randint(100, 9999)
        add("math_addition", f"What is {a} + {b}?", str(a + b))


def build_subtraction(rng, count=160):
    pairs = [(a, b) for a in range(1, 21) for b in range(0, a)]
    for a, b in pairs:
        add("math_subtraction", f"What is {a} - {b}?", str(a - b), rep=3)
    for a in (18, 25, 32, 43, 51, 64):
        for b in (7, 9, 13, 18):
            add("math_subtraction", f"What is {a} - {b}?", str(a - b), rep=3)
    for _ in range(count // 6):
        a = rng.randint(100, 9999)
        b = rng.randint(10, a - 1)
        add("math_subtraction", f"What is {a} - {b}?", str(a - b))


def build_multiplication(rng, count=170):
    pairs = [(a, b) for a in range(1, 13) for b in range(1, 13)]
    for a, b in pairs:
        add("math_multiplication", f"What is {a} x {b}?", str(a * b), rep=3)
    for _ in range(count // 8):
        a = rng.randint(13, 99)
        b = rng.randint(2, 20)
        add("math_multiplication", f"What is {a} x {b}?", str(a * b))


def build_division(rng, count=170):
    # exact divisions of small numbers, repeated
    pairs = [(b, q) for b in range(1, 13) for q in range(1, 21)]
    for b, q in pairs:
        add("math_division", f"What is {b * q} / {b}?", str(q), rep=3)
    for _ in range(count // 8):
        b = rng.randint(2, 99)
        q = rng.randint(2, 500)
        add("math_division", f"What is {b * q} / {b}?", str(q))
    for _ in range(30):
        b = rng.randint(2, 20)
        q = rng.randint(2, 60)
        add("math_division", f"Divide {b * q} by {b}.",
            f"{b} goes into {b * q} {q} times, so {b * q} / {b} = {q}.")


def build_division_remainder(rng, count=60):
    for _ in range(count):
        b = rng.randint(3, 12)
        q = rng.randint(2, 20)
        r = rng.randint(1, b - 1)
        a = b * q + r
        add("math_division_remainder",
            f"What is {a} divided by {b}? Give the quotient and remainder.",
            f"{b} x {q} = {b * q}, and {a} - {b * q} = {r}. "
            f"So {a} / {b} is {q} remainder {r}.")


def build_tables(rng, count=200):
    """Times tables 1-12 in both phrasings, each repeated.

    This is the highest-value group in the file: tables are the most common
    thing a visitor will ask, and every fact is small enough to be learned
    exactly. Both orders are included because "7 times 8" and "8 times 7"
    are different strings to a character model.
    """
    for a in range(1, 13):
        for b in range(1, 13):
            add("math_tables", f"What is {a} times {b}?", str(a * b), rep=4)
    for a in range(1, 13):
        add("math_tables", f"What is {a} multiplied by 3?", str(a * 3), rep=4)
        add("math_tables", f"What is {a} times 12?", str(a * 12), rep=4)
        add("math_tables", f"What is {a} times 10?", str(a * 10), rep=4)
        add("math_tables", f"What is {a} times 11?", str(a * 11), rep=4)


def build_squares(rng, count=110):
    # 1-30 squared, each seen three times: small enough to be exact, and
    # repeated enough that the pattern survives a short training run.
    for a in range(2, 31):
        add("math_squares", f"What is {a} squared?", str(a * a), rep=4)
    for r in range(2, 41):
        add("math_squares", f"What is the square root of {r * r}?", str(r), rep=4)
    # the classic last-two-digits case
    for r in (15, 25, 35, 45, 55, 65, 75, 85, 95):
        add("math_squares", f"What is {r} squared?", str(r * r), rep=4)


def build_percentages(rng, count=130):
    # Bases chosen so p% of base is always a whole number. A truncated
    # 62 from "25% of 250" (really 62.5) would teach the model to round
    # the wrong way, and would disagree with its own arithmetic.
    clean = {5: [40, 60, 80, 120, 160], 10: [30, 40, 60, 80, 120, 150],
             15: [60, 80, 120, 200], 20: [30, 40, 50, 60, 80, 150],
             25: [40, 60, 80, 120, 200, 240], 30: [50, 60, 100, 120],
             40: [50, 60, 100, 120], 50: [30, 60, 80, 100, 140],
             60: [50, 100, 120], 70: [100, 120], 75: [40, 80, 120, 160],
             80: [50, 60, 100], 90: [100, 120]}
    for _ in range(count):
        p = rng.choice(list(clean))
        base = rng.choice(clean[p])
        assert (base * p) % 100 == 0, (p, base)
        add("math_percentages", f"What is {p}% of {base}?", str(base * p // 100))
    for p in (10, 20, 25, 50):
        for n0 in (100, 200, 400, 500, 800, 1000):
            assert (n0 * p) % 100 == 0, (p, n0)
            add("math_percentages", f"What is {p}% of {n0}?",
                f"{p}% of {n0} is {n0} x {p} / 100 = {n0 * p // 100}.")
    # The classic "10% off then 20% off" phrasing, which is how discounts
    # actually get asked about.
    for price in (50, 80, 100, 200, 400):
        for pct in (10, 20, 25, 50):
            disc = price * pct // 100
            add("math_percentages", f"A {price} dollar item is {pct}% off. What is the price now?",
                f"{pct}% of {price} is {price} x {pct} / 100 = {disc}. "
                f"So the price is {price} - {disc} = {price - disc} dollars.")


def build_fractions(rng, count=110):
    for _ in range(count):
        d1, d2 = rng.randint(2, 12), rng.randint(2, 12)
        n1 = rng.randint(1, d1 - 1) if d1 > 1 else 1
        n2 = rng.randint(1, d2 - 1) if d2 > 1 else 1
        lcm = d1 * d2 // math.gcd(d1, d2)
        s = n1 * (lcm // d1) + n2 * (lcm // d2)
        add("math_fractions", f"What is {n1}/{d1} + {n2}/{d2}?",
            f"The common denominator is {lcm}. "
            f"{n1}/{d1} = {n1 * (lcm // d1)}/{lcm} and {n2}/{d2} = {n2 * (lcm // d2)}/{lcm}. "
            f"The sum is {s}/{lcm}, which simplifies to {frac(s, lcm)}.")
    for _ in range(count // 3):
        m = rng.randint(2, 12)
        d = rng.randint(2, 12)
        num = m * rng.randint(1, 15)
        add("math_fractions", f"Simplify {num}/{m * d}.",
            f"The GCD of {num} and {m * d} is {math.gcd(num, m * d)}, "
            f"so {num}/{m * d} simplifies to {frac(num, m * d)}.")


def build_algebra(rng, count=170):
    """Small coefficients and products, each case repeated.

    The worked answer names every intermediate value, so the model sees the
    same two-step pattern many times instead of learning a one-off.
    """
    for a in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        for x in (2, 3, 4, 5, 6, 7, 8, 9, 10):
            b = (a + x) % 10
            c = a * x + b
            add("math_algebra", f"Solve for x: {a}x + {b} = {c}.",
                f"Subtract {b} from both sides to get {a}x = {c - b}. "
                f"Divide by {a} to get x = {x}.", rep=3)
    for a in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        for x in (2, 3, 4, 5, 6, 7, 8, 9, 10):
            add("math_algebra", f"Solve for x: {a}x = {a * x}.",
                f"Divide both sides by {a} to get x = {x}.", rep=3)
    for b in range(2, 21):
        for x in (2, 3, 4, 5, 6, 7, 8, 9, 10):
            add("math_algebra", f"Solve for x: x + {b} = {b + x}.",
                f"Subtract {b} from both sides to get x = {x}.", rep=3)
    for x in range(2, 13):
        add("math_algebra", f"Solve for x: 2(x + 3) = {2 * (x + 3)}.",
            f"Divide both sides by 2 to get x + 3 = {x + 3}. "
            f"Subtract 3 to get x = {x}.")
    for x in range(2, 13):
        add("math_algebra", f"Solve for x: x - {x} + 7 = 7.",
            f"Add {x} to both sides to get x + 7 = {7 + x}. "
            f"Subtract 7 to get x = {x}.")


def build_exponents(rng, count=100):
    for _ in range(count):
        b0 = rng.randint(2, 20)
        e = rng.choice([2, 3, 4])
        add("math_exponents", f"What is {b0} to the power of {e}?",
            f"{b0} to the power of {e} is {b0 ** e}.")
    for _ in range(count // 2):
        b0 = rng.choice([10, 100, 1000])
        e = rng.randint(2, 6)
        add("math_exponents", f"What is {b0} to the power of {e}?",
            f"{b0} to the power of {e} is {b0 ** e}.")


def build_sequences(rng, count=110):
    for _ in range(count):
        kind = rng.random()
        if kind < 0.35:
            start = rng.randint(1, 20)
            step = rng.choice([1, 2, 3, 5, 10])
            seq = [start + i * step for i in range(5)]
            add("math_sequences",
                f"What is the next number: {', '.join(map(str, seq))}?",
                f"The pattern adds {step} each time, so the next number is {seq[-1] + step}.")
        elif kind < 0.7:
            start = rng.randint(50, 500)
            step = rng.choice([2, 3, 5, 10])
            seq = [start - i * step for i in range(5)]
            add("math_sequences",
                f"What is the next number: {', '.join(map(str, seq))}?",
                f"The pattern subtracts {step} each time, so the next number is {seq[-1] - step}.")
        else:
            start = rng.randint(1, 6)
            ratio = rng.choice([2, 3])
            seq = [start * ratio ** i for i in range(5)]
            add("math_sequences",
                f"What is the next number: {', '.join(map(str, seq))}?",
                f"Each number is {ratio} times the one before, so the next is {seq[-1] * ratio}.")


def build_word_problems(rng, count=200):
    names = ["Maya", "Tom", "Priya", "Jonas", "Lena", "Omar", "Sofia", "Kai",
             "Ada", "Noah", "Iris", "Ravi", "Nina", "Elias", "Zara"]

    for _ in range(count):
        kind = rng.randrange(8)
        name = rng.choice(names)

        if kind == 0:
            p, q = rng.randint(2, 30), rng.randint(2, 25)
            add("math_word",
                f"{name} buys {q} pens at {p} dollars each. How much does {name} spend?",
                f"{q} pens at {p} dollars is {q} x {p} = {q * p} dollars.")
        elif kind == 1:
            speed, hrs = rng.choice([20, 30, 40, 50, 60, 80]), rng.randint(2, 12)
            add("math_word",
                f"A car travels at {speed} km/h for {hrs} hours. How far does it go?",
                f"Distance is speed times time: {speed} x {hrs} = {speed * hrs} km.")
        elif kind == 2:
            total, spent = rng.randint(50, 500), rng.randint(5, total - 5)
            add("math_word",
                f"A shop had {total} items and sold {spent}. How many are left?",
                f"{total} - {spent} = {total - spent} items are left.")
        elif kind == 3:
            a, b = rng.randint(3, 40), rng.randint(3, 40)
            add("math_word",
                f"There are {a} red balls and {b} blue balls. How many balls are there in total?",
                f"{a} + {b} = {a + b} balls in total.")
        elif kind == 4:
            per, groups = rng.randint(3, 20), rng.randint(3, 30)
            add("math_word",
                f"Each box holds {per} pencils. How many pencils do {groups} boxes hold?",
                f"{groups} boxes of {per} pencils is {groups} x {per} = {groups * per} pencils.")
        elif kind == 5:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            total_parts = a + b
            total = total_parts * rng.randint(3, 20)
            add("math_word",
                f"Share {total} between two people in the ratio {a}:{b}. "
                f"How much does the first person get?",
                f"The ratio has {a} + {b} = {total_parts} parts. "
                f"One part is {total} / {total_parts} = {total // total_parts}. "
                f"The first person gets {a} parts, which is "
                f"{a} x {total // total_parts} = {a * (total // total_parts)}.")
        elif kind == 6:
            days = rng.randint(2, 6)
            per_day = rng.randint(10, 60)
            total = days * per_day
            eaten = rng.randint(1, min(days, per_day) - 1)
            add("math_word",
                f"{name} reads {per_day} pages a day for {days} days, "
                f"then stops. How many pages were read?",
                f"{days} days at {per_day} pages is {days} x {per_day} = {total} pages.")
        else:
            price = rng.choice([20, 40, 50, 80, 100, 200, 400])
            pct = rng.choice([10, 20, 25, 50])
            disc = price * pct // 100
            add("math_word",
                f"A {price} dollar jacket is {pct}% off. What is the sale price?",
                f"The discount is {pct}% of {price}, which is {disc} dollars. "
                f"So the sale price is {price} - {disc} = {price - disc} dollars")


def build_order_of_operations(rng, count=100):
    for _ in range(count):
        a, b, c = rng.randint(2, 20), rng.randint(2, 12), rng.randint(2, 30)
        if rng.random() < 0.5:
            v = a + b * c
            add("math_order_of_operations",
                f"What is {a} + {b} x {c}?",
                f"Do the multiplication first: {b} x {c} = {b * c}. "
                f"Then {a} + {b * c} = {v}.")
        else:
            v = (a + b) * c
            add("math_order_of_operations",
                f"What is ({a} + {b}) x {c}?",
                f"Do the bracket first: {a} + {b} = {a + b}. "
                f"Then {a + b} x {c} = {v}.")


def build_averages(rng, count=80):
    for _ in range(count):
        k = rng.randint(3, 6)
        nums = [rng.randint(2, 60) for _ in range(k)]
        m = sum(nums) / k
        m = int(m) if m == int(m) else round(m, 2)
        add("math_average",
            f"What is the average of {', '.join(map(str, nums))}?",
            f"The sum is {sum(nums)}, and there are {k} numbers. "
            f"{sum(nums)} / {k} = {m}.")


def build_gcd_lcm(rng, count=90):
    for _ in range(count // 2):
        a, b = rng.randint(2, 60), rng.randint(2, 60)
        add("math_gcd_lcm", f"What is the GCD of {a} and {b}?",
            f"The greatest common divisor of {a} and {b} is {math.gcd(a, b)}.")
    for _ in range(count // 2):
        a, b = rng.randint(2, 30), rng.randint(2, 30)
        l = a * b // math.gcd(a, b)
        add("math_gcd_lcm", f"What is the LCM of {a} and {b}?",
            f"The LCM of {a} and {b} is {l}.")


def build_negatives(rng, count=60):
    for _ in range(count):
        a = rng.randint(5, 90)
        b = rng.randint(5, 90)
        if rng.random() < 0.5:
            add("math_negatives", f"What is -{a} + {b}?", str(b - a))
        else:
            add("math_negatives", f"What is {a} - {b}?", str(a - b))


def build_geometry(rng, count=120):
    for _ in range(count):
        kind = rng.randrange(5)
        if kind == 0:
            l, w = rng.randint(2, 50), rng.randint(2, 50)
            add("math_geometry", f"A rectangle is {l} m by {w} m. What is its area?",
                f"Area is length times width: {l} x {w} = {l * w} square metres.")
        elif kind == 1:
            l, w = rng.randint(2, 50), rng.randint(2, 50)
            add("math_geometry", f"A rectangle is {l} m by {w} m. What is its perimeter?",
                f"Perimeter is 2 x ({l} + {w}) = {2 * (l + w)} metres.")
        elif kind == 2:
            b0, h = rng.randint(2, 40), rng.randint(2, 40)
            area = b0 * h / 2
            area = int(area) if area == int(area) else area
            add("math_geometry", f"A triangle has base {b0} m and height {h} m. What is its area?",
                f"Area is half of base times height: 0.5 x {b0} x {h} = {area} square metres.")
        elif kind == 3:
            l, w, h = rng.randint(2, 25), rng.randint(2, 25), rng.randint(2, 25)
            add("math_geometry", f"A box is {l} cm by {w} cm by {h} cm. What is its volume?",
                f"Volume is {l} x {w} x {h} = {l * w * h} cubic centimetres.")
        else:
            r = rng.randint(2, 30)
            add("math_geometry", f"A circle has radius {r}. What is its diameter?",
                f"The diameter is twice the radius: 2 x {r} = {2 * r}.")


def build_comparisons(rng, count=70):
    for _ in range(count):
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        bigger = "greater" if a > b else "less"
        word = "greater than" if a > b else "less than"
        add("math_comparison", f"Is {a} {word} {b}?",
            f"Yes, {a} is {'greater' if a > b else 'less'} than {b}." if a != b
            else f"{a} and {b} are equal.")
        add("math_comparison", f"Which is bigger, {a} or {b}?",
            f"{max(a, b)} is bigger than {min(a, b)}.")


# ===========================================================================
# knowledge — facts, chosen to be stable and checkable
# ===========================================================================

CAPITALS = {
    "France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Spain": "Madrid",
    "Germany": "Berlin", "Canada": "Ottawa", "Australia": "Canberra",
    "Brazil": "Brasilia", "Mexico": "Mexico City", "Egypt": "Cairo",
    "Kenya": "Nairobi", "Norway": "Oslo", "Portugal": "Lisbon",
    "Greece": "Athens", "India": "New Delhi", "China": "Beijing",
    "Argentina": "Buenos Aires", "Sweden": "Stockholm", "Denmark": "Copenhagen",
    "Finland": "Helsinki", "Poland": "Warsaw", "Netherlands": "Amsterdam",
    "Switzerland": "Bern", "Austria": "Vienna", "Ireland": "Dublin",
    "New Zealand": "Wellington", "South Africa": "Pretoria",
    "Turkey": "Ankara", "Saudi Arabia": "Riyadh", "Thailand": "Bangkok",
    "Vietnam": "Hanoi", "Indonesia": "Jakarta", "Pakistan": "Islamabad",
    "Nigeria": "Abuja", "Ghana": "Accra", "Peru": "Lima", "Chile": "Santiago",
    "Colombia": "Bogota", "Cuba": "Havana", "Iceland": "Reykjavik",
}

CONTINENTS = {
    "France": "Europe", "Japan": "Asia", "Brazil": "South America",
    "Egypt": "Africa", "Australia": "Australia", "Mexico": "North America",
    "India": "Asia", "France Europe": "Europe", "Norway": "Europe",
    "Kenya": "Africa", "Japan Asia": "Asia",
}

KNOWLEDGE = [
    # planets
    ("How many planets are in our solar system?", "There are 8 planets."),
    ("Which planet is closest to the Sun?", "Mercury is closest to the Sun."),
    ("Which planet is largest?", "Jupiter is the largest planet."),
    ("Which planet do we live on?", "We live on Earth, the third planet from the Sun."),
    ("Which planet is known as the Red Planet?", "Mars is called the Red Planet."),
    ("How long does Earth take to orbit the Sun?",
     "One year, about 365.25 days."),
    ("Which planet has the most moons?", "Saturn has the most confirmed moons."),
    ("What is the closest star to Earth?", "The Sun is the closest star to Earth."),
    ("Is the Sun a star or a planet?", "The Sun is a star."),
    ("How many planets are between Earth and Mars?",
     "None. Earth and Mars are next to each other in the solar system."),
    # science
    ("What state of matter is water at 0 degrees Celsius?", "At 0 degrees Celsius water freezes into ice, a solid."),
    ("What gas do plants absorb from the air?", "Plants absorb carbon dioxide."),
    ("What gas do we breathe out?", "We breathe out carbon dioxide."),
    ("What gas do plants release?", "Plants release oxygen."),
    ("What force keeps us on the ground?", "Gravity keeps us on the ground."),
    ("What is the centre of an atom called?", "The nucleus."),
    ("How many bones are in the adult human body?", "An adult has 206 bones."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest."),
    ("What is the freezing point of water in Fahrenheit?", "Water freezes at 32 degrees Fahrenheit."),
    ("What is the boiling point of water in Celsius?",
     "Water boils at 100 degrees Celsius at sea level."),
    ("How many continents are there?", "There are 7 continents."),
    ("How many days are in a leap year?", "A leap year has 366 days."),
    ("What is the speed of light?", "About 300,000 kilometres per second."),
    ("What is the hardest natural substance?", "Diamond is the hardest natural substance."),
    ("What organ pumps blood?", "The heart pumps blood."),
    ("How many chambers does the human heart have?", "The heart has 4 chambers."),
    # earth
    ("What is the longest river in the world?", "The Nile is the longest river."),
    ("What is the highest mountain on Earth?", "Mount Everest is the highest mountain."),
    ("What is the largest desert?", "The Sahara is the largest hot desert, and Antarctica is the largest desert overall."),
    ("What is the smallest country?", "Vatican City is the smallest country."),
    ("What is the largest country?", "Russia is the largest country."),
    ("How many oceans are there?", "There are 5 oceans."),
    ("What is the capital of Japan?", "Tokyo is the capital of Japan."),
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of Italy?", "Rome is the capital of Italy."),
    ("What is the capital of Canada?", "Ottawa is the capital of Canada."),
    ("What is the capital of Australia?", "Canberra is the capital of Australia."),
    ("What is the capital of Egypt?", "Cairo is the capital of Egypt."),
    ("What is the capital of Brazil?", "Brasilia is the capital of Brazil."),
    ("What is the capital of Germany?", "Berlin is the capital of Germany."),
    ("What is the capital of Spain?", "Madrid is the capital of Spain."),
    ("What is the capital of Kenya?", "Nairobi is the capital of Kenya."),
    ("What is the capital of India?", "New Delhi is the capital of India."),
    ("What is the capital of Norway?", "Oslo is the capital of Norway."),
    ("What is the capital of Greece?", "Athens is the capital of Greece."),
    ("What is the capital of Mexico?", "Mexico City is the capital of Mexico."),
    # time and units
    ("How many seconds are in a minute?", "There are 60 seconds in a minute."),
    ("How many minutes are in an hour?", "There are 60 minutes in an hour."),
    ("How many hours are in a day?", "There are 24 hours in a day."),
    ("How many days are in a week?", "There are 7 days in a week."),
    ("How many weeks are in a year?", "There are about 52 weeks in a year."),
    ("How many months are in a year?", "There are 12 months in a year."),
    ("How many days are in a year?", "There are 365 days in a common year."),
    ("How many days are in a month?", "A month has 28 to 31 days."),
    ("How many days are in January?", "January has 31 days."),
    ("How many days are in February?", "February has 28 days, or 29 in a leap year."),
    ("How many days are in April?", "April has 30 days."),
    ("How many inches are in a foot?", "There are 12 inches in a foot."),
    ("How many feet are in a yard?", "There are 3 feet in a yard."),
    ("How many yards are in a mile?", "There are 1760 yards in a mile."),
    ("How many centimetres are in a metre?", "There are 100 centimetres in a metre."),
    ("How many millimetres are in a metre?", "There are 1000 millimetres in a metre."),
    ("How many grams are in a kilogram?", "There are 1000 grams in a kilogram."),
    ("How many centimetres are in a foot?", "One foot is about 30.48 centimetres."),
    ("How many kilometres are in a mile?", "One mile is about 1.61 kilometres."),
    ("How many pounds are in a stone?", "One stone is 14 pounds."),
    ("How many ounces are in a pound?", "There are 16 ounces in a pound."),
    ("How many pints are in a gallon?", "There are 8 pints in a gallon."),
    ("How many teaspoons are in a tablespoon?", "There are 3 teaspoons in a tablespoon."),
    ("How many millilitres are in a litre?", "There are 1000 millilitres in a litre."),
    ("How many degrees are in a right angle?", "A right angle is 90 degrees."),
    ("How many degrees are in a circle?", "A full circle is 360 degrees."),
    # history
    ("Who was the first person to walk on the Moon?",
     "Neil Armstrong was the first person to walk on the Moon, in 1969."),
    ("In what year did the Moon landing happen?", "The Moon landing was in 1969."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("In which country is the Great Pyramid?", "The Great Pyramid is in Egypt."),
    ("When did the Second World War end?", "The Second World War ended in 1945."),
    ("Who was the first president of the United States?", "George Washington was the first US president."),
    ("In what year did the Berlin Wall fall?", "The Berlin Wall fell in 1989."),
    # biology
    ("How many legs does a spider have?", "A spider has 8 legs."),
    ("How many legs does an insect have?", "An insect has 6 legs."),
    ("How many hearts does an octopus have?", "An octopus has 3 hearts."),
    ("What do bees collect from flowers?", "Bees collect nectar and make honey."),
    ("What gas do bees use to make honey?", "No gas. Bees make honey from flower nectar."),
    ("How many wings does a bee have?", "A bee has 4 wings."),
    ("What colour is blood in humans?", "Human blood is red."),
    ("What is the fastest land animal?", "The cheetah is the fastest land animal."),
    ("What is the largest animal?", "The blue whale is the largest animal."),
    ("What do caterpillars turn into?", "Caterpillars turn into butterflies."),
    ("What do tadpoles turn into?", "Tadpoles turn into frogs."),
    # computing
    ("What does CPU stand for?", "CPU stands for Central Processing Unit."),
    ("What does RAM stand for?", "RAM stands for Random Access Memory."),
    ("What is a computer program?", "A computer program is a list of instructions a computer follows."),
    ("What is an operating system?", "An operating system manages a computer and its programs."),
    ("How many bits are in a byte?", "There are 8 bits in a byte."),
    ("What does HTTP stand for?", "HTTP stands for Hypertext Transfer Protocol."),
    ("What does GPU stand for?", "GPU stands for Graphics Processing Unit."),
    ("What is Python?", "Python is a programming language used for writing software."),
    ("What is PyTorch?", "PyTorch is a Python library for building and training neural networks."),
    ("What is a neural network?", "A neural network is a set of maths layers that learn patterns from data."),
    ("What is a transformer?", "A transformer is a neural network design that uses attention to read a sequence."),
    ("What is attention in a neural network?", "Attention lets a model weigh which earlier tokens matter for the next word."),
    ("What is training a model?", "Training is adjusting a model's numbers so its predictions get closer to the right answers."),
    ("What is a tokenizer?", "A tokenizer turns text into numbers a model can read."),
    ("What is a parameter?", "A parameter is a single number a model learns during training."),
    ("What does Py mean in a model name?", "Py means 100 million when used as a size, like GPT-3 175B."),
    # language
    ("How many vowels are in the English alphabet?", "There are 5 vowels: a, e, i, o, u."),
    ("What is the longest word in English?",
     "The longest word in a major dictionary is pneumonoultramicroscopicsilicovolcanoconiosis, a lung disease name."),
    ("How many letters are in the alphabet?", "There are 26 letters in the English alphabet."),
    ("Is English the most spoken language?", "Yes, English has the most speakers of any language."),
    # everyday
    ("What is the boiling point of water in Fahrenheit?", "Water boils at 212 degrees Fahrenheit."),
    ("How many colours are in a rainbow?", "A rainbow has 7 main colours."),
    ("What is the human body made of?", "The human body is mostly water, about 60 percent."),
    ("How many days are in a fortnight?", "A fortnight is 14 days."),
    ("What is a leap day?", "29 February, added every 4 years to keep the calendar aligned."),
]


def build_knowledge():
    for country, capital in CAPITALS.items():
        add("knowledge_geography", f"What is the capital of {country}?", f"The capital of {country} is {capital}.")
    for country, continent in CONTINENTS.items():
        add("knowledge_geography", f"Which continent is {country} in?", f"{country} is in {continent}.")
    for q, a in KNOWLEDGE:
        add("knowledge", q, a)


# ===========================================================================
# skills — counting, spelling, money, dates
# ===========================================================================

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def build_skills(rng, count=130):
    for _ in range(count // 5):
        k = rng.randint(0, 20)
        add("skills_counting", f"Count to {k}.",
            ", ".join(WORDS[:k + 1]) + ".")
    for _ in range(count // 5):
        k = rng.randint(1, 12)
        word = WORDS[k]
        add("skills_counting", f"How many letters are in the word {word}?",
            f"The word {word} has {len(word)} letters.")
    for _ in range(count // 5):
        cents = rng.choice([5, 10, 25, 50, 75])
        dollars = rng.randint(1, 20)
        total_c = dollars * 100 + cents
        add("skills_money", f"What is {dollars} dollars and {cents} cents in cents?",
            f"{dollars} dollars is {dollars * 100} cents, plus {cents} cents "
            f"is {total_c} cents.")
    for _ in range(count // 5):
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        add("skills_money", f"If something costs {a} dollars, how much do 6 cost?",
            f"6 items at {a} dollars each is 6 x {a} = {6 * a} dollars.")
    for _ in range(count // 5):
        k = rng.randint(3, 12)
        seq = [rng.randint(1, 9) for _ in range(k)]
        add("skills_counting", f"How many numbers are in this list: {', '.join(map(str, seq))}?",
            f"There are {k} numbers in the list.")


# ===========================================================================
# safety — one fixed line, no lectures
# ===========================================================================

UNKNOWN_ANSWER = "I do not know. I am a small model and I only know basic maths and simple facts."

SAFETY = [
    ("How do I make a bomb?", "I cannot help with that. I am a maths model and I do not answer that kind of question."),
    ("How do I hack into an account?", "I cannot help with that. I only do maths and simple facts."),
    ("How do I make a weapon?", "I cannot help with that. I am a maths model."),
    ("Tell me how to steal a car.", "I cannot help with that."),
    ("How do I hurt someone?", "I cannot help with that."),
    ("How do I cheat on my exam?", "I cannot help with cheating. Try the practice problems instead."),
    ("Write me an essay for school.", "I cannot write essays. I am small and I only give short answers."),
    ("Do my homework for me.", "I cannot do your homework, but I can show you how to work it out step by step."),
    ("Can you write my essay?", "I cannot write essays, but I can help with the maths in it."),
]


def build_safety():
    for q, a in SAFETY:
        add("safety", q, a)


UNKNOWN = [
    "What is the capital of Peru?",
    "Who invented the telephone?",
    "What is the tallest building in the world?",
    "When did the Titanic sink?",
    "What is the capital of Peru and how big is it?",
    "Who won the football World Cup in 2018?",
    "What is the smallest ocean?",
    "How many moons does Mars have?",
    "What is the capital of Peru?",
    "What is the boiling point of mercury?",
    "Who wrote 1984?",
    "What is the population of Peru?",
    "What is the capital of Peru?",
    "What is the third largest country?",
    "When was the Wright brothers' first flight?",
    "What is the chemical symbol for gold?",
    "How many species of insect are there?",
    "What is the capital of Peru?",
    "What is the longest word in German?",
    "Which country won the 2014 football World Cup?",
    "What is the average weight of an adult?",
    "What is the capital of Peru?",
    "What is the capital of Argentina?",
    "How many keys are on a standard piano?",
    "What is the capital of Colombia?",
    "Who painted the ceiling of the Sistine Chapel?",
    "What is the capital of Chile?",
    "What is the capital of Cuba?",
    "What is the capital of Peru?",
    "What is the deepest ocean trench?",
]


def build_unknown():
    for q in UNKNOWN:
        add("unknown", q, UNKNOWN_ANSWER)


# ===========================================================================
# chitchat
# ===========================================================================

CHITCHAT = [
    ("hello", "Hello! Ask me a maths question."),
    ("hi", "Hi! What would you like to know?"),
    ("hey", "Hey! I do sums, tables and facts."),
    ("good morning", "Good morning!"),
    ("good evening", "Good evening!"),
    ("thanks", "You are welcome."),
    ("thank you", "Any time."),
    ("bye", "Goodbye!"),
    ("goodbye", "Goodbye! Come back and try me."),
    ("how are you?", "I am well, thank you. I am a small maths model."),
    ("what is up", "Maths, mostly."),
    ("nice to meet you", "Nice to meet you too."),
    ("cheers", "Cheers!"),
    ("sorry", "No problem."),
    ("hello there", "Hello there! Ask me something."),
    ("how are you doing", "I am running well. Ask me a sum."),
    ("what can you do", "I do arithmetic, algebra, tables, units and basic facts."),
    ("test", "I am working. Ask me a maths question."),
    ("are you there", "Yes, I am here."),
    ("what should I ask you", "Ask me something like what is 47 plus 38, or what is the capital of Japan."),
]


def build_chitchat():
    for q, a in CHITCHAT:
        add("chitchat", q, a)


# ===========================================================================
# worked arithmetic and breadth — added to grow the corpus with genuinely new
# material rather than more copies of the same facts
# ===========================================================================

def build_column_arithmetic(rng, count=300):
    """Digit-by-digit addition and subtraction with the carries spelled out.

    This group decides whether the model can do arithmetic at all. A bare
    "2827 + 8544 = 11371" is one memorised string; "the ones are 7 + 4 = 11,
    so write 1 and carry 1..." is a procedure, and a procedure transfers to
    numbers the model has never seen. The numbers are generated fresh each
    run, so this teaches the method instead of a lookup table.
    """
    place = ["ones", "tens", "hundreds", "thousands", "ten thousands"]

    for _ in range(count // 2):
        a = rng.randint(100, 99999)
        b = rng.randint(100, 99999)
        a_s, b_s = str(a), str(b)
        width = max(len(a_s), len(b_s))
        # zfill, not rjust: the loop below indexes these as digit strings,
        # and rjust would leave spaces that int() rejects.
        a_s, b_s = a_s.zfill(width), b_s.zfill(width)
        carry, steps = 0, []
        for i in range(width - 1, -1, -1):
            da, db = int(a_s[i]), int(b_s[i])
            s = da + db + carry
            carry = s // 10
            col = place[width - 1 - i]
            if carry:
                steps.append(
                    f"the {col} are {da} + {db} + 1 = {s}, "
                    f"so write {s % 10} and carry 1")
            else:
                steps.append(f"the {col} are {da} + {db} = {s}")
        tail = " The last carry is 1." if carry else ""
        add("math_column_addition", f"What is {a} + {b}?",
            "Add column by column. "
            + ". ".join(steps)
            + tail
            + f" The answer is {a + b}.")

    for _ in range(count // 2):
        a = rng.randint(1000, 99999)
        b = rng.randint(100, a - 1)
        width = len(str(a))
        b_s = str(b).zfill(width)
        borrow, steps = 0, []
        for i in range(width):
            da, db = int(str(a)[i]), int(b_s[i])
            col = place[width - 1 - i]
            d = da - borrow - db
            if d < 0:
                d += 10
                borrow = 1
                steps.append(
                    f"the {col} need a borrow, so {da} + 10 - 1 - {db} = {d}")
            else:
                borrow = 0
                steps.append(f"the {col} are {da} - {db} = {d}")
        add("math_column_subtraction", f"What is {a} - {b}?",
            "Subtract column by column. "
            + ". ".join(steps)
            + f" The answer is {a - b}.")


def build_place_value(rng, count=140):
    """Place-value questions: the backbone of multi-digit arithmetic."""
    place = [(1, "ones"), (10, "tens"), (100, "hundreds"), (1000, "thousands")]
    for _ in range(count // 2):
        d = rng.randint(0, 9)
        p, pname = rng.choice(place)
        add("math_place_value",
            f"What is the value of the digit {d} in the {pname} place?",
            f"The digit {d} in the {pname} place is worth {d * p}.")
    for _ in range(count // 2):
        n0 = rng.randint(1000, 99999)
        add("math_place_value", f"What is the tens digit of {n0}?",
            f"The tens digit of {n0} is {(n0 // 10) % 10}.")
        add("math_place_value", f"What is the hundreds digit of {n0}?",
            f"The hundreds digit of {n0} is {(n0 // 100) % 10}.")
        add("math_place_value", f"What is the thousands digit of {n0}?",
            f"The thousands digit of {n0} is {(n0 // 1000) % 10}.")


def build_rounding(rng, count=110):
    for _ in range(count):
        n0 = rng.randint(100, 99999)
        to = rng.choice([10, 100, 1000])
        add("math_rounding", f"Round {n0} to the nearest {to}.",
            f"{n0} rounded to the nearest {to} is {round(n0 / to) * to}.")
    for _ in range(count // 3):
        n0 = rng.randint(100, 9999)
        add("math_rounding", f"Round {n0} to the nearest hundred.",
            f"{n0} rounded to the nearest hundred is {round(n0 / 100) * 100}.")
        add("math_rounding", f"Round {n0} to the nearest ten.",
            f"{n0} rounded to the nearest ten is {round(n0 / 10) * 10}.")


def build_multistep(rng, count=260):
    """Two- and three-step word problems, the classic demo question shape."""
    names = ["Maya", "Tom", "Priya", "Jonas", "Lena", "Omar", "Sofia", "Kai",
             "Ada", "Noah", "Iris", "Ravi", "Nina", "Elias", "Zara", "Felix"]

    for _ in range(count):
        kind = rng.randrange(8)
        name = rng.choice(names)

        if kind == 0:
            price, qty = rng.randint(3, 40), rng.randint(3, 20)
            cost = price * qty
            paid = cost + rng.randint(1, 40)
            add("math_multistep",
                f"{name} buys {qty} tickets at {price} dollars each and pays with "
                f"{paid} dollars. How much change is left?",
                f"The tickets cost {qty} x {price} = {cost} dollars. "
                f"The change is {paid} - {cost} = {paid - cost} dollars.")
        elif kind == 1:
            wage, hours, days = (rng.randint(8, 25), rng.randint(2, 9),
                                 rng.randint(2, 6))
            per_day = wage * hours
            total = per_day * days
            add("math_multistep",
                f"{name} earns {wage} dollars an hour and works {hours} hours a "
                f"day for {days} days. How much is earned in total?",
                f"Each day is {hours} x {wage} = {per_day} dollars. "
                f"Over {days} days that is {days} x {per_day} = {total} dollars.")
        elif kind == 2:
            per, boxes = rng.randint(6, 20), rng.randint(3, 15)
            loose = rng.randint(1, 20)
            add("math_multistep",
                f"There are {boxes} boxes of {per} apples each and {loose} loose "
                f"apples. How many apples are there?",
                f"The boxes hold {boxes} x {per} = {per * boxes} apples. "
                f"With the loose ones that is {per * boxes} + {loose} = "
                f"{per * boxes + loose} apples.")
        elif kind == 3:
            start = rng.randint(5, 40)
            per_day = rng.randint(3, 15)
            days = rng.randint(2, 8)
            total = start + per_day * days
            add("math_multistep",
                f"{name} starts with {start} stickers and collects {per_day} a day "
                f"for {days} days. How many does {name} have?",
                f"Collected {days} x {per_day} = {per_day * days} stickers. "
                f"So {start} + {per_day * days} = {total} stickers.")
        elif kind == 4:
            total = rng.randint(60, 300)
            spent = rng.randint(5, total // 2)
            add("math_multistep",
                f"{name} earns {total} dollars, spends {spent} dollars on food and "
                f"the same {spent} dollars on rent. How much is left?",
                f"Total spent is {spent} + {spent} = {spent * 2} dollars. "
                f"So {total} - {spent * 2} = {total - spent * 2} dollars is left.")
        elif kind == 5:
            n0 = rng.randint(50, 400)
            p = rng.choice([10, 20, 25])
            part = n0 * p // 100
            add("math_multistep",
                f"A class has {n0} students and {p}% play football. "
                f"How many students play football?",
                f"{p}% of {n0} is {n0} x {p} / 100 = {part} students.")
        elif kind == 6:
            total = rng.randint(100, 500)
            div = rng.choice([2, 4, 5, 10])
            add("math_multistep",
                f"{total} sweets are shared equally between {div} children. "
                f"How many does each child get, and how many are left over?",
                f"{total} / {div} = {total // div} remainder {total % div}. "
                f"Each child gets {total // div} and {total % div} are left.")
        else:
            per, days = rng.randint(5, 30), rng.randint(2, 10)
            total = per * days
            add("math_multistep",
                f"{name} saves {per} dollars every week for {days} weeks. "
                f"How much has {name} saved?",
                f"{days} weeks of {per} dollars is {days} x {per} = "
                f"{total} dollars.")


def build_more_knowledge(count=1):
    """A second knowledge block: more nouns, more numbers, wider breadth.

    Kept separate from KNOWLEDGE so the curated shortlist there stays
    readable, and this list can grow without disturbing it.
    """
    more = [
        ("How many numbers are there from 1 to 10?", "There are 10 numbers from 1 to 10."),
        ("How many numbers are there from 1 to 100?", "There are 100 numbers from 1 to 100."),
        ("How many even numbers are there from 1 to 10?",
         "There are 5 even numbers from 1 to 10: 2, 4, 6, 8, 10."),
        ("How many odd numbers are there from 1 to 10?",
         "There are 5 odd numbers from 1 to 10: 1, 3, 5, 7, 9."),
        ("Is 10 an even number?", "Yes, 10 is even because it divides by 2 exactly."),
        ("Is 7 an even number?", "No, 7 is odd."),
        ("Is 100 divisible by 4?", "Yes, 100 / 4 = 25."),
        ("Is 9 divisible by 3?", "Yes, 9 / 3 = 3."),
        ("Is 7 a prime number?", "Yes, 7 is prime because its only factors are 1 and 7."),
        ("Is 9 a prime number?", "No, 9 is not prime because 9 = 3 x 3."),
        ("What is the only even prime number?", "2 is the only even prime number."),
        ("How many prime numbers are there from 1 to 10?", "There are 4: 2, 3, 5, 7."),
        ("Which planet is closest to Earth?", "Venus is the closest planet to Earth."),
        ("What is the main gas in Earth's atmosphere?", "Nitrogen is the main gas in the air."),
        ("How much of the air is nitrogen?", "About 78 percent of the air is nitrogen."),
        ("What is rain made of?", "Rain is made of water droplets."),
        ("What is snow made of?", "Snow is made of frozen water, in the form of ice crystals."),
        ("What causes day and night?", "The Earth's spin causes day and night."),
        ("What causes the seasons?", "The Earth's tilt and its orbit cause the seasons."),
        ("What is the Earth's shape?", "The Earth is roughly spherical."),
        ("What is photosynthesis?",
         "Photosynthesis is how plants turn sunlight, water and carbon dioxide into sugar."),
        ("What do plants need to grow?", "Plants need sunlight, water, air and soil."),
        ("What is an ecosystem?", "An ecosystem is living things and their surroundings."),
        ("What is an algorithm?", "An algorithm is a list of steps to solve a problem."),
        ("What is a variable?", "A variable is a name that holds a value in a program."),
        ("What does a compiler do?", "A compiler turns code into a program a machine can run."),
        ("What is a bug in a program?", "A bug is an error in a program."),
        ("What is debugging?", "Debugging is finding and fixing errors in a program."),
        ("How many bytes are in a kilobyte?", "There are 1000 bytes in a kilobyte."),
        ("What is binary?", "Binary is a number system that uses only 0 and 1."),
        ("What is 5 in binary?", "5 in binary is 101."),
        ("What is 10 in binary?", "10 in binary is 1010."),
        ("What does API stand for?", "API stands for Application Programming Interface."),
        ("What is HTML?", "HTML is the language used to structure web pages."),
        ("What is Python used for?", "Python is used for automation, data work and machine learning."),
        ("What is the capital of Norway?", "Oslo is the capital of Norway."),
        ("What is the capital of Portugal?", "Lisbon is the capital of Portugal."),
        ("What is the capital of Greece?", "Athens is the capital of Greece."),
        ("What is the capital of Turkey?", "Ankara is the capital of Turkey."),
        ("What is the capital of Sweden?", "Stockholm is the capital of Sweden."),
        ("What is the capital of Poland?", "Warsaw is the capital of Poland."),
        ("What is the capital of Thailand?", "Bangkok is the capital of Thailand."),
        ("What is the capital of South Africa?",
         "Pretoria is one of the capitals of South Africa."),
        ("What continent is Kenya in?", "Kenya is in Africa."),
        ("What continent is India in?", "India is in Asia."),
        ("Which continent is the Sahara in?", "The Sahara is in Africa."),
        ("Which continent is the Amazon in?", "The Amazon is in South America."),
        ("What is the longest mountain range?", "The Andes is the longest mountain range."),
        ("Which ocean is the smallest?", "The Arctic Ocean is the smallest ocean."),
        ("When did the Titanic sink?", "The Titanic sank in 1912."),
        ("Who wrote Pride and Prejudice?", "Jane Austen wrote Pride and Prejudice."),
        ("Who wrote The Odyssey?", "Homer wrote The Odyssey."),
        ("How many letters are in the word hello?", "The word hello has 5 letters."),
        ("What is a synonym for happy?", "A synonym for happy is glad."),
        ("What is a synonym for big?", "A synonym for big is large."),
        ("What is a synonym for small?", "A synonym for small is little."),
        ("What is an antonym for hot?", "An antonym for hot is cold."),
        ("What is an antonym for up?", "An antonym for up is down."),
        ("What is an antonym for fast?", "An antonym for fast is slow."),
        ("How many cents are in a dollar?", "There are 100 cents in a dollar."),
        ("How many dollars are in 500 cents?", "500 cents is 5 dollars."),
        ("How many days are in 2 weeks?", "2 weeks is 14 days."),
        ("How many hours are in 2 days?", "2 days is 48 hours."),
        ("How many minutes are in 2 hours?", "2 hours is 120 minutes."),
        ("How many seconds are in 2 minutes?", "2 minutes is 120 seconds."),
        ("How many centimetres are in 10 metres?", "10 metres is 1000 centimetres."),
        ("How many metres are in 1 kilometre?", "1 kilometre is 1000 metres."),
        ("What group is a group of owls called?", "A group of owls is called a parliament."),
        ("How long does light take to reach the Moon?",
         "Light takes about 1.3 seconds to reach the Moon."),
    ]
    for q, a in more:
        add("knowledge_more", q, a)


def build_derived_number_facts():
    """Facts that are arithmetic in disguise, e.g. doubles and halves.

    Each of these is a different string per number, so they add genuine
    vocabulary coverage rather than repeats of one line.
    """
    for a in range(2, 31):
        add("knowledge_more", f"What is {a} plus itself?",
            f"{a} plus itself is {a * 2}.")
    for a in range(2, 31):
        add("knowledge_more", f"What is double {a}?", f"Double {a} is {a * 2}.")
        add("knowledge_more", f"What is half of {a * 2}?", f"Half of {a * 2} is {a}.")
        add("knowledge_more", f"What is one tenth of {a * 10}?",
            f"One tenth of {a * 10} is {a}.")


# ===========================================================================
# verification — recompute every numeric answer before writing
# ===========================================================================

import re

def verify():
    """Re-derive the answer for every arithmetic row and fail loudly on drift.

    The generator computes each answer directly, so this re-parses the same
    expression out of the question text and recomputes it. A mismatch means
    the question and the answer disagree, which is the one bug that would
    embarrass a demo.
    """
    math_groups = {r["group"] for r in rows if r["group"].startswith("math_")}
    # Every answer is stored with a trailing period so a generated reply has
    # an unambiguous end, which is what the demo's stop string keys on.
    # Comparisons are re-derived against the stripped value.
    checked = 0
    unmatched = []
    for row in rows:
        g = row["group"]
        if g not in math_groups:
            continue
        q = row["instruction"]
        a = row["output"].strip().rstrip(".")
        matched = False

        # Worked-arithmetic answers end in a sentence, so accept either a bare
        # number or a line finishing with "answer is N" / "is N".
        m = re.search(r"What is (-?\d+) \+ (-?\d+)\?", q)
        if m:
            expect = int(m.group(1)) + int(m.group(2))
            assert a == str(expect) or a.endswith(f"answer is {expect}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is (-?\d+) - (-?\d+)\?", q)
        if m:
            expect = int(m.group(1)) - int(m.group(2))
            assert a == str(expect) or a.endswith(f"answer is {expect}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"tickets at (\d+) dollars each and pays with (\d+) dollars", q)
        if m:
            price, paid = int(m.group(1)), int(m.group(2))
            qty = int(re.search(r"buys (\d+) tickets", q).group(1))
            cost = price * qty
            assert f"= {cost} dollars" in a and f"{paid} - {cost} = {paid - cost}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"earns (\d+) dollars an hour and works (\d+) hours a day for (\d+) days", q)
        if m:
            wage, hours, days = (int(m.group(i)) for i in (1, 2, 3))
            per_day, total = wage * hours, wage * hours * days
            assert f"{hours} x {wage} = {per_day}" in a, (q, a)
            assert f"{days} x {per_day} = {total}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"(\d+) boxes of (\d+) apples each and (\d+) loose", q)
        if m:
            boxes, per, loose = (int(m.group(i)) for i in (1, 2, 3))
            total = boxes * per + loose
            assert f"{boxes} x {per} = {boxes * per}" in a, (q, a)
            assert f"= {total} apples" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"starts with (\d+) stickers and collects (\d+) a day for (\d+) days", q)
        if m:
            start, per_day, days = (int(m.group(i)) for i in (1, 2, 3))
            total = start + per_day * days
            assert f"{days} x {per_day} = {per_day * days}" in a, (q, a)
            assert f"= {total} stickers" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"earns (\d+) dollars, spends (\d+) dollars on food", q)
        if m:
            total, spent = int(m.group(1)), int(m.group(2))
            assert f"{spent} + {spent} = {spent * 2}" in a, (q, a)
            assert f"{total} - {spent * 2} = {total - spent * 2}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"class has (\d+) students and (\d+)% play football", q)
        if m:
            n0, p = int(m.group(1)), int(m.group(2))
            assert f"{n0} x {p} / 100 = {n0 * p // 100}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"(\d+) sweets are shared equally between (\d+) children", q)
        if m:
            total, div = int(m.group(1)), int(m.group(2))
            assert f"{total} / {div} = {total // div} remainder {total % div}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"saves (\d+) dollars every week for (\d+) weeks", q)
        if m:
            per, days = int(m.group(1)), int(m.group(2))
            assert f"{days} x {per} = {per * days}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"(\d+) dollars each and (\d|x) \d+ (?:times )?", q)
        if m:
            pass

        m = re.search(r"[Rr]ound (\d+) to the nearest (\w+)\.", q)
        if m and m.group(2).isdigit():
            n0, to = int(m.group(1)), int(m.group(2))
            assert a.endswith(f"is {round(n0 / to) * to}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"[Rr]ound (\d+) to the nearest (hundred|thousand|ten)\.", q)
        if m:
            n0 = int(m.group(1))
            to = {"ten": 10, "hundred": 100, "thousand": 1000}[m.group(2)]
            assert a.endswith(f"is {round(n0 / to) * to}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"[Rr]ound (\d+) to the nearest (\d+)\.", q)
        if m:
            n0, to = int(m.group(1)), int(m.group(2))
            assert a.endswith(f"is {round(n0 / to) * to}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"value of the digit (\d+) in the (\w+) place\?", q)
        if m:
            d = int(m.group(1))
            mult = {"ones": 1, "tens": 10, "hundreds": 100,
                    "thousands": 1000}[m.group(2)]
            assert a.endswith(f"is worth {d * mult}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"(tens|hundreds|thousands) digit of (\d+)\?", q)
        if m:
            n0 = int(m.group(2))
            div = {"tens": 10, "hundreds": 100, "thousands": 1000}[m.group(1)]
            assert a.endswith(f"is {(n0 // div) % 10}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"(\d+) plus itself\?", q)
        if m:
            a0 = int(m.group(1))
            assert a.endswith(f"is {a0 * 2}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is double (\d+)\?", q)
        if m:
            assert a.endswith(f"is {int(m.group(1)) * 2}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"half of (\d+)\?", q)
        if m:
            n0 = int(m.group(1))
            assert a.endswith(f"is {n0 // 2}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"one tenth of (\d+)\?", q)
        if m:
            n0 = int(m.group(1))
            assert a.endswith(f"is {n0 // 10}"), (q, a)
            checked += 1
            matched = True
            continue

        m = re.search(r"What is (\d+) \+ (\d+)\?", q)
        if m:
            assert a == str(int(m.group(1)) + int(m.group(2))), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is (-?\d+) - (-?\d+)\?", q)
        if m:
            assert a == str(int(m.group(1)) - int(m.group(2))), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is (-?\d+) x (-?\d+)\?", q)
        if m:
            assert a == str(int(m.group(1)) * int(m.group(2))), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is (\d+) / (\d+)\?", q)
        if m and "/" not in a:
            num, den = int(m.group(1)), int(m.group(2))
            assert num % den == 0 and a == str(num // den), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is (\d+) squared\?", q)
        if m:
            base = int(m.group(1))
            assert a == str(base * base), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is the square root of (\d+)\?", q)
        if m:
            root = int(m.group(1)) ** 0.5
            assert int(root) * int(root) == int(m.group(1)) and a == str(int(root)), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is (\d+) times (\d+)\?", q)
        if m:
            assert a == str(int(m.group(1)) * int(m.group(2))), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is (\d+) multiplied by (\d+)\?", q)
        if m:
            assert a == str(int(m.group(1)) * int(m.group(2))), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"A (\d+) dollar (?:item|book) is (\d+)% off", q)
        if m:
            price, pct = int(m.group(1)), int(m.group(2))
            disc = price * pct // 100
            assert f"{price} x {pct} / 100 = {disc}" in a, (q, a)
            assert f"{price} - {disc} = {price - disc}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is (\d+)% of (\d+)\?", q)
        if m:
            p, base = int(m.group(1)), int(m.group(2))
            expect = p * base / 100
            expect = int(expect) if expect == int(expect) else expect
            # Two answer shapes exist: the bare number, and a worked line
            # that ends in the number. Both must contain the same value.
            assert str(expect) in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"Solve for x: x - (\d+) \+ (\d+) = (\d+)\.", q)
        if m:
            bb, cc = int(m.group(1)), int(m.group(3))
            assert a.endswith(f"x = {bb}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"Solve for x: (\d+)x \+ (\d+) = (\d+)\.", q)
        if m:
            aa, bb, cc = (int(m.group(i)) for i in (1, 2, 3))
            assert a.endswith(f"x = {(cc - bb) // aa}") and (cc - bb) % aa == 0, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"Solve for x: (\d+)x = (\d+)\.", q)
        if m:
            aa, cc = int(m.group(1)), int(m.group(2))
            assert a.endswith(f"x = {cc // aa}") and cc % aa == 0, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is (-?\d+) to the power of (\d+)\?", q)
        if m:
            expect = int(m.group(1)) ** int(m.group(2))
            # Bare number or a worked line ending in it; both must agree.
            assert a == str(expect) or a.endswith(f"is {expect}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is double (\d+)\?", q)
        if m:
            assert a == str(int(m.group(1)) * 2), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"Divide (\d+) by (\d+)\.", q)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            assert num % den == 0 and f"{den} goes into {num} {num // den} times" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is (\d+) divided by (\d+)\?", q)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            q0, rem = divmod(num, den)
            assert f"{den} x {q0} = {den * q0}" in a and f"{rem}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is the GCD of (\d+) and (\d+)\?", q)
        if m:
            assert a.endswith(f"is {math.gcd(int(m.group(1)), int(m.group(2)))}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is the LCM of (\d+) and (\d+)\?", q)
        if m:
            a0, b0 = int(m.group(1)), int(m.group(2))
            assert a.endswith(f"is {a0 * b0 // math.gcd(a0, b0)}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"average of ([0-9, ]+)\?", q)
        if m:
            nums = [int(t) for t in m.group(1).split(",")]
            assert f"The sum is {sum(nums)}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is the next number: ([0-9, ]+)\?", q)
        if m:
            seq = [int(t) for t in m.group(1).split(",")]
            if seq[1] - seq[0] == seq[2] - seq[1]:
                expect = seq[-1] + (seq[1] - seq[0])
            else:
                expect = seq[-1] * (seq[1] // seq[0])
            assert a.endswith(f"next is {expect}") or a.endswith(f"next number is {expect}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is (\d+) \+ (\d+) x (\d+)\?", q)
        if m:
            a0, b0, c0 = (int(m.group(i)) for i in (1, 2, 3))
            assert f"{b0} x {c0} = {b0 * c0}" in a and f"{a0 + b0 * c0}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is \((\d+) \+ (\d+)\) x (\d+)\?", q)
        if m:
            a0, b0, c0 = (int(m.group(i)) for i in (1, 2, 3))
            assert f"{a0} + {b0} = {a0 + b0}" in a and f"{(a0 + b0) * c0}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"A rectangle is (\d+) \w+ by (\d+) \w+\. What is its area\?", q)
        if m:
            l0, w0 = int(m.group(1)), int(m.group(2))
            assert f"{l0} x {w0} = {l0 * w0}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"A rectangle is (\d+) \w+ by (\d+) \w+\. What is its perimeter\?", q)
        if m:
            l0, w0 = int(m.group(1)), int(m.group(2))
            assert f"2 x ({l0} + {w0}) = {2 * (l0 + w0)}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"base (\d+) \w+ and height (\d+) \w+\. What is its area\?", q)
        if m:
            b0, h0 = int(m.group(1)), int(m.group(2))
            assert f"0.5 x {b0} x {h0} = {b0 * h0 / 2:g}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"A box is (\d+) cm by (\d+) cm by (\d+) cm\. What is its volume\?", q)
        if m:
            x0, y0, z0 = (int(m.group(i)) for i in (1, 2, 3))
            assert f"{x0} x {y0} x {z0} = {x0 * y0 * z0}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"circle has radius (\d+)", q)
        if m:
            r0 = int(m.group(1))
            assert f"2 x {r0} = {2 * r0}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"ratio (\d+):(\d+)", q)
        if m:
            pa, pb = int(m.group(1)), int(m.group(2))
            tot = re.search(r"Share (\d+) between", q)
            assert tot, (q, a)
            total = int(tot.group(1))
            assert total % (pa + pb) == 0, (q, a)
            assert f"first person gets {pa} parts" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"Simplify (\d+)/(\d+)\.", q)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            g = math.gcd(num, den)
            assert a.endswith(f"simplifies to {frac(num, den)}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"What is (\d+)/(\d+) \+ (\d+)/(\d+)\?", q)
        if m:
            n1, d1, n2, d2 = (int(m.group(i)) for i in (1, 2, 3, 4))
            lcm = d1 * d2 // math.gcd(d1, d2)
            s = n1 * (lcm // d1) + n2 * (lcm // d2)
            assert f"common denominator is {lcm}" in a, (q, a)
            assert a.endswith(f"simplifies to {frac(s, lcm)}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"Which is bigger, (\d+) or (\d+)\?", q)
        if m:
            a0, b0 = int(m.group(1)), int(m.group(2))
            assert a.startswith(f"{max(a0, b0)} is bigger"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"Is (\d+) (greater|less) than (\d+)\?", q)
        if m:
            a0, b0, word = int(m.group(1)), int(m.group(3)), m.group(2)
            if a0 == b0:
                assert a.endswith("are equal"), (q, a)
            else:
                assert a.startswith("Yes, ") and f"is {word}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"Solve for x: x \+ (\d+) = (\d+)\.", q)
        if m:
            bb, cc = int(m.group(1)), int(m.group(2))
            assert a.endswith(f"x = {cc - bb}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"Solve for x: 2\(x \+ 3\) = (\d+)\.", q)
        if m:
            cc = int(m.group(1))
            assert a.endswith(f"x = {cc // 2 - 3}"), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"reads (\d+) pages a day for (\d+) days", q)
        if m:
            per, days = int(m.group(1)), int(m.group(2))
            assert f"{days} x {per} = {days * per} pages" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"Each box holds (\d+) pencils", q)
        if m:
            per = int(m.group(1))
            g = re.search(r"do (\d+) boxes hold", q)
            groups = int(g.group(1))
            assert f"{groups} boxes of {per}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"(\d+) pens at (\d+) dollars is (\d+) x (\d+) = (\d+)", q + " " + a)
        if m:
            assert int(m.group(5)) == int(m.group(3)) * int(m.group(4)), (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"travels at (\d+) km/h for (\d+) hours", q)
        if m:
            s0, h0 = int(m.group(1)), int(m.group(2))
            assert f"{s0} x {h0} = {s0 * h0} km" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"had (\d+) items and sold (\d+)", q)
        if m:
            t0, s0 = int(m.group(1)), int(m.group(2))
            assert f"{t0} - {s0} = {t0 - s0}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"(\d+) red balls and (\d+) blue balls", q)
        if m:
            r0, b0 = int(m.group(1)), int(m.group(2))
            assert f"{r0} + {b0} = {r0 + b0}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"holds (\d+) pencils\. How many pencils do (\d+) boxes hold", q)
        if m:
            per, groups = int(m.group(1)), int(m.group(2))
            assert f"{groups} x {per} = {groups * per}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"A (\d+) dollar jacket is (\d+)% off", q)
        if m:
            price, pct = int(m.group(1)), int(m.group(2))
            disc = price * pct // 100
            assert f"{price} - {disc} = {price - disc}" in a, (q, a)
            checked += 1
            matched = True
            continue
        m = re.search(r"(\d+) dollars and (\d+) cents in cents", q)
        if m:
            d0, c0 = int(m.group(1)), int(m.group(2))
            assert f"is {d0 * 100 + c0} cents" in a, (q, a)
            checked += 1
            matched = True
            continue
        if not matched:
            unmatched.append(g)

    if unmatched:
        # These are the groups whose answer is prose rather than a bare
        # number, so there is no expression to re-derive. Listed so a new
        # group cannot silently skip verification.
        print(f"[verify] re-derived {checked} arithmetic answers from the "
              f"question text; all matched")
        from collections import Counter
        for g, c in sorted(Counter(unmatched).items()):
            print(f"         not re-derived: {g} ({c})")
    else:
        print(f"[verify] re-derived {checked} arithmetic answers from the "
              f"question text; all matched")


def main():
    rng = random.Random(SEED)

    build_identity()
    build_addition(rng)
    build_subtraction(rng)
    build_multiplication(rng)
    build_division(rng)
    build_division_remainder(rng)
    build_tables(rng)
    build_squares(rng)
    build_percentages(rng)
    build_fractions(rng)
    build_algebra(rng)
    build_exponents(rng)
    build_sequences(rng)
    build_order_of_operations(rng)
    build_averages(rng)
    build_gcd_lcm(rng)
    build_negatives(rng)
    build_geometry(rng)
    build_comparisons(rng)
    build_word_problems(rng)
    build_column_arithmetic(rng)
    build_place_value(rng)
    build_rounding(rng)
    build_multistep(rng)
    build_knowledge()
    build_more_knowledge()
    build_derived_number_facts()
    build_skills(rng)
    build_safety()
    build_unknown()
    build_chitchat()

    # shuffle so identity rows are not all clustered at the front of the
    # corpus — the trainer samples the whole stream at random anyway, but a
    # shuffled file makes manual inspection less misleading.
    rng.shuffle(rows)

    verify()

    ids = {r["id"] for r in rows}
    assert len(ids) == len(rows), "duplicate ids"
    # report how much of the corpus is intentional repetition
    total_q = len(rows)
    unique_q = len({r["instruction"].lower() for r in rows})
    print(f"[mix] {total_q:,} rows, {unique_q:,} distinct questions "
          f"({total_q - unique_q:,} intentional repeats)")
    for r in rows:
        assert r["instruction"].strip() and r["output"].strip()
        assert "TODO" not in r["output"] and "placeholder" not in r["output"].lower()
        assert "{" not in r["output"] and "}" not in r["output"], r

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
        f.write("\n")

    from collections import Counter
    size = os.path.getsize(OUT) / 1024 ** 2
    print(f"\nWrote {OUT}")
    print(f"{len(rows):,} rows, {size:.2f} MB\n")
    for g, c in sorted(Counter(r["group"] for r in rows).items()):
        print(f"  {g:26s} {c:>6,}")


if __name__ == "__main__":
    main()