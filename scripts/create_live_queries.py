"""Generates eval/live_queries.jsonl containing 40 simulated live queries.

Distribution:
  - 28 in-domain queries (FAST-NUCES academic rules, fees, attendance, grading, registration)
  - 12 out-of-domain queries (hostel, sports, canteen, Wi-Fi, transport, library hours)

Format per line:
  {"id": "lq01", "query": "...", "expected_domain": "in" | "out"}
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_FILE = HERE.parent / "eval" / "live_queries.jsonl"

LIVE_QUERIES = [
    # -------------------------------------------------------------------------
    # In-Domain Queries (28)
    # -------------------------------------------------------------------------
    {"id": "lq01", "query": "What happens if my attendance in CS101 falls below 80 percent?", "expected_domain": "in"},
    {"id": "lq02", "query": "Is attendance compulsory for laboratory sessions and tutorials?", "expected_domain": "in"},
    {"id": "lq03", "query": "Can the teacher condone attendance if I submit a hospital medical certificate?", "expected_domain": "in"},
    {"id": "lq04", "query": "Does an FA grade affect my cumulative GPA the same as an F grade?", "expected_domain": "in"},
    {"id": "lq05", "query": "What is the tuition cost per credit hour for software engineering undergraduate students?", "expected_domain": "in"},
    {"id": "lq06", "query": "What is the penalty fee if I pay my semester dues after the bank due date?", "expected_domain": "in"},
    {"id": "lq07", "query": "Can undergraduate students pay their semester tuition fee in monthly installments?", "expected_domain": "in"},
    {"id": "lq08", "query": "How much fee is refunded if I drop all courses in the first week of classes?", "expected_domain": "in"},
    {"id": "lq09", "query": "What is the minimum CGPA required to avoid being placed on first academic warning?", "expected_domain": "in"},
    {"id": "lq10", "query": "If I am on academic warning, how many credit hours am I permitted to register?", "expected_domain": "in"},
    {"id": "lq11", "query": "Does getting two consecutive warnings mean I am immediately expelled from university?", "expected_domain": "in"},
    {"id": "lq12", "query": "How many consecutive academic warnings will lead to permanent drop from degree?", "expected_domain": "in"},
    {"id": "lq13", "query": "What is the exact letter grade point for A minus on the university grading chart?", "expected_domain": "in"},
    {"id": "lq14", "query": "What is the lowest passing grade required for a prerequisite programming course?", "expected_domain": "in"},
    {"id": "lq15", "query": "Under 2026 rules if I repeat a course will my previous grade be replaced completely?", "expected_domain": "in"},
    {"id": "lq16", "query": "Was course repeat grade capped at B under the previous 2023 undergraduate regulations?", "expected_domain": "in"},
    {"id": "lq17", "query": "What semester GPA is needed to get my name on the Dean's Honor List?", "expected_domain": "in"},
    {"id": "lq18", "query": "What is the deadline week to withdraw from a registered course with a W grade?", "expected_domain": "in"},
    {"id": "lq19", "query": "Can a student withdraw from a course after the 10th week under any circumstances?", "expected_domain": "in"},
    {"id": "lq20", "query": "What is the maximum course credit load allowed for regular full-time undergraduate students?", "expected_domain": "in"},
    {"id": "lq21", "query": "Can final year students take 21 credit hours with special advisor approval?", "expected_domain": "in"},
    {"id": "lq22", "query": "How many total credit hours must I pass to graduate with a BS in Computer Science?", "expected_domain": "in"},
    {"id": "lq23", "query": "What is the maximum number of years allowed to complete a four year BS program?", "expected_domain": "in"},
    {"id": "lq24", "query": "What is the minimum cumulative GPA needed to receive the bachelor degree at graduation?", "expected_domain": "in"},
    {"id": "lq25", "query": "What intermediate percentage is required for computing department admissions eligibility?", "expected_domain": "in"},
    {"id": "lq26", "query": "Are undergraduate students required to do a mandatory summer internship before graduating?", "expected_domain": "in"},
    {"id": "lq27", "query": "What are the rules for supplementary examinations in the final graduating semester?", "expected_domain": "in"},
    {"id": "lq28", "query": "What is the charge for obtaining an official duplicate university transcript?", "expected_domain": "in"},

    # -------------------------------------------------------------------------
    # Out-of-Domain Queries (12)
    # -------------------------------------------------------------------------
    {"id": "lq29", "query": "What is the guest Wi-Fi password for students in the girls hostel?", "expected_domain": "out"},
    {"id": "lq30", "query": "What are the curfew hours for entering the boys hostel on weekends?", "expected_domain": "out"},
    {"id": "lq31", "query": "How do I apply for room allotment and pay the hostel security deposit?", "expected_domain": "out"},
    {"id": "lq32", "query": "What is the weekly dinner menu and chicken biryani schedule in the student mess?", "expected_domain": "out"},
    {"id": "lq33", "query": "What are the morning shuttle pickup timings from Rawalpindi Saddar to campus?", "expected_domain": "out"},
    {"id": "lq34", "query": "How much is the semester bus card fee for students commuting from Islamabad?", "expected_domain": "out"},
    {"id": "lq35", "query": "What time does the campus library close during midterm examination weeks?", "expected_domain": "out"},
    {"id": "lq36", "query": "How many books can an undergraduate student borrow simultaneously from the central library?", "expected_domain": "out"},
    {"id": "lq37", "query": "When are the trials for the university varsity cricket and badminton teams?", "expected_domain": "out"},
    {"id": "lq38", "query": "What are the operating hours and monthly membership fees for the campus gym?", "expected_domain": "out"},
    {"id": "lq39", "query": "Where is the lost and found office located if I forgot my student ID card?", "expected_domain": "out"},
    {"id": "lq40", "query": "What lunch items and sandwiches are sold at the main cafeteria tuck shop?", "expected_domain": "out"},
]


def create_live_queries(output_path: Path = TARGET_FILE) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in LIVE_QUERIES:
            f.write(json.dumps(entry) + "\n")
    print(f"Successfully generated {len(LIVE_QUERIES)} live queries at {output_path}")
    in_cnt = sum(1 for q in LIVE_QUERIES if q["expected_domain"] == "in")
    out_cnt = sum(1 for q in LIVE_QUERIES if q["expected_domain"] == "out")
    print(f"  Distribution: {in_cnt} in-domain, {out_cnt} out-of-domain")


if __name__ == "__main__":
    create_live_queries()
