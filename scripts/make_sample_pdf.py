"""Builds sample_docs/employee_handbook.pdf. The company and all its details are invented."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    ("1. Working Hours and Remote Work", [
        "Larkspur Outdoor Co. operates a standard 38-hour working week. Core hours, when everyone is expected to be reachable, are 10:00 to 15:00 local time, Monday to Thursday. Friday has no core hours.",
        "Employees who have completed their three-month probation may work remotely up to three days per week. Workshop and warehouse staff are not eligible for remote work because their duties require equipment on site.",
        "Working from another country is allowed for a maximum of 20 working days per calendar year and must be approved by the People team at least four weeks in advance.",
    ]),
    ("2. Annual Leave", [
        "Full-time employees receive 26 days of paid annual leave per calendar year, in addition to public holidays. Part-time employees receive a pro-rated amount.",
        "After five full years of service the allowance rises by 3 days. Up to 6 unused days may be carried into the next year, but carried days expire on 31 March.",
        "Leave of more than ten consecutive working days requires approval from both the line manager and the department head. Requests are made in the Trailhead system.",
    ]),
    ("3. Sickness and Parental Leave", [
        "Employees should tell their line manager about sickness absence before 09:30 on the first day. A medical certificate is required from the fourth consecutive calendar day of absence.",
        "Sick pay is 100 percent of salary for the first 30 working days in any rolling twelve-month period, then 60 percent for a further 40 working days.",
        "The primary carer of a new child receives 22 weeks of fully paid parental leave. The secondary carer receives 8 weeks of fully paid leave, which may be taken in up to two blocks during the first year.",
    ]),
    ("4. Expenses and Travel", [
        "Expense claims must be submitted in Trailhead within 45 days of the purchase, with a receipt for every item above 15 euros. Approved claims are paid with the next monthly salary.",
        "The daily meal limit while travelling is 55 euros. Hotel stays are limited to 160 euros per night, or 210 euros in capital cities. Rail travel is second class; journeys above four hours may be booked in first class.",
        "Flights are permitted only where the rail journey would exceed seven hours. Private car use is reimbursed at 0.38 euros per kilometre.",
    ]),
    ("5. Equipment and Security", [
        "Office staff receive a laptop, which is replaced every four years. Each employee may spend up to 350 euros on home office equipment once every three years, after finishing probation.",
        "A lost or stolen device must be reported to the security desk at extension 4410 within two hours of discovery. Company data may not be stored on personal devices.",
        "All accounts must use a hardware security key. Shared passwords are not permitted.",
    ]),
    ("6. Learning and Staff Discount", [
        "Every employee has a learning budget of 1,200 euros per calendar year for courses, books and conferences, plus four paid learning days. The unused budget does not carry over.",
        "Courses costing more than 800 euros require a short written proposal to the line manager.",
        "Staff receive a 40 percent discount on Larkspur products for personal use, limited to 1,500 euros of purchases per year at full retail value.",
    ]),
]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "sample_docs" / "employee_handbook.pdf"
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paragraphs) in enumerate(PAGES):
        if i:
            story.append(PageBreak())
        story.append(Paragraph("Larkspur Outdoor Co. Employee Handbook", styles["Italic"]))
        story += [Spacer(1, 12), Paragraph(title, styles["Heading1"]), Spacer(1, 6)]
        for text in paragraphs:
            story += [Paragraph(text, styles["BodyText"]), Spacer(1, 8)]
    SimpleDocTemplate(str(out), pagesize=A4, title="Employee Handbook", invariant=1).build(story)
    print(out)


if __name__ == "__main__":
    main()
