import re
from pypdf import PdfReader


def load_pdf_text(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_inbody(text):
    data = {
        "BASIC DEMOGRAPHIC & TEST INFORMATION": [],
        "BODY COMPOSITION ANALYSIS": [],
        "MUSCLE & FAT ANALYSIS": [],
        "OBESITY & METABOLIC ANALYSIS": [],
        "SEGMENTAL LEAN ANALYSIS": [],
        "BODY SCORE & GOALS": []
    }

    text = re.sub(r"\s+", " ", text).strip()

    # Basic information
    basic_patterns = [
        (r"\bAge\s*[:\-]?\s*(\d+)", "Age", " years"),
        (r"\bGender\s*[:\-]?\s*(Male|Female)", "Gender", ""),
        (r"\bHeight\s*[:\-]?\s*([\d.]+)\s*cm", "Height", " cm"),
        (r"\bWeight\s*[:\-]?\s*([\d.]+)\s*kg", "Weight", " kg")
    ]

    for pattern, name, unit in basic_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            data["BASIC DEMOGRAPHIC & TEST INFORMATION"].append(
                f"{name}: {match.group(1)}{unit}"
            )

    # Body composition
    body_patterns = [
        (r"Total Body Water\s*\(TBW\)\s*([\d.]+)\s*L", "Total Body Water (TBW)", " L"),
        (r"Dry Lean Mass\s*([\d.]+)\s*kg", "Dry Lean Mass", " kg"),
        (r"Body Fat Mass\s*\(BFM\)\s*([\d.]+)\s*kg", "Body Fat Mass (BFM)", " kg"),
        (r"Total Weight\s*([\d.]+)\s*kg", "Total Weight", " kg")
    ]

    for pattern, name, unit in body_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            data["BODY COMPOSITION ANALYSIS"].append(
                f"{name}: {match.group(1)}{unit}"
            )

    # Muscle & Fat
    match = re.search(
        r"Skeletal Muscle Mass\s*\(SMM\)\s*([\d.]+)\s*kg",
        text,
        re.I
    )

    if match:
        data["MUSCLE & FAT ANALYSIS"].append(
            f"Skeletal Muscle Mass (SMM): {match.group(1)} kg"
        )

    match = re.search(
        r"Body Fat Mass\s*\(BFM\)\s*([\d.]+)\s*kg",
        text,
        re.I
    )

    if match:
        data["MUSCLE & FAT ANALYSIS"].append(
            f"Body Fat Mass (BFM): {match.group(1)} kg"
        )

    # Metabolic
    metabolic_patterns = [
        (
            r"BMI\s*\(Body Mass Index\)\s*([\d.]+)\s*kg/m²",
            "BMI (Body Mass Index)",
            " kg/m²"
        ),
        (
            r"(?:Percent Body Fat|PBF)\s*\(?(?:PBF)?\)?\s*([\d.]+)\s*%",
            "Percent Body Fat (PBF)",
            " %"
        ),
        (
            r"Basal Metabolic Rate\s*\(BMR\)\s*([\d,]+)\s*kcal",
            "Basal Metabolic Rate (BMR)",
            " kcal"
        ),
        (
            r"Waist-Hip Ratio\s*\(WHR\)\s*([\d.]+)",
            "Waist-Hip Ratio (WHR)",
            ""
        ),
        (
            r"Visceral Fat Level\s*(Level\s*\d+)",
            "Visceral Fat Level",
            ""
        )
    ]

    for pattern, name, unit in metabolic_patterns:
        match = re.search(pattern, text, re.I)

        if match:
            data["OBESITY & METABOLIC ANALYSIS"].append(
                f"{name}: {match.group(1)}{unit}"
            )

    # Segmental analysis
    segments = [
        ("Right Arm", r"Right Arm\s*([\d.]+)\s*kg\s*(?:Normal\s*)?\(?(\d+)%\)?"),
        ("Left Arm", r"Left Arm\s*([\d.]+)\s*kg\s*(?:Normal\s*)?\(?(\d+)%\)?"),
        ("Trunk", r"Trunk\s*(?:\(Torso\))?\s*([\d.]+)\s*kg\s*(?:Normal\s*)?\(?(\d+)%\)?"),
        ("Right Leg", r"Right Leg\s*([\d.]+)\s*kg\s*(?:Normal\s*)?\(?(\d+)%\)?"),
        ("Left Leg", r"Left Leg\s*([\d.]+)\s*kg\s*(?:Normal\s*)?\(?(\d+)%\)?")
    ]

    for name, pattern in segments:
        match = re.search(pattern, text, re.I)

        if match:
            data["SEGMENTAL LEAN ANALYSIS"].append(
                f"{name} Lean Mass: {match.group(1)} kg. "
                f"Percentage of Normal: {match.group(2)}%"
            )

    # Score & goals
    goal_patterns = [
        (
            r"InBody Score\s*(?:\(GOOD CONDITION\))?\s*[:\-]?\s*(\d+)\s*/\s*100",
            "InBody Score",
            ""
        ),
        (
            r"Target Weight\s*([\d.]+)\s*kg",
            "Target Weight",
            " kg"
        ),
        (
            r"Fat Control\s*([+-]?[\d.]+)\s*kg",
            "Fat Control",
            " kg"
        ),
        (
            r"Muscle Control\s*([+-]?[\d.]+)\s*kg",
            "Muscle Control",
            " kg"
        )
    ]

    for pattern, name, unit in goal_patterns:
        match = re.search(pattern, text, re.I)

        if match:
            value = match.group(1)

            if name == "InBody Score":
                value += "/100"
            else:
                value += unit

            data["BODY SCORE & GOALS"].append(
                f"{name}: {value}"
            )

    return data


def create_chunks(data):
    chunks = []

    for section, values in data.items():
        if values:
            chunks.append(
                f"SECTION: {section}\n" +
                "\n".join(values)
            )

    return chunks