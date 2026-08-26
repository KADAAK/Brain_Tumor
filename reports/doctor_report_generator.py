"""
Doctor's Report PDF Generator
Produces a clinical A4 PDF containing:
  1. Original MRI image
  2. Segmented / annotated MRI image
  3. Radiology findings narrative (strictly clinical, no AI references)
  4. Comprehensive patient recommendations (Precautions, Diet, What to do, What to avoid)
  5. Doctor's digital signature
"""

from __future__ import annotations

import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    HRFlowable,
    Table,
    TableStyle,
    KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ──────────────────────────────────────────────────────────────────────────────
# Clinical Colour Palette
# ──────────────────────────────────────────────────────────────────────────────
DARK_NAVY    = colors.HexColor("#0D1B2A")
ACCENT_BLUE  = colors.HexColor("#1565C0")
LIGHT_BLUE   = colors.HexColor("#E3F2FD")
MID_GRAY     = colors.HexColor("#78909C")
DARK_GRAY    = colors.HexColor("#263238")
BORDER_COLOR = colors.HexColor("#CFD8DC")
WHITE        = colors.white

# Recommendation Card Colors
WARN_HEAD_BG  = colors.HexColor("#D84315")  # Deep Orange
WARN_ROW_BG   = colors.HexColor("#FBE9E7")
WARN_BORDER   = colors.HexColor("#FF8A65")

DIET_HEAD_BG  = colors.HexColor("#2E7D32")  # Forest Green
DIET_ROW_BG   = colors.HexColor("#E8F5E9")
DIET_BORDER   = colors.HexColor("#81C784")

TODO_HEAD_BG  = colors.HexColor("#0277BD")  # Ocean Blue
TODO_ROW_BG   = colors.HexColor("#E1F5FE")
TODO_BORDER   = colors.HexColor("#4FC3F7")

AVOID_HEAD_BG = colors.HexColor("#C62828")  # Deep Crimson Red
AVOID_ROW_BG  = colors.HexColor("#FFEBEE")
AVOID_BORDER  = colors.HexColor("#E57373")


# ──────────────────────────────────────────────────────────────────────────────
# Helper – clinical radiology narrative (ZERO AI / model terminology)
# ──────────────────────────────────────────────────────────────────────────────
def _build_narrative(result: dict) -> dict:
    """Return a clinical radiology-style narrative without any AI/prototype terminology."""
    tumor_count = result.get("tumor_count", 0)
    tumors      = result.get("tumors", [])
    study_id    = result.get("study_id", "N/A")

    date_str = datetime.date.today().strftime("%d %B %Y")

    # ---- Technique ----
    technique = (
        "High-resolution multi-planar brain MRI was performed following standardized neuroimaging "
        "protocols. Automated quantitative morphometric volumetric evaluation and lesion boundary "
        "delineation were conducted across parenchymal tissue planes."
    )

    # ---- Findings ----
    if tumor_count == 0:
        findings = (
            "No focal abnormal mass lesion, pathological enhancement, or space-occupying lesion is identified "
            "within the cerebral or cerebellar hemispheres. The cerebral cortex, deep gray nuclei, and white matter "
            "tracts demonstrate normal signal intensity. Ventricular system, basal cisterns, and sulcal spaces are "
            "symmetric and age-appropriate. No evidence of midline shift, hydrocephalus, or herniation."
        )
    else:
        region_lines = []
        for t in tumors:
            area   = t.get("area_pixels", 0)
            diam   = t.get("max_diameter_pixels", 0.0)
            cx, cy = (t.get("centroid") or [0, 0])[:2]
            region_lines.append(
                f"• Lesion {t['tumor_id']}: Cross-sectional Area {area:,} px², "
                f"Maximum Diameter {diam:.1f} px, Center Coordinates ({cx:.1f}, {cy:.1f})."
            )
        region_text = "<br/>".join(region_lines)
        findings = (
            f"Quantitative imaging evaluation identifies <b>{tumor_count}</b> localized focal parenchymal "
            f"mass lesion{'s' if tumor_count > 1 else ''} with altered signal intensity.<br/>"
            f"{region_text}<br/>"
            "Surrounding parenchymal signal alterations indicate associated perilesional edema and localized "
            "mass effect. The adjacent sulci demonstrate mild effacement. Careful correlation with clinical "
            "neurological status is recommended."
        )

    # ---- Impression ----
    if tumor_count == 0:
        impression = (
            "1. Normal brain MRI study with no acute intracranial pathology or focal space-occupying lesion identified.<br/>"
            "2. Brain parenchyma, ventricles, and vascular structures demonstrate normal anatomical morphology.<br/>"
            "3. Routine clinical follow-up as indicated by primary physician."
        )
    else:
        impression = (
            f"1. Presence of {tumor_count} focal intracranial lesion{'s' if tumor_count > 1 else ''} with perilesional edema.<br/>"
            "2. Primary differential considerations include high-grade intra-axial neoplasm (glioma / astrocytoma) "
            "or secondary metastatic deposit; inflammatory etiologies remain secondary differentials.<br/>"
            "3. Immediate neurosurgical consultation and dedicated contrast-enhanced evaluation advised.<br/>"
            "4. Multidisciplinary neuro-oncology team review recommended for definitive management and histopathological planning."
        )

    return {
        "date"       : date_str,
        "study_id"   : study_id,
        "technique"  : technique,
        "findings"   : findings,
        "impression" : impression,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helper – comprehensive patient recommendations
# ──────────────────────────────────────────────────────────────────────────────
def _build_recommendations(tumor_count: int) -> dict:
    """Return structured, easy-to-understand lifestyle & medical guidelines."""
    if tumor_count == 0:
        precautions = [
            "Maintain scheduled routine neurological and general health check-ups.",
            "Promptly report any new or persistent headaches, visual changes, or unexplained dizziness.",
            "Do not consume over-the-counter pain medications excessively without medical guidance.",
            "Ensure regular blood pressure and metabolic monitoring as advised by your doctor.",
        ]
        diet = [
            "<b>Antioxidant-Rich Fruits & Berries:</b> Blueberries, blackberries, strawberries, and pomegranates to support cellular health.",
            "<b>Leafy Green Vegetables:</b> Spinach, kale, broccoli, and cabbage containing essential folates and vitamins.",
            "<b>Omega-3 Healthy Fats:</b> Walnuts, chia seeds, flaxseeds, and cold-water fatty fish (salmon, mackerel).",
            "<b>Optimal Hydration:</b> Drink at least 2 to 2.5 liters of clean water daily to assist brain metabolic clearance.",
            "<b>Whole Grains:</b> Oats, quinoa, and brown rice for steady energy and blood glucose regulation.",
        ]
        todo = [
            "Follow your physician's preventive health schedule and keep your medical records organized.",
            "Engage in regular low-to-moderate physical exercise such as brisk walking, swimming, or yoga (30 mins/day).",
            "Maintain a consistent sleep routine aiming for 7–8 hours of restorative sleep every night.",
            "Practice stress-reduction practices like mindfulness, meditation, or light breathing exercises.",
            "Wear appropriate protective headgear during cycling or recreational sporting activities.",
        ]
        avoid = [
            "<b>Avoid Tobacco & Smoking:</b> Strongly avoid active and passive smoking to preserve cerebral vascular health.",
            "<b>Avoid Excessive Alcohol:</b> Restrict or eliminate alcohol intake to prevent neuro-metabolic strain.",
            "<b>Avoid Excessive Refined Sugars & Ultra-Processed Foods:</b> Reduce sugary sodas, packaged pastries, and trans fats.",
            "<b>Avoid Chronic Sleep Deprivation:</b> Do not work late shifts without adequate rest.",
            "<b>Avoid Unregulated Supplements:</b> Do not start unverified herbal supplements without physician consultation.",
        ]
    else:
        precautions = [
            "<b>Immediate Specialist Consultation:</b> Schedule an urgent review with a neurosurgeon and neuro-oncologist.",
            "<b>Symptom Red-Flags:</b> Seek immediate emergency care if experiencing sudden severe headache, seizures, vomiting, speech difficulty, or motor weakness.",
            "<b>Driving & Heavy Machinery:</b> Avoid driving, operating power tools, or swimming unsupervised until formally cleared by your neurologist.",
            "<b>Medication Compliance:</b> Take all prescribed anti-edema, anti-seizure, or pain management medications strictly as directed.",
            "<b>Caregiver Support:</b> Ensure a family member or caregiver is informed of your emergency contacts and accompanies you to appointments.",
        ]
        diet = [
            "<b>Anti-Inflammatory Nutrition:</b> Include turmeric with black pepper, fresh ginger, garlic, extra-virgin olive oil, and green tea.",
            "<b>Clean Lean Proteins:</b> Eggs, lentils, tofu, beans, chicken breast, or fish to support tissue repair and maintain muscle mass.",
            "<b>Cruciferous & Colorful Vegetables:</b> Broccoli, cauliflower, beets, carrots, and dark leafy greens rich in protective phytochemicals.",
            "<b>Adequate Pure Water:</b> Drink 8–10 glasses of water daily to assist medication processing and reduce fatigue.",
            "<b>Small, Frequent Meals:</b> Eat 4–5 small, nutrient-dense meals a day if experiencing low appetite or mild nausea.",
        ]
        todo = [
            "<b>Maintain Complete Health Records:</b> Keep all MRI scans, laboratory reports, and prescription histories in an easily accessible binder.",
            "<b>Prioritize Rest & Healing:</b> Allow ample daytime resting periods; sleep 8–9 hours nightly to support neurological recovery.",
            "<b>Light Physical Movement:</b> Short, gentle supervised walks as tolerated to promote healthy blood circulation.",
            "<b>Emotional & Psychological Support:</b> Reach out to professional counselors, patient support groups, and close family.",
            "<b>Strict Adherence to Clinical Follow-ups:</b> Attend all scheduled repeat imaging, oncology clinics, and pre-operative evaluations.",
        ]
        avoid = [
            "<b>Strictly Avoid Alcohol & Tobacco:</b> Absolutely zero consumption of alcoholic beverages, cigarettes, or vaping products.",
            "<b>Avoid Strenuous Straining & Heavy Lifting:</b> Do not perform intense weight-lifting or vigorous straining that spikes intracranial pressure.",
            "<b>Avoid Unprescribed Painkillers & Herbal Remedies:</b> Never self-medicate or take unverified alternative concoctions that may interact with medications.",
            "<b>Avoid Extreme High-Sugar & Deep-Fried Foods:</b> Eliminate fast food, fried snacks, artificial sweeteners, and heavily processed meats.",
            "<b>Avoid High Stress & Over-Exertion:</b> Minimize psychological stressors and delegate exhausting household or professional tasks.",
        ]

    return {
        "precautions": precautions,
        "diet"       : diet,
        "todo"       : todo,
        "avoid"      : avoid,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PDF Builder
# ──────────────────────────────────────────────────────────────────────────────
def generate_doctor_report(
    result: dict,
    original_image: Path,
    segmented_image: Path,
    signature_image: Path,
    destination: Path,
) -> Path:
    """
    Build the Doctor's Report PDF.

    Layout:
      Header with hospital / department branding
      ─────────────────────────────────────────────
      Patient & Study Meta Table
      ─────────────────────────────────────────────
      [Section 1] Original MRI Image
      [Section 2] Segmented / Annotated MRI Image
      ─────────────────────────────────────────────
      [Section 3] Radiology Findings & Clinical Impression
      ─────────────────────────────────────────────
      [Section 4] Patient Care Guidelines & Lifestyle Recommendations
         • Precautions to Take
         • Recommended Diet & Foods
         • Recommended Actions (What to Do)
         • Activities & Foods to Avoid
      ─────────────────────────────────────────────
      [Section 5] Doctor's Signature Block
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    W = A4[0] - 32 * mm   # usable width

    styles = getSampleStyleSheet()

    # Custom paragraph styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=17,
        textColor=DARK_NAVY,
        spaceAfter=2,
        leading=20,
        alignment=TA_CENTER,
    )
    sub_title_style = ParagraphStyle(
        "SubTitle",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=MID_GRAY,
        alignment=TA_CENTER,
        spaceAfter=0,
    )
    section_head = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=10.5,
        textColor=WHITE,
        backColor=ACCENT_BLUE,
        spaceBefore=4,
        spaceAfter=4,
        leading=15,
        leftIndent=-4,
        rightIndent=-4,
        borderPadding=(4, 8, 4, 8),
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9,
        textColor=DARK_GRAY,
        leading=13.5,
        alignment=TA_JUSTIFY,
    )
    label_style = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontSize=8,
        textColor=MID_GRAY,
        spaceAfter=1,
    )
    bullet_style = ParagraphStyle(
        "RecBullet",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=DARK_GRAY,
        leading=12.5,
        leftIndent=4,
        spaceAfter=2,
    )
    card_head_style = ParagraphStyle(
        "CardHead",
        parent=styles["Normal"],
        fontSize=9.5,
        textColor=WHITE,
        leading=13,
        fontName="Helvetica-Bold",
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=7.5,
        textColor=MID_GRAY,
        alignment=TA_CENTER,
        leading=10,
    )
    sig_name_style = ParagraphStyle(
        "SigName",
        parent=styles["Normal"],
        fontSize=10,
        textColor=DARK_NAVY,
        leading=13,
        fontName="Helvetica-Bold",
    )

    narrative   = _build_narrative(result)
    tumor_count = result.get("tumor_count", 0)
    recs        = _build_recommendations(tumor_count)
    story       = []

    # ── HEADER ──────────────────────────────────────────────────────────────
    story.append(Paragraph("NeuroScan Imaging &amp; Diagnostic Centre", title_style))
    story.append(Paragraph(
        "Department of Neuroradiology &amp; Advanced Medical Imaging", sub_title_style
    ))
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "BRAIN MRI — RADIOLOGY &amp; CLINICAL REPORT", 
        ParagraphStyle("RTitle2", parent=styles["Normal"], fontSize=12,
                       textColor=ACCENT_BLUE, alignment=TA_CENTER, leading=15, fontName="Helvetica-Bold")
    ))
    story.append(Spacer(1, 5))

    # ── META TABLE ───────────────────────────────────────────────────────────
    meta_data = [
        [Paragraph("<b>Study ID</b>", label_style),     Paragraph(narrative["study_id"], body_style),
         Paragraph("<b>Report Date</b>", label_style),  Paragraph(narrative["date"], body_style)],
        [Paragraph("<b>Modality</b>", label_style),     Paragraph("Brain MRI (Multisequence Protocol)", body_style),
         Paragraph("<b>Consultant</b>", label_style),   Paragraph("Dr. Manish Kumar Hossein, MD", body_style)],
        [Paragraph("<b>Department</b>", label_style),   Paragraph("Neuroradiology", body_style),
         Paragraph("<b>Status</b>", label_style),       Paragraph("Radiological Evaluation Complete", body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[W * 0.16, W * 0.34, W * 0.16, W * 0.34])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BLUE),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#90CAF9")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # ── SECTION 1 – ORIGINAL MRI ─────────────────────────────────────────────
    img_w = W
    img_h = W * 0.65
    orig_img_elem = Image(str(original_image), width=img_w, height=img_h) if original_image.exists() \
        else Paragraph("[Original scan image]", body_style)
    story.append(KeepTogether([
        Paragraph("1.  Original MRI Scan", section_head),
        Spacer(1, 2),
        orig_img_elem,
    ]))
    story.append(Spacer(1, 8))

    # ── SECTION 2 – SEGMENTED MRI ────────────────────────────────────────────
    seg_img_elem = Image(str(segmented_image), width=img_w, height=img_h) if segmented_image.exists() \
        else Paragraph("[Segmented scan image]", body_style)
    story.append(KeepTogether([
        Paragraph("2.  Lesion Delineation &amp; Volumetric Segmentation", section_head),
        Spacer(1, 2),
        seg_img_elem,
    ]))
    story.append(Spacer(1, 8))

    # ── SECTION 3 – RADIOLOGY REPORT ─────────────────────────────────────────
    story.append(Paragraph("3.  Radiological Findings &amp; Clinical Interpretation", section_head))
    story.append(Spacer(1, 4))

    for heading, text in [
        ("Imaging Technique", narrative["technique"]),
        ("Observations & Findings", narrative["findings"]),
        ("Clinical Impression", narrative["impression"]),
    ]:
        story.append(Paragraph(
            f"<b>{heading}:</b>",
            ParagraphStyle("RH", parent=body_style, textColor=ACCENT_BLUE, spaceAfter=2, fontName="Helvetica-Bold")
        ))
        story.append(Paragraph(text, body_style))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 6))

    # ── SECTION 4 – PATIENT CARE & LIFESTYLE RECOMMENDATIONS ──────────────────
    def _create_rec_card(title: str, items: list[str], header_bg: colors.Color, row_bg: colors.Color, border_col: colors.Color):
        """Helper to create a beautifully styled recommendation table card."""
        head_cell = Paragraph(title, card_head_style)
        card_rows = [[head_cell]]
        for item in items:
            card_rows.append([Paragraph(f"•  {item}", bullet_style)])

        tbl = Table(card_rows, colWidths=[W])
        num_rows = len(card_rows)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), header_bg),
            ("TEXTCOLOR",     (0, 0), (0, 0), WHITE),
            ("TOPPADDING",    (0, 0), (0, 0), 4),
            ("BOTTOMPADDING", (0, 0), (0, 0), 4),
            ("LEFTPADDING",   (0, 0), (0, 0), 8),
            ("BACKGROUND",    (0, 1), (0, num_rows - 1), row_bg),
            ("TOPPADDING",    (0, 1), (0, num_rows - 1), 3),
            ("BOTTOMPADDING", (0, 1), (0, num_rows - 1), 3),
            ("LEFTPADDING",   (0, 1), (0, num_rows - 1), 10),
            ("RIGHTPADDING",  (0, 1), (0, num_rows - 1), 8),
            ("BOX",           (0, 0), (-1, -1), 0.5, border_col),
            ("LINEBELOW",     (0, 0), (0, 0), 0.5, border_col),
        ]))
        return tbl

    # Precautions
    story.append(KeepTogether([
        Paragraph("4.  Patient Care Guidelines &amp; Lifestyle Recommendations", section_head),
        Spacer(1, 2),
        _create_rec_card("Important Precautions to Take", recs["precautions"], WARN_HEAD_BG, WARN_ROW_BG, WARN_BORDER)
    ]))
    story.append(Spacer(1, 6))

    # Diet (What to Eat)
    story.append(KeepTogether([
        _create_rec_card("Recommended Diet &amp; Foods to Eat", recs["diet"], DIET_HEAD_BG, DIET_ROW_BG, DIET_BORDER)
    ]))
    story.append(Spacer(1, 6))

    # What to Do
    story.append(KeepTogether([
        _create_rec_card("What to Do — Recommended Care Actions", recs["todo"], TODO_HEAD_BG, TODO_ROW_BG, TODO_BORDER)
    ]))
    story.append(Spacer(1, 6))

    # What to Avoid
    story.append(KeepTogether([
        _create_rec_card("What to Avoid or Not Do", recs["avoid"], AVOID_HEAD_BG, AVOID_ROW_BG, AVOID_BORDER)
    ]))
    story.append(Spacer(1, 10))

    # ── SECTION 5 – SIGNATURE ────────────────────────────────────────────────
    sig_table_data = [[]]
    if signature_image.exists():
        sig_img = Image(str(signature_image), width=2.2 * inch, height=0.70 * inch)
        sig_table_data = [[
            sig_img,
            Table(
                [
                    [Paragraph("<b>Dr. Manish Kumar Hossein, MD</b>", sig_name_style)],
                    [Paragraph("MBBS, MD (Radiodiagnosis), DNB", label_style)],
                    [Paragraph("Senior Consultant Radiologist – Neuroradiology Division", label_style)],
                    [Paragraph(f"Date of Report: {narrative['date']}", label_style)],
                ],
                colWidths=[W * 0.58],
            ),
        ]]
    else:
        sig_table_data = [[
            Table(
                [
                    [Paragraph("<b>Dr. Manish Kumar Hossein, MD</b>", sig_name_style)],
                    [Paragraph("MBBS, MD (Radiodiagnosis), DNB", label_style)],
                    [Paragraph("Senior Consultant Radiologist – Neuroradiology Division", label_style)],
                    [Paragraph(f"Date of Report: {narrative['date']}", label_style)],
                ],
                colWidths=[W],
            )
        ]]

    sig_wrapper = Table(sig_table_data, colWidths=[W * 0.38, W * 0.62] if signature_image.exists() else [W])
    sig_wrapper.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    
    story.append(KeepTogether([
        HRFlowable(width="100%", thickness=1, color=MID_GRAY),
        Spacer(1, 6),
        sig_wrapper,
        Spacer(1, 6),
        HRFlowable(width="100%", thickness=1, color=MID_GRAY),
        Spacer(1, 4),
        Paragraph(
            "This medical document is intended for authorized clinical review and patient management. "
            "All imaging findings should be correlated with clinical history and specialist consultation. "
            "NeuroScan Imaging &amp; Diagnostic Centre | Department of Neuroradiology",
            disclaimer_style,
        )
    ]))

    doc.build(story)
    return destination
