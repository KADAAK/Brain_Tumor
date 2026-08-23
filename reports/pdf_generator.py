from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak


DISCLAIMER = "This is AI-assisted image analysis using model-predicted segmented regions. It is not a diagnosis and requires review by a qualified medical professional. No treatment recommendation is provided."


def _table(rows):
    table=Table(rows, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#154360")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.25,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),5)]))
    return table


def generate_pdf(result: dict, assets: dict, destination: Path) -> Path:
    styles=getSampleStyleSheet(); story=[]
    story += [Spacer(1,1.5*inch), Paragraph("AI-Assisted Brain MRI Analysis", styles["Title"]), Spacer(1,0.3*inch), Paragraph(f"Study ID: {result['study_id']}", styles["Heading2"]), Spacer(1,0.3*inch), Paragraph(DISCLAIMER, styles["BodyText"]), PageBreak()]
    story += [Paragraph("Study Information", styles["Heading1"]), Paragraph(f"Detected segmented regions: {result['tumor_count']}", styles["BodyText"])]
    for label in ("original", "annotated"):
        if assets.get(label): story += [Spacer(1,8), Image(str(assets[label]), width=5.5*inch, height=5.5*inch)]
    story += [Paragraph("Tumor Summary", styles["Heading1"])]
    rows=[["ID","Area (px)","Max diameter (px)","Centroid"]]+[[t["tumor_id"],str(t["area_pixels"]),f"{t['max_diameter_pixels']:.1f}",str([round(x,1) for x in t["centroid"]])] for t in result["tumors"]]
    story.append(_table(rows))
    if result["pairwise_analysis"]:
        story += [Spacer(1,12),Paragraph("Tumor-to-Tumor Distances", styles["Heading1"])]
        rows=[["Pair","Centroid px","Boundary px","Relative position"]]+[[f"{p['tumor_a']}–{p['tumor_b']}",f"{p['centroid_distance_pixels']:.1f}",f"{p['boundary_distance_pixels']:.1f}",p["relative_position"]] for p in result["pairwise_analysis"]]
        story.append(_table(rows))
    for i,t in enumerate(result["tumors"]):
        story += [PageBreak(),Paragraph(f"Segmented Region {t['tumor_id']}", styles["Heading1"]),Paragraph(f"Bounding box: {t['bbox']}<br/>Equivalent radius: {t['equivalent_radius_pixels']:.1f} pixels",styles["BodyText"])]
        if i < len(assets.get("crops",[])): story.append(Image(str(assets["crops"][i]),width=3*inch,height=3*inch))
    story += [PageBreak(),Paragraph("Model and Limitations",styles["Heading1"]),Paragraph(f"Model: {result['model']['name']} v{result['model']['version']}. Segmentation metrics: placeholder pending validation with a trained model.",styles["BodyText"]),Spacer(1,8),Paragraph("Supportive health information placeholder: discuss findings and any concerns with a qualified clinician. Specialist information placeholder: neuroradiology/neurosurgery review may be appropriate at the treating team's discretion.",styles["BodyText"])]
    SimpleDocTemplate(str(destination),pagesize=A4).build(story); return destination
