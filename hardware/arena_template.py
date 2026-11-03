"""Generate a vector 1:1 arena sheet and overlapping A4 assembly tiles.

No printer or physical arena is verified by this generator. Wall inner faces align
to the nominal boundary centerline; wall thickness/height/material are unconfirmed.
"""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import reportlab
import pypdf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader
from bari.config import Config

MM = 72/25.4
RADIUS_MM = 1000.
CONTENT = (10., 17., 190., 260.)
OVERLAP_MM = 10.
COLS, ROWS = 12, 8
TOTAL_PAGES = COLS*ROWS+1
PAGE_MM = 2*RADIUS_MM+100
STEP_X, STEP_Y = CONTENT[2]-OVERLAP_MM, CONTENT[3]-OVERLAP_MM
ORIGIN_X = -(CONTENT[2]+(COLS-1)*STEP_X)/2
ORIGIN_Y = -(CONTENT[3]+(ROWS-1)*STEP_Y)/2


def circle_segments(radius=RADIUS_MM, count=24):
    k = 4/3*math.tan(math.pi/(2*count))
    for index in range(count):
        a, b = 2*math.pi*index/count, 2*math.pi*(index+1)/count
        start = (radius*math.cos(a), radius*math.sin(a))
        end = (radius*math.cos(b), radius*math.sin(b))
        control1 = (start[0]-k*radius*math.sin(a), start[1]+k*radius*math.cos(a))
        control2 = (end[0]+k*radius*math.sin(b), end[1]-k*radius*math.cos(b))
        yield start, control1, control2, end


def circle_path(c):
    path = c.beginPath()
    path.moveTo(RADIUS_MM*MM, 0)
    for _, p1, p2, end in circle_segments():
        path.curveTo(p1[0]*MM, p1[1]*MM, p2[0]*MM, p2[1]*MM, end[0]*MM, end[1]*MM)
    path.close()
    return path


def line(c, x1, y1, x2, y2):
    c.line(x1*MM, y1*MM, x2*MM, y2*MM)


def text(c, x, y, value, size=10, bold=False, align="left"):
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    method = {"left": c.drawString, "right": c.drawRightString, "center": c.drawCentredString}[align]
    method(x*MM, y*MM, value)


def ruler(c, x, y, label=True):
    c.setStrokeGray(0)
    c.setFillGray(0)
    c.setLineWidth(.2*MM)
    line(c, x, y, x+100, y)
    for offset in range(0, 101, 10):
        height = 2 if offset % 50 == 0 else 1.2
        line(c, x+offset, y-height, x+offset, y+height)
    if label:
        text(c, x+50, y-5, "100 mm - verify after printing at actual size", 7, align="center")


def registration(c, x, y):
    c.setStrokeGray(.15)
    c.setLineWidth(.15*MM)
    line(c, x-1.7, y, x+1.7, y)
    line(c, x, y-1.7, x, y+1.7)
    c.circle(x*MM, y*MM, .8*MM, stroke=1, fill=0)


def arena(c, registration_marks=False):
    c.setStrokeGray(.62)
    c.setFillGray(.25)
    c.setLineWidth(.15*MM)
    c.setDash(2*MM, 2*MM)
    line(c, -RADIUS_MM, 0, RADIUS_MM, 0)
    line(c, 0, -RADIUS_MM, 0, RADIUS_MM)
    c.setDash()
    for value in range(-int(RADIUS_MM)+100, int(RADIUS_MM), 100):
        if value:
            line(c, value, -1.8, value, 1.8)
            line(c, -1.8, value, 1.8, value)
            if not registration_marks:
                text(c, value, -7, str(value), 9, align="center")
                text(c, 4, value-1, str(value), 9)
    c.setStrokeGray(0)
    c.setFillGray(0)
    c.setLineWidth(.5*MM)
    c.drawPath(circle_path(c), stroke=1, fill=0)
    c.setLineWidth(.22*MM)
    line(c, -6, 0, 6, 0)
    line(c, 0, -6, 0, 6)
    c.circle(0, 0, 2.5*MM, stroke=1, fill=0)
    if not registration_marks:
        # Tile artwork contains no labels that could split across sheet edges.
        text(c, 8, -9, "CENTER (0, 0)", 11, bold=True)
        text(c, RADIUS_MM-15, 6, "+X", 12, bold=True, align="right")
        text(c, 6, RADIUS_MM-16, "+Y", 12, bold=True)
        text(c, 0, -RADIUS_MM+16, "R = 1000 mm | inner diameter = 2000 mm", 12, align="center")
    if registration_marks:
        for col in range(1, COLS):
            x = ORIGIN_X+col*STEP_X+OVERLAP_MM/2
            for y in range(-int(RADIUS_MM), int(RADIUS_MM)+1, 100):
                registration(c, x, y)
        for row in range(1, ROWS):
            y = ORIGIN_Y+row*STEP_Y+OVERLAP_MM/2
            for x in range(-int(RADIUS_MM), int(RADIUS_MM)+1, 100):
                registration(c, x, y)


def new_canvas(path, pagesize):
    c = canvas.Canvas(str(path), pagesize=pagesize, pageCompression=1, invariant=1)
    c.setTitle("BARI arena - 2000 mm inner diameter - print at actual size")
    c.setAuthor("BARI preliminary simulation project")
    c.setSubject("Nominal vector print template; physical scale and wall construction unverified")
    c.setViewerPreference("PrintScaling", "None")
    return c


def guide(c):
    c.setFillColorRGB(.08, .18, .25)
    text(c, 18, 281, "BARI / ARENA SETUP", 23, bold=True)
    text(c, 18, 268, "1:1 template | inner diameter 2000 mm | radius 1000 mm", 11)
    c.setFillColorRGB(.91, .95, .96)
    c.roundRect(18*MM, 244*MM, 174*MM, 15*MM, 2*MM, stroke=0, fill=1)
    c.setFillColorRGB(.08, .18, .25)
    text(c, 23, 249, "PRINT AT 100% / ACTUAL SIZE", 15, bold=True)
    steps = [
        ("1  Check a test print.", "The 100 mm ruler must measure 100 mm. Disable Fit / Shrink / Poster."),
        ("2  Print tile pages 2-97 on A4.", "Keep portrait orientation. Headers, frames and rulers must be visible."),
        ("3  Assemble A1 at top left; H12 at bottom right.", "Trim white margins to the frame. Overlap artwork by 10 mm; align crosses."),
        ("4  Tape from the back and measure both diameters.", "Measure through the center, horizontally and vertically, before building."),
        ("5  Place the wall inner face on the solid circle centerline.", "Wall thickness, height, material and IR response need physical verification.")]
    for index, (title, detail) in enumerate(steps):
        y = 231-index*18
        c.setFillGray(0)
        text(c, 18, y, title, 10, bold=True)
        text(c, 18, y-6, detail, 8.3)
    c.setFillColorRGB(.08, .18, .25)
    text(c, 18, 134, "ASSEMBLY MAP / 96 TILES", 11, bold=True)
    cell_w, cell_h = 174/COLS, 60/ROWS
    for row in range(ROWS):
        for col in range(COLS):
            x, y = 18+col*cell_w, 66+(ROWS-1-row)*cell_h
            c.setFillColorRGB(.96, .97, .98)
            c.setStrokeColorRGB(.65, .72, .76)
            c.setLineWidth(.2*MM)
            c.rect(x*MM, y*MM, cell_w*MM, cell_h*MM, stroke=1, fill=1)
            c.setFillGray(0)
            text(c, x+cell_w/2, y+cell_h/2-.8, f"{chr(65+row)}{col+1}/p{2+row*COLS+col:02d}", 6.5, align="center")
    c.setFillGray(.2)
    text(c, 18, 54, "Measured inner diameter: X __________ mm   Y __________ mm", 9)
    text(c, 18, 46, "Wall height/material/IR check: ______________________________", 9)
    ruler(c, 18, 29)
    text(c, 18, 12, "Guide is schematic. Arena artwork is actual size. Physical build is not yet verified.", 7.5)
    text(c, 192, 12, f"1 / {TOTAL_PAGES}", 7.5, align="right")
    c.showPage()


def build(output, config=None):
    config = (config or Config()).validate()
    if abs(config.arena_radius-1.0) > 1e-12:
        raise ValueError("The physical BARI template requires the specified radius 1.0 m")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    full = output/"bari-arena-full-size.pdf"
    tiled = output/"bari-arena-a4-tiles.pdf"
    c = new_canvas(full, (PAGE_MM*MM, PAGE_MM*MM))
    c.saveState(); c.translate(PAGE_MM/2*MM, PAGE_MM/2*MM); arena(c); c.restoreState()
    text(c, 50, PAGE_MM-20, "BARI / 1:1 ARENA TEMPLATE", 28, bold=True)
    text(c, 50, PAGE_MM-35, "Solid circle centerline: inner diameter 2000 mm. Place the wall inner face here.", 15)
    text(c, 50, 31, "Print at actual size. Verify the ruler and both 2000 mm diameters before building.", 14)
    text(c, 50, 16, "Axes are assembly aids. Wall height, thickness, material and IR response remain unmeasured.", 10)
    ruler(c, PAGE_MM-280, 25)
    c.showPage(); c.save()
    c = new_canvas(tiled, A4)
    guide(c)
    tiles = []
    for row in range(ROWS):
        for col in range(COLS):
            label = f"{chr(65+row)}{col+1}"
            origin_x = ORIGIN_X+col*STEP_X
            origin_y = ORIGIN_Y+(ROWS-1-row)*STEP_Y
            x, y, width, height = CONTENT
            c.saveState()
            clip = c.beginPath(); clip.rect(x*MM, y*MM, width*MM, height*MM)
            c.clipPath(clip, stroke=0, fill=0)
            c.translate((x-origin_x)*MM, (y-origin_y)*MM)
            arena(c, registration_marks=True)
            c.restoreState()
            c.setStrokeGray(.5); c.setLineWidth(.15*MM); c.setDash(1*MM, 1*MM)
            c.rect(x*MM, y*MM, width*MM, height*MM, stroke=1, fill=0); c.setDash()
            c.setFillGray(0)
            text(c, 10, 288, "BARI ARENA / 2000 mm INNER DIAMETER", 9, bold=True)
            text(c, 10, 282, "100% / ACTUAL SIZE | 10 mm overlap | match crosses", 8)
            text(c, 200, 287, f"{label}  |  {2+row*COLS+col:02d}/{TOTAL_PAGES}", 11, bold=True, align="right")
            ruler(c, 55, 10.5)
            c.showPage()
            tiles.append({"id": label, "page": 2+row*COLS+col,
                          "world_bounds_mm": [origin_x, origin_y, origin_x+width, origin_y+height],
                          "page_content_mm": list(CONTENT),
                          "translation_pt": [(x-origin_x)*MM, (y-origin_y)*MM]})
    c.save()
    manifest = {"version": "bari-arena-template-v2", "nominal_radius_mm": RADIUS_MM, "nominal_inner_diameter_mm": 2*RADIUS_MM,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "runtime_versions": {"python": platform.python_version(), "reportlab": reportlab.Version, "pypdf": pypdf.__version__},
                "config_sha256": config.digest(), "full_page_mm": [PAGE_MM, PAGE_MM], "tile_paper_mm": [210, 297],
                "tile_grid": [COLS, ROWS], "tile_overlap_mm": OVERLAP_MM, "tiles": tiles,
                "print_scale": "actual size / 100%; PDF PrintScaling=None; ruler and assembled diameters must be measured",
                "boundary": "wall inner face at solid-line centerline; no wall thickness/height assumed",
                "physical_measurements": {"diameter_x_mm": None, "diameter_y_mm": None, "wall_height_mm": None,
                                          "wall_material": None, "ir_response": None},
                "artifacts": []}
    for path in (full, tiled):
        raw = path.read_bytes()
        manifest["artifacts"].append({"path": str(path), "pages": len(PdfReader(path).pages),
                                       "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    (output/"arena-template-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", default="output/pdf")
    args = parser.parse_args()
    result = build(args.output, Config.load(args.config))
    print(json.dumps({"output": args.output, "inner_diameter_mm": result["nominal_inner_diameter_mm"],
                      "pdf_pages": [item["pages"] for item in result["artifacts"]],
                      "tiles": len(result["tiles"]), "physical_verification": "pending"}))
