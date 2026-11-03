import importlib.util
import math
from pathlib import Path
import pytest

pytest.importorskip("reportlab")
pypdf = pytest.importorskip("pypdf")
module_path = Path(__file__).resolve().parents[1]/"hardware"/"arena_template.py"
spec = importlib.util.spec_from_file_location("arena_template", module_path)
template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(template)


def test_generated_pdf_keeps_physical_units_page_sizes_and_calibration_rulers(tmp_path):
    manifest = template.build(tmp_path)
    assert [entry["pages"] for entry in manifest["artifacts"]] == [1, 97]
    for index, entry in enumerate(manifest["artifacts"]):
        reader = pypdf.PdfReader(entry["path"])
        assert reader.trailer["/Root"]["/ViewerPreferences"]["/PrintScaling"] == "/None"
        for page in reader.pages:
            width = float(page.mediabox.width)*25.4/72
            height = float(page.mediabox.height)*25.4/72
            assert (width, height) == pytest.approx((2100, 2100) if index == 0 else (210, 297), abs=.001)
            assert float(page.get("/UserUnit", 1)) == 1
            operations = page.get_contents().operations
            found_ruler = False
            for position, (args, operator) in enumerate(operations):
                if operator == b"cm":
                    assert [float(value) for value in args[:4]] == pytest.approx([1, 0, 0, 1])
                if operator == b"m" and position+1 < len(operations):
                    end, next_operator = operations[position+1]
                    if next_operator == b"l":
                        dx, dy = float(end[0]-args[0])*25.4/72, float(end[1]-args[1])*25.4/72
                        found_ruler |= abs(dx-100) < .001 and abs(dy) < .001
            assert found_ruler


def test_pdf_boundary_is_a_2000_mm_circle_and_tiles_preserve_its_coordinates(tmp_path):
    manifest = template.build(tmp_path)
    full = pypdf.PdfReader(manifest["artifacts"][0]["path"])
    ops = full.pages[0].get_contents().operations
    start = next(index for index, (args, operator) in enumerate(ops)
                 if operator == b"m" and abs(float(args[0])*25.4/72-1000) < .001 and abs(float(args[1])) < .001)
    current = [float(value)*25.4/72 for value in ops[start][0]]
    for args, operator in ops[start+1:start+25]:
        assert operator == b"c"
        points = [current, *[[float(args[i])*25.4/72, float(args[i+1])*25.4/72] for i in (0, 2, 4)]]
        for index in range(101):
            t = index/100
            x, y = [sum(points[i][axis]*math.comb(3, i)*(1-t)**(3-i)*t**i for i in range(4)) for axis in (0, 1)]
            assert abs(math.hypot(x, y)-1000) < .001  # Digital numeric precision, not a printer tolerance.
        current = points[-1]
    tiled = pypdf.PdfReader(manifest["artifacts"][1]["path"])
    for tile in manifest["tiles"]:
        page = tiled.pages[tile["page"]-1]
        transforms = [[float(value) for value in args] for args, op in page.get_contents().operations if op == b"cm"]
        assert any(transform[4:] == pytest.approx(tile["translation_pt"], abs=.001) for transform in transforms)
        assert tile["id"] in page.extract_text()
    bounds = [tile["world_bounds_mm"] for tile in manifest["tiles"]]
    for angle in range(360):
        x, y = 1000*math.cos(math.radians(angle)), 1000*math.sin(math.radians(angle))
        assert any(left <= x <= right and bottom <= y <= top for left, bottom, right, top in bounds)
