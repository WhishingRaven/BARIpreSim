Historical radius-0.5 m report. Commands and links below describe the original
creation context. Use the separately archived source/files and
[historical manifest](arena-template-verification-radius-0.5m.json); active templates
now have radius 1 m.

# Digital arena template verification

The radius-0.5 m arena now has two generated vector print artifacts: a single
1,100 × 1,100 mm sheet and an A4 bundle with one assembly guide plus 24 tiles.
The nominal inner diameter is 1,000 mm. A 10 mm overlap, matching crosses, center
mark, axes, tile labels, and 100 mm scale bars support assembly. The solid circle's
centerline marks the wall's inner face; wall height, thickness, and material are
not inferred from the template.

The [generator](../hardware/arena_template.py) uses physical PDF points and sets
`PrintScaling=None`. Content-stream checks verify page dimensions, unscaled
transforms, 100 mm ruler geometry, tile translations, and complete circular-boundary
coverage. Sampling the actual cubic boundary paths verifies their nominal 500 mm
radius within 0.001 mm of digital geometry. That precision is a file check and is
not a physical printer tolerance. The complete software suite passed **85 tests**;
its immutable output is [verification-arena-template.txt](verification-arena-template.txt).

Poppler rendered every A4 page at 120 dpi and the full sheet at 60 dpi. Visual
review covered the assembly guide, full sheet, all tile geometry, and all headers,
footers, scale labels, and page counters. Final render composites and a manifest
binding them to the delivered PDF hashes are retained in `output/pdf/qa/`.
No clipping, overlapping text, broken glyphs, or incomplete page labels remained.
The [machine-readable verification](arena-template-verification.json) records
artifact and generator hashes and distinguishes these checks from physical work.

Install optional dependencies and reproduce the files with:

```sh
python -m pip install -e '.[hardware]'
python hardware/arena_template.py --output output/pdf
```

Use the [hardware print instructions](../hardware/README.md) to print at actual
size, measure the 100 mm ruler, trim and overlap the tile frames, and measure both
assembled diameters. Copy [the physical record](../hardware/arena.template.json)
into the experiment output before recording measurements. Printing, assembly,
wall material and height, IR response, boundary sensing, stopping, and robot
calibration remain unverified. No physical-MVP TODO is marked complete.
