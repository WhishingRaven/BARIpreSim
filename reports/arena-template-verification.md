# Digital arena template verification

The user revised the arena to radius 1.0 m, inner diameter 2.0 m. Generated vector
files comprise a 2,100 mm square full-size sheet and an A4 bundle of one guide plus
96 tiles (12 columns by 8 rows, 10 mm overlap).

The complete suite passed **87 tests**, including physical PDF point units, page
sizes, 100 mm rulers, the actual 1,000 mm-radius boundary, tile translations, and
coverage. See [verification-arena-radius-1m.txt](verification-arena-radius-1m.txt).

All 97 A4 pages and the full sheet are rendered and visually reviewed. The guide,
all tile geometry, every header/footer and page counter, and full-sheet text
passed final review without unresolved layout defects. Print at actual size and
measure the 100 mm ruler before assembly. Render composites and hashes are in
`output/pdf/qa/` and the
[machine-readable report](arena-template-verification.json). Physical printing,
assembled dimensions, walls, IR response, stopping, and calibration remain pending.

Earlier radius-0.5 m files, generator, tests, and visual evidence remain in
`output/pdf/arena-radius-0.5m/`, with separate
[historical verification](arena-template-verification-radius-0.5m.json).
