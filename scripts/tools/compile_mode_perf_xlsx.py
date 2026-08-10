"""Write the compile-mode comparison report as a dependency-free XLSX file."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
COLUMNS = (
    "case",
    "first_shape",
    "shape",
    "static_us",
    "static_tiling",
    "dynamic_us",
    "dynamic_static_ratio",
    "dynamic_tiling",
    "group_us",
    "group_static_ratio",
    "group_tiling",
)
NUMERIC_COLUMNS = {
    "static_us",
    "dynamic_us",
    "dynamic_static_ratio",
    "group_us",
    "group_static_ratio",
}

ET.register_namespace("", SHEET_NS)
ET.register_namespace("r", REL_NS)


def tag(name: str) -> str:
    return f"{{{SHEET_NS}}}{name}"


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def contiguous_ranges(rows, column, key_fn):
    ranges = []
    start = 0
    while start < len(rows):
        value = rows[start].get(column, "")
        if not value:
            start += 1
            continue
        key = key_fn(rows[start])
        end = start + 1
        while end < len(rows) and key_fn(rows[end]) == key:
            end += 1
        if end - start > 1:
            ranges.append((start + 2, end + 1, COLUMNS.index(column) + 1))
        start = end
    return ranges


def group_ids(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        records = json.loads(value)
    except json.JSONDecodeError:
        return ()
    return tuple(
        str(record["group_id"])
        for record in records
        if record.get("group_id") is not None
    )


def merge_ranges(rows):
    ranges = []
    ranges.extend(contiguous_ranges(rows, "case", lambda row: row.get("case", "")))
    ranges.extend(
        contiguous_ranges(
            rows,
            "first_shape",
            lambda row: (row.get("case", ""), row.get("first_shape", "")),
        )
    )
    ranges.extend(
        contiguous_ranges(
            rows,
            "dynamic_tiling",
            lambda row: (
                row.get("case", ""),
                row.get("first_shape", ""),
                row.get("dynamic_tiling", ""),
            ),
        )
    )
    ranges.extend(
        contiguous_ranges(
            rows,
            "group_tiling",
            lambda row: (
                row.get("case", ""),
                row.get("first_shape", ""),
                group_ids(row.get("group_tiling", "")),
                row.get("group_tiling", ""),
            ),
        )
    )
    return ranges


def display_value(column: str, value: str) -> str:
    if not value or not column.endswith("_tiling"):
        return value
    try:
        return json.dumps(json.loads(value), indent=2, sort_keys=True)
    except json.JSONDecodeError:
        return value


def cell_style(column: str) -> int:
    if column in ("case", "first_shape", "shape"):
        return 2
    if column.endswith("_tiling"):
        return 4
    if "ratio" in column:
        return 6
    if column.startswith("static_"):
        return 3
    if column.startswith("dynamic_"):
        return 5
    if column.startswith("group_"):
        return 7
    return 2


def column_width(rows, column: str) -> int:
    width = max([len(column), *(len(row.get(column, "")) for row in rows)]) + 2
    if column.endswith("_tiling"):
        return min(max(width, 18), 48)
    if column in ("case", "first_shape", "shape"):
        return min(max(width, 14), 42)
    return min(max(width, 12), 24)


def append_cell(row, row_number, column_index, value, style, numeric=False):
    reference = f"{column_name(column_index)}{row_number}"
    attributes = {"r": reference, "s": str(style)}
    if numeric and value:
        try:
            number = float(value)
        except ValueError:
            pass
        else:
            cell = ET.SubElement(row, tag("c"), attributes)
            ET.SubElement(cell, tag("v")).text = str(number)
            return
    if not value:
        ET.SubElement(row, tag("c"), attributes)
        return
    attributes["t"] = "inlineStr"
    cell = ET.SubElement(row, tag("c"), attributes)
    inline = ET.SubElement(cell, tag("is"))
    text_attributes = (
        {"{http://www.w3.org/XML/1998/namespace}space": "preserve"}
        if "\n" in value
        else {}
    )
    ET.SubElement(inline, tag("t"), text_attributes).text = value


def worksheet_xml(rows) -> bytes:
    worksheet = ET.Element(tag("worksheet"))
    views = ET.SubElement(worksheet, tag("sheetViews"))
    view = ET.SubElement(views, tag("sheetView"), {"workbookViewId": "0"})
    ET.SubElement(
        view,
        tag("pane"),
        {
            "ySplit": "1",
            "topLeftCell": "A2",
            "activePane": "bottomLeft",
            "state": "frozen",
        },
    )
    ET.SubElement(worksheet, tag("sheetFormatPr"), {"defaultRowHeight": "40"})

    widths = ET.SubElement(worksheet, tag("cols"))
    for index, column in enumerate(COLUMNS, start=1):
        ET.SubElement(
            widths,
            tag("col"),
            {
                "min": str(index),
                "max": str(index),
                "width": str(column_width(rows, column)),
                "customWidth": "1",
            },
        )

    ranges = merge_ranges(rows)
    merged_children = {
        (row_number, column_index)
        for start, end, column_index in ranges
        for row_number in range(start + 1, end + 1)
    }
    data = ET.SubElement(worksheet, tag("sheetData"))
    header = ET.SubElement(
        data,
        tag("row"),
        {"r": "1", "ht": "45", "customHeight": "1"},
    )
    for column_index, column in enumerate(COLUMNS, start=1):
        append_cell(header, 1, column_index, column, 1)

    for row_number, values in enumerate(rows, start=2):
        row = ET.SubElement(
            data,
            tag("row"),
            {"r": str(row_number), "ht": "40", "customHeight": "1"},
        )
        for column_index, column in enumerate(COLUMNS, start=1):
            if (row_number, column_index) in merged_children:
                continue
            append_cell(
                row,
                row_number,
                column_index,
                display_value(column, values.get(column, "")),
                cell_style(column),
                numeric=column in NUMERIC_COLUMNS,
            )

    last_cell = f"{column_name(len(COLUMNS))}{max(len(rows) + 1, 1)}"
    ET.SubElement(worksheet, tag("autoFilter"), {"ref": f"A1:{last_cell}"})
    if ranges:
        merged = ET.SubElement(
            worksheet,
            tag("mergeCells"),
            {"count": str(len(ranges))},
        )
        for start, end, column_index in ranges:
            column = column_name(column_index)
            ET.SubElement(
                merged,
                tag("mergeCell"),
                {"ref": f"{column}{start}:{column}{end}"},
            )

    for priority, column in enumerate(
        ("dynamic_static_ratio", "group_static_ratio"), start=1
    ):
        column_letter = column_name(COLUMNS.index(column) + 1)
        conditional = ET.SubElement(
            worksheet,
            tag("conditionalFormatting"),
            {"sqref": f"{column_letter}2:{column_letter}{len(rows) + 1}"},
        )
        rule = ET.SubElement(
            conditional,
            tag("cfRule"),
            {
                "type": "cellIs",
                "dxfId": "0",
                "priority": str(priority),
                "operator": "greaterThan",
            },
        )
        ET.SubElement(rule, tag("formula")).text = "1.15"

    ET.SubElement(
        worksheet,
        tag("pageMargins"),
        {
            "left": "0.25",
            "right": "0.25",
            "top": "0.5",
            "bottom": "0.5",
            "header": "0.2",
            "footer": "0.2",
        },
    )
    return ET.tostring(worksheet, encoding="utf-8", xml_declaration=True)


CONTENT_TYPES_XML = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml"
    ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml"
    ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml"
    ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>
"""
ROOT_RELS_XML = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="xl/workbook.xml"/>
</Relationships>
"""
WORKBOOK_XML = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="comparison" sheetId="1" r:id="rId1"/></sheets>
</workbook>
"""
WORKBOOK_RELS_XML = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
    Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
    Target="styles.xml"/>
</Relationships>
"""
STYLES_XML = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1"><numFmt numFmtId="164" formatCode="0.000"/></numFmts>
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="8">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF375A64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFDDEFE8"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFE8F1FA"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF5F5F5"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFF4D8"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFBE5DF"/></patternFill></fill>
  </fills>
  <borders count="2"><border/><border><left style="thin"/><right style="thin"/>
    <top style="thin"/><bottom style="thin"/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="8">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyAlignment="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0" applyAlignment="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="164" fontId="0" fillId="4" borderId="1" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="0" fontId="0" fillId="5" borderId="1" xfId="0" applyAlignment="1">
      <alignment horizontal="left" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="164" fontId="0" fillId="6" borderId="1" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="164" fontId="0" fillId="6" borderId="1" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="164" fontId="0" fillId="7" borderId="1" xfId="0" applyNumberFormat="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
  <dxfs count="1"><dxf><font><color rgb="FFE53935"/></font></dxf></dxfs>
</styleSheet>
"""


def write_xlsx_report(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        workbook.writestr("_rels/.rels", ROOT_RELS_XML)
        workbook.writestr("xl/workbook.xml", WORKBOOK_XML)
        workbook.writestr("xl/_rels/workbook.xml.rels", WORKBOOK_RELS_XML)
        workbook.writestr("xl/styles.xml", STYLES_XML)
        workbook.writestr("xl/worksheets/sheet1.xml", worksheet_xml(rows))
