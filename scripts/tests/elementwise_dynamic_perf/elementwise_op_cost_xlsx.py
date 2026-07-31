"""Write the elementwise comparison report as a dependency-free XLSX file."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
LABEL_COLUMNS = ("bound", "scalar_ops", "dtype", "first_shape")
BASE_COLUMNS = (*LABEL_COLUMNS, "runtime_shape")
ROUND_TILING_SUFFIXES = ("_dynamic_tiling", "_custom_tiling")

ET.register_namespace("", SHEET_NS)
ET.register_namespace("r", REL_NS)


def _tag(name: str) -> str:
    return f"{{{SHEET_NS}}}{name}"


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _merge_ranges(
    rows: list[dict[str, str]],
    columns: tuple[str, ...],
) -> list[tuple[int, int, int]]:
    ranges = []
    for column_index, column in enumerate(LABEL_COLUMNS, start=1):
        if column not in columns:
            continue
        key_columns = LABEL_COLUMNS[:column_index]
        start = 0
        while start < len(rows):
            key = tuple(rows[start].get(name, "") for name in key_columns)
            end = start + 1
            while end < len(rows):
                next_key = tuple(rows[end].get(name, "") for name in key_columns)
                if next_key != key:
                    break
                end += 1
            if end - start > 1:
                ranges.append((start + 2, end + 1, column_index))
            start = end
    for column_index, column in enumerate(columns, start=1):
        if column.endswith(ROUND_TILING_SUFFIXES):
            ranges.extend(
                _value_merge_ranges(rows, column, column_index, group_aware=False)
            )
        elif column.endswith("_group_tiling"):
            ranges.extend(
                _value_merge_ranges(rows, column, column_index, group_aware=True)
            )
    return ranges


def _group_ids(value: str) -> tuple[str, ...]:
    records = json.loads(value)
    return tuple(
        str(record["group_id"])
        for record in records
        if record.get("group_id") is not None
    )


def _value_merge_ranges(
    rows: list[dict[str, str]],
    column: str,
    column_index: int,
    group_aware: bool,
) -> list[tuple[int, int, int]]:
    ranges = []
    start = 0
    while start < len(rows):
        value = rows[start].get(column, "")
        if not value:
            start += 1
            continue
        scene = tuple(rows[start].get(name, "") for name in LABEL_COLUMNS)
        group_ids = _group_ids(value) if group_aware else ()
        if group_aware and not group_ids:
            start += 1
            continue
        key = (scene, group_ids, value) if group_aware else (scene, value)
        end = start + 1
        while end < len(rows):
            next_value = rows[end].get(column, "")
            next_scene = tuple(rows[end].get(name, "") for name in LABEL_COLUMNS)
            next_group_ids = _group_ids(next_value) if group_aware and next_value else ()
            next_key = (
                (next_scene, next_group_ids, next_value)
                if group_aware
                else (next_scene, next_value)
            )
            if next_key != key:
                break
            end += 1
        if end - start > 1:
            ranges.append((start + 2, end + 1, column_index))
        start = end
    return ranges


def _cell_style(column: str, bound: str) -> int:
    if column in LABEL_COLUMNS:
        return 3 if bound == "memory_bound" else 2
    if column == "runtime_shape":
        return 4
    if column.endswith("_tiling"):
        return 9
    if "ratio_of_lift" in column:
        return 8
    if "ratio" in column:
        return 6
    if column.startswith("cuda_"):
        return 5
    if column.startswith("npu_"):
        return 7
    return 4


def _column_width(
    rows: list[dict[str, str]],
    column: str,
) -> int:
    value_width = max((len(row.get(column, "")) for row in rows), default=0)
    if column.endswith("_tiling"):
        return min(max(value_width + 2, 18), 48)
    if column not in BASE_COLUMNS:
        return max(value_width + 2, 10)
    return min(max(len(column) + 2, value_width + 2, 12), 28)


def _display_value(column: str, value: str) -> str:
    if not value or not column.endswith("_tiling"):
        return value
    try:
        return json.dumps(json.loads(value), indent=2, sort_keys=True)
    except json.JSONDecodeError:
        return value


def _append_cell(
    row_element: ET.Element,
    row_number: int,
    column_index: int,
    value: str,
    style: int,
    numeric: bool,
) -> None:
    reference = f"{_column_name(column_index)}{row_number}"
    attributes = {"r": reference, "s": str(style)}
    if numeric and value:
        try:
            number = float(value)
        except ValueError:
            pass
        else:
            cell = ET.SubElement(row_element, _tag("c"), attributes)
            ET.SubElement(cell, _tag("v")).text = str(number)
            return
    if not value:
        ET.SubElement(row_element, _tag("c"), attributes)
        return
    attributes["t"] = "inlineStr"
    cell = ET.SubElement(row_element, _tag("c"), attributes)
    inline_string = ET.SubElement(cell, _tag("is"))
    text_attributes = {"{http://www.w3.org/XML/1998/namespace}space": "preserve"} if "\n" in value else {}
    ET.SubElement(inline_string, _tag("t"), text_attributes).text = value


def _worksheet_xml(
    rows: list[dict[str, str]],
    columns: tuple[str, ...],
) -> bytes:
    worksheet = ET.Element(_tag("worksheet"))
    views = ET.SubElement(worksheet, _tag("sheetViews"))
    view = ET.SubElement(views, _tag("sheetView"), {"workbookViewId": "0"})
    ET.SubElement(
        view,
        _tag("pane"),
        {"ySplit": "1", "topLeftCell": "A2", "activePane": "bottomLeft", "state": "frozen"},
    )
    ET.SubElement(view, _tag("selection"), {"pane": "bottomLeft", "activeCell": "A2", "sqref": "A2"})
    ET.SubElement(worksheet, _tag("sheetFormatPr"), {"defaultRowHeight": "20"})

    widths = ET.SubElement(worksheet, _tag("cols"))
    for index, column in enumerate(columns, start=1):
        ET.SubElement(
            widths,
            _tag("col"),
            {
                "min": str(index),
                "max": str(index),
                "width": str(_column_width(rows, column)),
                "customWidth": "1",
            },
        )

    merge_ranges = _merge_ranges(rows, columns)
    merged_children = {
        (row_number, column_index)
        for start, end, column_index in merge_ranges
        for row_number in range(start + 1, end + 1)
    }
    sheet_data = ET.SubElement(worksheet, _tag("sheetData"))
    header = ET.SubElement(sheet_data, _tag("row"), {"r": "1", "ht": "60", "customHeight": "1"})
    for column_index, column in enumerate(columns, start=1):
        _append_cell(header, 1, column_index, column, 1, numeric=False)

    for row_number, row in enumerate(rows, start=2):
        row_element = ET.SubElement(
            sheet_data,
            _tag("row"),
            {"r": str(row_number), "ht": "20", "customHeight": "1"},
        )
        for column_index, column in enumerate(columns, start=1):
            if (row_number, column_index) in merged_children:
                continue
            _append_cell(
                row_element,
                row_number,
                column_index,
                _display_value(column, row.get(column, "")),
                _cell_style(column, row.get("bound", "")),
                numeric=column not in BASE_COLUMNS,
            )

    last_cell = f"{_column_name(len(columns))}{max(len(rows) + 1, 1)}"
    ET.SubElement(worksheet, _tag("autoFilter"), {"ref": f"A1:{last_cell}"})
    if merge_ranges:
        merged = ET.SubElement(worksheet, _tag("mergeCells"), {"count": str(len(merge_ranges))})
        for start, end, column_index in merge_ranges:
            column = _column_name(column_index)
            ET.SubElement(merged, _tag("mergeCell"), {"ref": f"{column}{start}:{column}{end}"})

    if rows:
        for priority, column_index in enumerate(
            (index for index, column in enumerate(columns, start=1) if "ratio" in column),
            start=1,
        ):
            column = _column_name(column_index)
            conditional = ET.SubElement(
                worksheet,
                _tag("conditionalFormatting"),
                {"sqref": f"{column}2:{column}{len(rows) + 1}"},
            )
            for offset, (operator, threshold) in enumerate(
                (("greaterThan", "1.15"), ("lessThan", "0.85"))
            ):
                rule = ET.SubElement(
                    conditional,
                    _tag("cfRule"),
                    {
                        "type": "cellIs",
                        "dxfId": "0",
                        "priority": str(priority * 2 + offset),
                        "operator": operator,
                    },
                )
                ET.SubElement(rule, _tag("formula")).text = threshold

    ET.SubElement(
        worksheet,
        _tag("pageMargins"),
        {"left": "0.25", "right": "0.25", "top": "0.5", "bottom": "0.5", "header": "0.2", "footer": "0.2"},
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
    <font><sz val="11"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="9">
    <fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF375A64"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFDCEAF7"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFDDEFE8"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFE8F1FA"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFF4D8"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFE4F0ED"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFBE5DF"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border/>
    <border>
      <left style="thin"/><right style="thin"/><top style="thin"/><bottom style="thin"/>
    </border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="10">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyAlignment="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0" applyAlignment="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0" applyAlignment="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1">
      <alignment horizontal="center" vertical="center"/>
    </xf>
    <xf numFmtId="164" fontId="0" fillId="5" borderId="1" xfId="0"
      applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center"/></xf>
    <xf numFmtId="164" fontId="0" fillId="6" borderId="1" xfId="0"
      applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center"/></xf>
    <xf numFmtId="164" fontId="0" fillId="7" borderId="1" xfId="0"
      applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center"/></xf>
    <xf numFmtId="164" fontId="0" fillId="8" borderId="1" xfId="0"
      applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1">
      <alignment horizontal="left" vertical="center" wrapText="1"/>
    </xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
  <dxfs count="1"><dxf><font><color rgb="FFE53935"/></font></dxf></dxfs>
  <tableStyles count="0" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>
</styleSheet>
"""


def write_xlsx_report(
    rows: list[dict[str, str]],
    output_path: Path,
    columns: tuple[str, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        workbook.writestr("_rels/.rels", ROOT_RELS_XML)
        workbook.writestr("xl/workbook.xml", WORKBOOK_XML)
        workbook.writestr("xl/_rels/workbook.xml.rels", WORKBOOK_RELS_XML)
        workbook.writestr("xl/styles.xml", STYLES_XML)
        workbook.writestr("xl/worksheets/sheet1.xml", _worksheet_xml(rows, columns))
