"""从中文 Markdown 生成排版后的纵向跌倒预警功能说明 Word 文档。"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "docs" / "老年人纵向跌倒预警功能说明.md"
OUTPUT = PROJECT_ROOT / "docs" / "老年人纵向跌倒预警功能说明.docx"

NAVY = "12345B"
OLIVE = "707A1A"
PALE_OLIVE = "F0F1DC"
ORANGE = "D95F0E"
PALE_ORANGE = "FFF0E4"
SURFACE = "F6F7F5"
GRID = "D6D9CF"
TEXT = "243247"
MUTED = "5B6573"


def set_run_font(run, east_asia: str = "微软雅黑", latin: str = "Aptos") -> None:
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)


def set_style_fonts(style, east_asia: str, latin: str) -> None:
    r_pr = style._element.get_or_add_rPr()
    r_fonts = r_pr.rFonts
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, r_fonts)
    r_fonts.set(qn("w:eastAsia"), east_asia)
    r_fonts.set(qn("w:ascii"), latin)
    r_fonts.set(qn("w:hAnsi"), latin)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_keep_with_next(paragraph) -> None:
    paragraph.paragraph_format.keep_with_next = True


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("第 ")
    set_run_font(run)
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)
    suffix = paragraph.add_run(" 页")
    set_run_font(suffix)


def add_inline(paragraph, text: str) -> None:
    """支持 Markdown 的粗体和行内代码，保持正文可维护。"""
    pattern = re.compile(r"(\*\*.*?\*\*|`.*?`)")
    position = 0
    for match in pattern.finditer(text):
        if match.start() > position:
            run = paragraph.add_run(text[position:match.start()])
            set_run_font(run, east_asia="宋体", latin="Aptos")
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
            run.font.color.rgb = RGBColor.from_string(NAVY)
            set_run_font(run)
        else:
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "等线")
            run.font.color.rgb = RGBColor.from_string(OLIVE)
            run.font.size = Pt(9.5)
        position = match.end()
    if position < len(text):
        run = paragraph.add_run(text[position:])
        set_run_font(run, east_asia="宋体", latin="Aptos")


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(TEXT)
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.widow_control = True
    set_style_fonts(normal, "宋体", "Aptos")

    for style_name, size, color, before, after in (
        ("Title", 28, NAVY, 0, 18),
        ("Subtitle", 13, OLIVE, 0, 10),
        ("Heading 1", 17, NAVY, 18, 8),
        ("Heading 2", 13, OLIVE, 14, 6),
    ):
        style = styles[style_name]
        style.font.size = Pt(size)
        style.font.bold = style_name != "Subtitle"
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        set_style_fonts(style, "微软雅黑", "Aptos Display")

    caption = styles["Caption"]
    caption.font.size = Pt(9)
    caption.font.italic = False
    caption.font.color.rgb = RGBColor.from_string(MUTED)
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(8)
    set_style_fonts(caption, "宋体", "Aptos")

    for name in ("List Bullet", "List Number"):
        style = styles[name]
        style.font.size = Pt(10.5)
        style.font.color.rgb = RGBColor.from_string(TEXT)
        style.paragraph_format.space_after = Pt(3)
        set_style_fonts(style, "宋体", "Aptos")


def add_cover(doc: Document) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(42)
    run = p.add_run("纵向个人基线 · 功能说明")
    run.bold = True
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor.from_string(OLIVE)
    set_run_font(run)

    title = doc.add_paragraph(style="Title")
    title.add_run("老年人纵向跌倒\n预警系统")
    title.paragraph_format.space_before = Pt(10)
    title.paragraph_format.space_after = Pt(18)

    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.add_run("个人步态画像 · 14 日稳健基线 · Landmark Cox 动态风险")

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    cell = table.cell(0, 0)
    set_cell_shading(cell, PALE_OLIVE)
    set_cell_margins(cell, top=220, start=260, bottom=220, end=260)
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    add_inline(
        paragraph,
        "本说明聚焦新增的天/周级纵向功能。系统以步态和姿态为主，心理因素仅作辅助；没有真实纵向标签和独立校准时，不输出未来跌倒概率。",
    )

    notice = doc.add_table(rows=1, cols=1)
    notice.alignment = WD_TABLE_ALIGNMENT.CENTER
    notice.autofit = True
    notice_cell = notice.cell(0, 0)
    set_cell_shading(notice_cell, PALE_ORANGE)
    set_cell_margins(notice_cell, top=170, start=230, bottom=170, end=230)
    notice_p = notice_cell.paragraphs[0]
    notice_run = notice_p.add_run("使用边界：本功能用于统计变化监测与人工复核，不替代临床筛查、照护判断或紧急处置。")
    notice_run.bold = True
    notice_run.font.color.rgb = RGBColor.from_string(ORANGE)
    set_run_font(notice_run)

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(150)
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    meta.add_run("Demo 1.0\n2026 年 11 月 11 日")
    for run in meta.runs:
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor.from_string(MUTED)
        set_run_font(run)
    doc.add_page_break()


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    width = max(len(row) for row in rows)
    table = doc.add_table(rows=1, cols=width)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    table.style = "Table Grid"
    header = table.rows[0]
    set_repeat_table_header(header)
    for index in range(width):
        cell = header.cells[index]
        set_cell_shading(cell, NAVY)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(0)
        text = rows[0][index] if index < len(rows[0]) else ""
        run = paragraph.add_run(text)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(9)
        set_run_font(run)
    for row_index, values in enumerate(rows[1:], start=1):
        cells = table.add_row().cells
        for index in range(width):
            cell = cells[index]
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if row_index % 2 == 0:
                set_cell_shading(cell, SURFACE)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            value = values[index] if index < len(values) else ""
            add_inline(paragraph, value)
            for run in paragraph.runs:
                run.font.size = Pt(8.8)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_callout(doc: Document, text: str) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_shading(cell, PALE_ORANGE if "重要" in text or "Flask" in text else PALE_OLIVE)
    set_cell_margins(cell, top=140, start=210, bottom=140, end=210)
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    add_inline(paragraph, text)
    for run in paragraph.runs:
        if "重要" in text or "不能" in text or "不替代" in text:
            run.font.color.rgb = RGBColor.from_string(ORANGE)


def add_code_block(doc: Document, lines: list[str]) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_shading(cell, "F1F3F4")
    set_cell_margins(cell, top=120, start=180, bottom=120, end=180)
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run("\n".join(lines))
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor.from_string(TEXT)


def parse_markdown(doc: Document, source: str) -> None:
    lines = source.splitlines()
    index = 0
    in_body = False
    while index < len(lines):
        line = lines[index].rstrip()
        if line.startswith("## 摘要"):
            in_body = True
        if not in_body:
            index += 1
            continue
        if not line:
            index += 1
            continue
        if line.startswith("```"):
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            add_code_block(doc, code_lines)
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[index + 1]):
            table_lines = [line]
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
            add_table(doc, rows)
            continue
        if line.startswith("### "):
            paragraph = doc.add_paragraph(line[4:], style="Heading 2")
            set_keep_with_next(paragraph)
        elif line.startswith("## "):
            paragraph = doc.add_paragraph(line[3:], style="Heading 1")
            set_keep_with_next(paragraph)
        elif line.startswith("> "):
            add_callout(doc, line[2:])
        elif re.match(r"^\d+\. ", line):
            paragraph = doc.add_paragraph(style="List Number")
            add_inline(paragraph, re.sub(r"^\d+\. ", "", line))
        elif line.startswith("- "):
            paragraph = doc.add_paragraph(style="List Bullet")
            add_inline(paragraph, line[2:])
        elif re.match(r"^!\[.*?\]\((.*?)\)$", line):
            match = re.match(r"^!\[(.*?)\]\((.*?)\)$", line)
            assert match is not None
            image_path = SOURCE.parent / match.group(2)
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run()
            run.add_picture(str(image_path), width=Cm(16.2))
        elif line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            paragraph = doc.add_paragraph(style="Caption")
            add_inline(paragraph, line[1:-1])
        elif line == "---":
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.space_after = Pt(0)
        else:
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_inline(paragraph, line)
        index += 1


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.1)
    section.right_margin = Cm(2.1)
    section.header_distance = Cm(0.8)
    section.footer_distance = Cm(0.8)

    footer = section.footer
    table = footer.add_table(rows=1, cols=2, width=Cm(16.8))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    left, right = table.rows[0].cells
    left_p = left.paragraphs[0]
    left_run = left_p.add_run("老年人纵向跌倒预警功能说明")
    left_run.font.size = Pt(8)
    left_run.font.color.rgb = RGBColor.from_string(MUTED)
    set_run_font(left_run)
    right_p = right.paragraphs[0]
    add_page_number(right_p)
    for run in right_p.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(MUTED)


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    doc = Document()
    configure_document(doc)
    configure_styles(doc)
    props = doc.core_properties
    props.title = "老年人纵向跌倒预警功能说明"
    props.subject = "个人步态画像、14日稳健基线与Landmark Cox动态预警"
    props.author = "项目组"
    props.keywords = "老年人, 跌倒预警, 步态画像, 个人基线, Landmark Cox"
    add_cover(doc)
    parse_markdown(doc, markdown)
    doc.save(OUTPUT)
    print(f"已生成：{OUTPUT}")


if __name__ == "__main__":
    main()
