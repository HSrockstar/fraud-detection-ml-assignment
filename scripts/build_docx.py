"""用教师模板的段落与版面生成报告；使用文档运行时的 Python 执行。"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from docx import Document
from docx.shared import Inches
from lxml import etree as ET

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "作业要求/机器学习报告模板-2026.docx"
OUTPUT = ROOT / "reports/信用卡欺诈预测_分类实验报告.docx"
QA = ROOT / "artifacts/document_qa"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"w": W, "r": R, "a": "http://schemas.openxmlformats.org/drawingml/2006/main", "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"}


def el(name, **attrs):
    item = ET.Element(f"{{{W}}}{name}")
    for k, v in attrs.items(): item.set(f"{{{W}}}{k}", str(v))
    return item


def xml_bytes(item):
    return ET.tostring(item, encoding="UTF-8", xml_declaration=True, standalone=True)


def main():
    QA.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(REFERENCE) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    document = ET.fromstring(parts["word/document.xml"])
    body = document.find("w:body", NS)
    paras = body.findall("w:p", NS)
    prototypes = {"body": copy.deepcopy(paras[3]), "h1": copy.deepcopy(paras[0]), "h2": copy.deepcopy(paras[1])}
    team_prototype = copy.deepcopy(body.find("w:tbl", NS))
    section = copy.deepcopy(body.find("w:sectPr", NS))
    inventory = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name,data in parts.items()}
    (QA / "template-package-inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    # 依据已渲染的两页原件和结构审计建立模板执行约定。
    contract = f"""# 模板执行约定

Reference: {REFERENCE}
SHA256: {hashlib.sha256(REFERENCE.read_bytes()).hexdigest()}
Reference render: artifacts/document_qa/reference/page-1.png, page-2.png，共 2 页。
Section/style evidence: template-style-evidence.json 与 section_audit 输出。

单节 A4 11906 × 16838 twips，纵向；上下边距 1440、左右 1800 twips。
页眉距离 851、页脚 992 twips；原件无可见页眉页脚。保留 sectPr、文档网格和单栏。
正文复用 word/document.xml/body/p[4] 的 pPr/rPr：12 pt、1.5 倍行距、首行 420 twips。
正文显式启用孤行控制；参考资料沿用 12 pt 字体，左对齐、1.2 倍行距、每条保持同页。
一级章节复用 p[1]：14 pt 黑色粗体、居中、段前 480/段后 120 twips。
二级章节复用 p[2]：12 pt 粗体、段前后 120 twips；移除原自动编号，文本明确编号。
表格复用原两列表格，网格宽度 1835/6461 twips，增加第四位成员的行，贡献留白。
模型数值表按同一页面内容宽度增加列，12 pt 正文、表头粗体、固定列宽、自动行高。
新增图片使用内联排版，宽 5.7 英寸，图注置于后方，防止图与图注拆页。
标题采用同源一级标题字体和居中方式，新增 Title 语义样式；不添加装饰线。
编辑槽：原表格改为四人名单；模板提示、回归示例、红色提交说明全部替换为实际分类实验。
章节顺序保留课题综述、数据准备、模型搭建、训练评估、可视化、优化分析、拓展探索，增加总结与参考资料。
正文容量按内容自然续页，要求不超过 15 页；不通过缩小正文满足页数。
保留所有原 styles 定义、numbering、theme、settings、fontTable、脚注尾注和元数据包内容；仅允许新增 Title 样式。
允许变更的原包部件：document.xml、document.xml.rels、Content_Types.xml、styles.xml（仅新增 Title）。其余原部件逐字节核验。
新增图像关系和 word/media/report_*.png；不改变原关系。成员分工按用户要求为空。
Final QA: 重新渲染并逐页检查，比较页幅、正文与标题样式，核验所有 preserve-only 包部件。
"""
    (QA / "artifact.md").write_text(contract, encoding="utf-8")

    def paragraph(text, role="body", small=False, center=False, keep=False):
        source = prototypes.get(role, prototypes["body"])
        item = ET.Element(f"{{{W}}}p")
        props = copy.deepcopy(source.find("w:pPr", NS))
        if props is None: props = el("pPr")
        for old in list(props.findall("w:numPr", NS)) + list(props.findall("w:keepNext", NS)):
            props.remove(old)
        if center or role == "title":
            for old in props.findall("w:jc", NS): props.remove(old)
            props.append(el("jc", val="center"))
            for old in props.findall("w:ind", NS): props.remove(old)
            props.append(el("ind", firstLine=0))
        if keep or role in {"h1", "h2", "title"}: props.append(el("keepNext"))
        props.append(el("widowControl", val=1))
        if role == "title":
            for old in props.findall("w:pStyle", NS): props.remove(old)
            props.insert(0, el("pStyle", val="Title"))
        item.append(props)
        run = el("r")
        source_run = source.find("w:r/w:rPr", NS)
        rp = copy.deepcopy(source_run) if source_run is not None else el("rPr")
        if role == "title":
            rp.append(el("b")); rp.append(el("sz", val=28)); rp.append(el("color", val="000000"))
        if small:
            for old in rp.findall("w:sz", NS): rp.remove(old)
            rp.append(el("sz", val=20))
        run.append(rp)
        for i, line in enumerate(str(text).split("\n")):
            if i: run.append(el("br"))
            token = el("t"); token.text = line; token.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); run.append(token)
        item.append(run)
        return item

    def make_table(headers, rows, team=False):
        if team:
            result = copy.deepcopy(team_prototype)
            proto_rows = result.findall("w:tr", NS)
            for row in proto_rows: result.remove(row)
            for i, values in enumerate([headers] + rows):
                row = copy.deepcopy(proto_rows[0 if i == 0 else 1])
                for cell, text in zip(row.findall("w:tc", NS), values):
                    p = cell.find("w:p", NS)
                    existing_run = p.find("w:r/w:rPr", NS)
                    rp = copy.deepcopy(existing_run) if existing_run is not None else None
                    for old in list(p):
                        if old.tag != f"{{{W}}}pPr": p.remove(old)
                    run = el("r")
                    if rp is not None: run.append(rp)
                    for j,line in enumerate(str(text).replace(" / ", "\n").split("\n")):
                        if j: run.append(el("br"))
                        t=el("t");t.text=line;run.append(t)
                    p.append(run)
                result.append(row)
            return result
        count = len(headers)
        widths = [1600, 3600, 3096] if count == 3 else ([2000] + [int(6296/(count-1))]*(count-1))
        widths[-1] = 8296 - sum(widths[:-1])
        result = el("tbl"); props=copy.deepcopy(team_prototype.find("w:tblPr", NS)); result.append(props)
        props.append(el("tblLayout", type="fixed"))
        grid=el("tblGrid")
        for width in widths: grid.append(el("gridCol", w=width))
        result.append(grid)
        for i, values in enumerate([headers]+rows):
            row=el("tr"); trprops=el("trPr");trprops.append(el("cantSplit"))
            if i == 0: trprops.append(el("tblHeader"))
            row.append(trprops)
            for j, (width,text) in enumerate(zip(widths,values)):
                cell=el("tc");cp=el("tcPr");cp.append(el("tcW", w=width,type="dxa"));cp.append(el("vAlign",val="center"));cell.append(cp)
                p=paragraph(text, center=(count > 3 and j > 0))
                pp=p.find("w:pPr",NS)
                for tag in ["ind","spacing"]:
                    for old in pp.findall("w:"+tag,NS): pp.remove(old)
                pp.append(el("ind",firstLine=0));pp.append(el("spacing",before=60,after=60,line=288,lineRule="auto"))
                if i == 0:
                    p.find("w:r/w:rPr",NS).append(el("b"))
                cell.append(p);row.append(cell)
            result.append(row)
        return result

    for child in list(body): body.remove(child)
    relationships=ET.fromstring(parts["word/_rels/document.xml.rels"])
    content_types=ET.fromstring(parts["[Content_Types].xml"])
    if not any(x.get("Extension")=="png" for x in content_types):
        ET.SubElement(content_types,f"{{{CT}}}Default",Extension="png",ContentType="image/png")
    styles=ET.fromstring(parts["word/styles.xml"])
    if not styles.xpath("w:style[@w:styleId='Title']",namespaces=NS):
        title=el("style",type="paragraph",styleId="Title");title.append(el("name",val="Title"));title.append(el("basedOn",val="Normal"));title.append(el("qFormat"));styles.append(title)
    blocks=json.loads((ROOT / "reports/report_content.json").read_text(encoding="utf-8"))
    scratch=Document()
    is_reference=False
    for i,block in enumerate(blocks):
        kind=block["type"]
        if kind == "title": body.append(paragraph(block["text"],role="title"))
        elif kind == "heading":
            is_reference = is_reference or block["text"] == "参考资料"
            body.append(paragraph(block["text"],role="h1" if block["level"]==1 else "h2"))
        elif kind == "paragraph":
            item=paragraph(block["text"])
            if is_reference:
                props=item.find("w:pPr",NS)
                for tag in ["jc","ind","spacing"]:
                    for old in props.findall("w:"+tag,NS):props.remove(old)
                props.append(el("jc",val="left"));props.append(el("ind",firstLine=0));props.append(el("keepLines"))
                props.append(el("spacing",before=60,after=60,line=288,lineRule="auto"))
            body.append(item)
        elif kind == "table":
            body.append(make_table(block["headers"],block["rows"],team=(len(block["headers"])==2)))
            body.append(paragraph(""))
        elif kind == "figure":
            path=ROOT / "reports" / block["path"]
            image=scratch.add_paragraph(); image.add_run().add_picture(str(path),width=Inches(5.7))
            item=copy.deepcopy(image._p)
            item.insert(0,el("pPr")); pp=item.find("w:pPr",NS);pp.append(el("jc",val="center"));pp.append(el("keepNext"));pp.append(el("spacing",before=60,after=60))
            rid=f"rIdReport{i}";name=f"report_{i}.png"
            for blip in item.findall(".//a:blip",NS):blip.set(f"{{{R}}}embed",rid)
            for prop in item.findall(".//wp:docPr",NS):prop.set("id",str(i+100));prop.set("descr",block["caption"])
            ET.SubElement(relationships,f"{{{REL}}}Relationship",Id=rid,Type=R+"/image",Target="media/"+name)
            parts["word/media/"+name]=path.read_bytes()
            body.append(item);body.append(paragraph(block["caption"],small=True,center=True))
    body.append(section)
    parts["word/document.xml"]=xml_bytes(document)
    parts["word/_rels/document.xml.rels"]=xml_bytes(relationships)
    parts["[Content_Types].xml"]=xml_bytes(content_types)
    parts["word/styles.xml"]=xml_bytes(styles)
    with zipfile.ZipFile(OUTPUT,"w",zipfile.ZIP_DEFLATED) as z:
        for name,data in parts.items():z.writestr(name,data)
    allowed={"word/document.xml","word/_rels/document.xml.rels","[Content_Types].xml","word/styles.xml"}
    with zipfile.ZipFile(REFERENCE) as original,zipfile.ZipFile(OUTPUT) as final:
        preserved=[name for name in original.namelist() if name not in allowed]
        assert all(original.read(name)==final.read(name) for name in preserved)
    (QA / "template-fidelity.json").write_text(json.dumps({"reference_sha256":hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),"preserved_parts":preserved,"passed":True},ensure_ascii=False,indent=2),encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
