from __future__ import annotations

import html
import shutil
import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "皮肤病变图像分类研究_task5_updated.pptx"
FALLBACK = ROOT / "皮肤病变图像分类研究.pptx"
OUTPUT = ROOT / "任务5_错误样例_可编辑.pptx"

EMU = 914400
SLIDE_CX = 12192000
SLIDE_CY = 6858000


def emu(inches: float) -> int:
    return int(inches * EMU)


def get_image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data):
            while i < len(data) and data[i] == 0xFF:
                i += 1
            marker = data[i]
            i += 1
            if marker in (0xD8, 0xD9):
                continue
            length = struct.unpack(">H", data[i : i + 2])[0]
            if marker in range(0xC0, 0xCF) and marker not in (0xC4, 0xC8, 0xCC):
                height, width = struct.unpack(">HH", data[i + 3 : i + 7])
                return width, height
            i += length
    raise ValueError(path)


def fit_box(path: Path, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    iw, ih = get_image_size(path)
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    return x + (w - nw) // 2, y + (h - nh) // 2, nw, nh


def run(text: str, size: float, color: str = "334155", bold: bool = False) -> str:
    attrs = f' lang="zh-CN" sz="{int(size * 100)}"'
    if bold:
        attrs += ' b="1"'
    return (
        f"<a:r><a:rPr{attrs}>"
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        f'<a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/>'
        f"</a:rPr><a:t>{html.escape(text)}</a:t></a:r>"
    )


def paragraph(text: str, size: float = 12, color: str = "334155", bold: bool = False) -> str:
    return f"<a:p>{run(text, size=size, color=color, bold=bold)}</a:p>"


def textbox(
    sid: int,
    name: str,
    x: int,
    y: int,
    w: int,
    h: int,
    paras: list[str],
    fill: str | None = None,
    line: str | None = None,
    round_rect: bool = False,
    margin: int = 45720,
) -> str:
    fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
    line_xml = (
        f'<a:ln><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        if line
        else "<a:ln><a:noFill/></a:ln>"
    )
    geom = "roundRect" if round_rect else "rect"
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{sid}" name="{html.escape(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr>
  <p:txBody><a:bodyPr wrap="square" anchor="t" lIns="{margin}" rIns="{margin}" tIns="{margin}" bIns="{margin}"/><a:lstStyle/>{''.join(paras)}</p:txBody>
</p:sp>"""


def shape_rect(sid: int, name: str, x: int, y: int, w: int, h: int, fill: str, line: str = "CBD5E1") -> str:
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{sid}" name="{html.escape(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="roundRect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln></p:spPr>
</p:sp>"""


def picture(sid: int, name: str, rid: str, path: Path, x: int, y: int, w: int, h: int) -> str:
    fx, fy, fw, fh = fit_box(path, x, y, w, h)
    return f"""
<p:pic>
  <p:nvPicPr><p:cNvPr id="{sid}" name="{html.escape(name)}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
  <p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
  <p:spPr><a:xfrm><a:off x="{fx}" y="{fy}"/><a:ext cx="{fw}" cy="{fh}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:ln><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:ln></p:spPr>
</p:pic>"""


def add_case(
    base_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    image_id: str,
    true_label: str,
    trad_pred: str,
    deep_pred: str,
    confidence: str,
    localization: str,
    origin: Path,
    enhanced: Path,
    reason: str,
    rids: tuple[str, str],
) -> str:
    xi, yi, wi, hi = emu(x), emu(y), emu(w), emu(h)
    parts = [shape_rect(base_id, f"case {image_id}", xi, yi, wi, hi, "F8FAFC")]
    parts.append(textbox(base_id + 1, "case title", emu(x + 0.18), emu(y + 0.12), emu(w - 0.36), emu(0.28), [paragraph(title, 18, "003F88", True)]))
    parts.append(textbox(base_id + 2, "origin label", emu(x + 0.18), emu(y + 0.48), emu(1.1), emu(0.18), [paragraph("原图", 9, "334155")]))
    parts.append(textbox(base_id + 3, "enhanced label", emu(x + 2.2), emu(y + 0.48), emu(1.2), emu(0.18), [paragraph("预处理后", 9, "334155")]))
    parts.append(picture(base_id + 4, f"{image_id} origin", rids[0], origin, emu(x + 0.18), emu(y + 0.7), emu(1.78), emu(1.36)))
    parts.append(picture(base_id + 5, f"{image_id} enhanced", rids[1], enhanced, emu(x + 2.2), emu(y + 0.7), emu(1.78), emu(1.36)))
    facts = [
        f"image_id：{image_id}",
        f"真实标签：{true_label}",
        f"传统预测：{trad_pred}",
        f"深度预测：{deep_pred}",
        f"置信度：{confidence}",
        f"部位：{localization}",
    ]
    parts.append(textbox(base_id + 6, "facts", emu(x + 4.2), emu(y + 0.64), emu(w - 4.45), emu(1.42), [paragraph(line, 11, "1E293B") for line in facts]))
    parts.append(textbox(base_id + 7, "reason box", emu(x + 0.18), emu(y + 2.27), emu(w - 0.36), emu(h - 2.42), [paragraph("错误原因", 10.5, "9A3412", True), paragraph(reason, 9.5, "475569")], fill="FFF7ED", line="FDBA74", round_rect=True))
    return "".join(parts)


def add_compact_case(
    base_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    image_id: str,
    true_label: str,
    trad_pred: str,
    deep_pred: str,
    confidence: str,
    localization: str,
    origin: Path,
    enhanced: Path,
    reason: str,
    rids: tuple[str, str],
) -> str:
    xi, yi, wi, hi = emu(x), emu(y), emu(w), emu(h)
    parts = [shape_rect(base_id, f"case {image_id}", xi, yi, wi, hi, "F8FAFC")]
    parts.append(textbox(base_id + 1, "case title", emu(x + 0.12), emu(y + 0.08), emu(w - 0.24), emu(0.28), [paragraph(title, 14.5, "003F88", True)], margin=22860))
    parts.append(textbox(base_id + 2, "origin label", emu(x + 0.12), emu(y + 0.42), emu(0.75), emu(0.15), [paragraph("原图", 7.5, "334155")], margin=15240))
    parts.append(textbox(base_id + 3, "enhanced label", emu(x + 1.42), emu(y + 0.42), emu(0.95), emu(0.15), [paragraph("预处理", 7.5, "334155")], margin=15240))
    parts.append(picture(base_id + 4, f"{image_id} origin", rids[0], origin, emu(x + 0.12), emu(y + 0.60), emu(1.18), emu(0.96)))
    parts.append(picture(base_id + 5, f"{image_id} enhanced", rids[1], enhanced, emu(x + 1.42), emu(y + 0.60), emu(1.18), emu(0.96)))
    facts = [
        f"id：{image_id}",
        f"真实：{true_label}",
        f"传统：{trad_pred}",
        f"深度：{deep_pred}",
        f"置信度：{confidence}",
        f"部位：{localization}",
    ]
    parts.append(textbox(base_id + 6, "facts", emu(x + 2.78), emu(y + 0.44), emu(w - 2.92), emu(1.20), [paragraph(line, 8.5, "1E293B") for line in facts], margin=15240))
    parts.append(textbox(base_id + 7, "reason box", emu(x + 0.12), emu(y + 1.72), emu(w - 0.24), emu(h - 1.84), [paragraph("错误原因", 8.5, "9A3412", True), paragraph(reason, 7.8, "475569")], fill="FFF7ED", line="FDBA74", round_rect=True, margin=30480))
    return "".join(parts)


def build_slide() -> tuple[str, list[tuple[Path, str]]]:
    cases = [
        (
            "传统错、深度对：mel -> nv",
            "ISIC_0032046",
            "mel",
            "nv",
            "mel",
            "0.9373",
            "trunk",
            ROOT / "preprocessing" / "mel" / "origin" / "ISIC_0032046.jpg",
            ROOT / "preprocessing" / "mel" / "enhanced" / "ISIC_0032046.jpg",
            "传统手工特征把 mel 判为 nv，说明颜色/纹理相似时容易推向多数类；深度模型能抓到更细粒度病灶线索。",
        ),
        (
            "高置信错误：bkl -> akiec",
            "ISIC_0028611",
            "bkl",
            "nv",
            "akiec",
            "0.9996",
            "lower extremity",
            ROOT / "preprocessing" / "bkl" / "origin" / "ISIC_0028611.jpg",
            ROOT / "preprocessing" / "bkl" / "enhanced" / "ISIC_0028611.jpg",
            "深度模型以接近 1 的置信度判为 akiec，说明 softmax 高置信不等于医学可靠，仍需要错误样本复核。",
        ),
        (
            "传统对、深度错：mel -> nv",
            "ISIC_0034205",
            "mel",
            "mel",
            "nv",
            "0.9959",
            "upper extremity",
            ROOT / "preprocessing" / "mel" / "origin" / "ISIC_0034205.jpg",
            ROOT / "preprocessing" / "mel" / "enhanced" / "ISIC_0034205.jpg",
            "传统模型判对但深度模型高置信判为 nv，属于深度模型回退；提示深度模型仍可能漏掉少数 mel 样本。",
        ),
        (
            "两个模型都错：mel -> nv",
            "ISIC_0030107",
            "mel",
            "nv",
            "nv",
            "0.9997",
            "scalp",
            ROOT / "preprocessing" / "mel" / "origin" / "ISIC_0030107.jpg",
            ROOT / "preprocessing" / "mel" / "enhanced" / "ISIC_0030107.jpg",
            "传统和深度都把 mel 判为 nv，且深度置信度接近 1；这类共同困难样本应优先人工复核和模型校准。",
        ),
    ]
    media = []
    for idx, case in enumerate(cases, start=1):
        media.append((case[7], f"task5_editable_case{idx}_origin.jpg"))
        media.append((case[8], f"task5_editable_case{idx}_enhanced.jpg"))

    parts = []
    positions = [(0.25, 0.18), (6.85, 0.18), (0.25, 3.85), (6.85, 3.85)]
    for idx, (case, pos) in enumerate(zip(cases, positions), start=0):
        base_id = 10 + idx * 20
        rid_pair = (f"rId{3 + idx * 2}", f"rId{4 + idx * 2}")
        parts.append(add_compact_case(base_id, pos[0], pos[1], 6.20, 3.35, *case[:7], case[7], case[8], case[9], rid_pair))
    body = "".join(parts)
    slide = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name="Group 1"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {body}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''
    return slide, media


def content_types(xml: bytes, media_names: list[str]) -> bytes:
    root = ET.fromstring(xml)
    ns = "http://schemas.openxmlformats.org/package/2006/content-types"
    ET.register_namespace("", ns)
    defaults = {child.attrib.get("Extension") for child in root if child.tag.endswith("Default")}
    for ext, typ in {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.items():
        if ext not in defaults and any(name.lower().endswith("." + ext) for name in media_names):
            el = ET.SubElement(root, f"{{{ns}}}Default")
            el.set("Extension", ext)
            el.set("ContentType", typ)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    source = SOURCE if SOURCE.exists() else FALLBACK
    slide, media = build_slide()
    media_names = [target for _, target in media]
    for src, _ in media:
        if not src.exists():
            raise FileNotFoundError(src)

    tmp = OUTPUT.with_suffix(".tmp.pptx")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(source, "r") as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        keep = {
            "[Content_Types].xml",
            "_rels/.rels",
            "docProps/app.xml",
            "docProps/core.xml",
            "docProps/thumbnail.jpeg",
            "ppt/presentation.xml",
            "ppt/_rels/presentation.xml.rels",
            "ppt/presProps.xml",
            "ppt/viewProps.xml",
            "ppt/tableStyles.xml",
            "ppt/theme/theme1.xml",
            "ppt/slideMasters/slideMaster1.xml",
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            "ppt/slideLayouts/slideLayout1.xml",
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
        }
        for item in zin.infolist():
            if item.filename not in keep:
                continue
            data = zin.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = content_types(data, media_names)
            elif item.filename == "ppt/presentation.xml":
                data = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
  <p:sldSz cx="{SLIDE_CX}" cy="{SLIDE_CY}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>'''.encode("utf-8")
            elif item.filename == "ppt/_rels/presentation.xml.rels":
                data = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
</Relationships>'''
            zout.writestr(item, data)
        zout.writestr("ppt/slides/slide1.xml", slide.encode("utf-8"))
        rels = [
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>',
        ]
        for i, (_, target) in enumerate(media, start=3):
            rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{target}"/>')
        zout.writestr(
            "ppt/slides/_rels/slide1.xml.rels",
            ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(rels) + "</Relationships>").encode("utf-8"),
        )
        for src, target in media:
            zout.writestr(f"ppt/media/{target}", src.read_bytes())
    shutil.move(str(tmp), str(OUTPUT))
    print(f"created: {OUTPUT}")


if __name__ == "__main__":
    main()
