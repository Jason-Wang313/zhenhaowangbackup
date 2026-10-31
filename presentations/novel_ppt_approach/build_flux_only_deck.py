"""Build the explicitly revised, FLUX-only research-plan deck (no GPU work).

Uses only Python's standard library and the existing deck's Office theme/layout.
All old slides, pictures and notes are replaced. The original remains in Git at
610d23ac9df682376743fe96b44fd753992a42c4; this is not an unchanged original.
Run with the project Python, not by changing the training environment.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import posixpath
import zipfile
from html import escape
import xml.etree.ElementTree as ET


FONT = "Microsoft YaHei"
NAVY = "172A45"
BLUE = "185ABD"
TEAL = "087E8B"
GRAY = "526174"
LIGHT = "F1F5FA"
RED = "A32929"
WHITE = "FFFFFF"
OUTPUT = Path(__file__).with_name("Novel_PPT_Approach_VLM_Restoration_Energy.pptx")


P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'


def Inches(value):
    return round(value * 914400)


def relationships(items):
    return XML + f'<Relationships xmlns="{PKG}">' + ''.join(
        f'<Relationship Id="{rid}" Type="{R}/{kind}" Target="{target}"/>'
        for rid, kind, target in items) + '</Relationships>'


class Presentation:
    def __init__(self):
        self.slides = []
        self.core_properties = SimpleNamespace()
        with zipfile.ZipFile(OUTPUT) as source:
            self.skeleton = {name: source.read(name) for name in source.namelist()
                             if name.startswith(('ppt/slideMasters/', 'ppt/slideLayouts/', 'ppt/theme/'))}

    def save(self, output):
        files = dict(self.skeleton)
        files['_rels/.rels'] = relationships([('rId1', 'officeDocument', 'ppt/presentation.xml')])
        ids = ''.join(f'<p:sldId id="{256+i}" r:id="rId{i+2}"/>' for i in range(len(self.slides)))
        files['ppt/presentation.xml'] = XML + (
            f'<p:presentation xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}">'
            '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
            f'<p:sldIdLst>{ids}</p:sldIdLst><p:sldSz cx="{self.slide_width}" cy="{self.slide_height}"/>'
            '<p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
        files['ppt/_rels/presentation.xml.rels'] = relationships(
            [('rId1', 'slideMaster', 'slideMasters/slideMaster1.xml')] +
            [(f'rId{i+2}', 'slide', f'slides/slide{i+1}.xml') for i in range(len(self.slides))])
        for i, shapes in enumerate(self.slides, 1):
            group = ('<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
                     '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
                     '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>')
            body = group + ''.join(shape_xml(shape, j) for j, shape in enumerate(shapes, 2))
            files[f'ppt/slides/slide{i}.xml'] = XML + (
                f'<p:sld xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld>'
                '<p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
                '<a:effectLst/></p:bgPr></p:bg>'
                f'<p:spTree>{body}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/>'
                '</p:clrMapOvr></p:sld>')
            files[f'ppt/slides/_rels/slide{i}.xml.rels'] = relationships(
                [('rId1', 'slideLayout', '../slideLayouts/slideLayout7.xml')])
        types = []
        for name in sorted(files):
            if not name.endswith('.xml'):
                continue
            if name == 'ppt/presentation.xml': kind = 'presentationml.presentation.main'
            elif '/slideMasters/' in name: kind = 'presentationml.slideMaster'
            elif '/slideLayouts/' in name: kind = 'presentationml.slideLayout'
            elif '/theme/' in name: kind = 'theme'
            else: kind = 'presentationml.slide'
            types.append(f'<Override PartName="/{name}" ContentType="application/vnd.openxmlformats-officedocument.{kind}+xml"/>')
        files['[Content_Types].xml'] = XML + (
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>' + ''.join(types) + '</Types>')
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(files.items()):
                info = zipfile.ZipInfo(name, (2026, 10, 31, 12, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)


def shape_xml(shape, index):
    x, y, w, h = (Inches(shape[k]) for k in ('x', 'y', 'w', 'h'))
    fill = (f'<a:solidFill><a:srgbClr val="{shape["fill"]}"/></a:solidFill>'
            if shape.get('fill') else '<a:noFill/>')
    geom = 'roundRect' if shape.get('rounded') else 'rect'
    result = (f'<p:sp><p:nvSpPr><p:cNvPr id="{index}" name="Shape {index}"/>'
              '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>'
              f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
              f'<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>{fill}'
              '<a:ln><a:noFill/></a:ln></p:spPr>')
    if 'text' in shape:
        paragraphs = []
        for text in shape['text'].split('\n'):
            paragraphs.append(
                f'<a:p><a:pPr algn="{"ctr" if shape["center"] else "l"}">'
                '<a:spcAft><a:spcPts val="800"/></a:spcAft></a:pPr><a:r>'
                f'<a:rPr lang="zh-CN" sz="{shape["size"]*100}" b="{int(shape["bold"])}">'
                f'<a:solidFill><a:srgbClr val="{shape["ink"]}"/></a:solidFill>'
                f'<a:latin typeface="{FONT}"/><a:ea typeface="{FONT}"/></a:rPr>'
                f'<a:t>{escape(text)}</a:t></a:r></a:p>')
        result += ('<p:txBody><a:bodyPr wrap="square" anchor="ctr" lIns="36576" '
                   'rIns="36576" tIns="27432" bIns="27432"/><a:lstStyle/>' +
                   ''.join(paragraphs) + '</p:txBody>')
    return result + '</p:sp>'


def textbox(slide, x, y, w, h, text, size=22, ink=NAVY, bold=False, center=False):
    box = dict(x=x, y=y, w=w, h=h, text=text, size=size, ink=ink, bold=bold, center=center)
    slide.append(box)
    return box


def panel(slide, x, y, w, h, title, body, accent=BLUE, size=21):
    slide.append(dict(x=x, y=y, w=w, h=h, fill=LIGHT, rounded=True))
    textbox(slide, x+.2, y+.13, w-.4, .55, title, 23, accent, True)
    textbox(slide, x+.2, y+.8, w-.4, h-.95, body, size)


def slide_base(prs, title, subtitle):
    slide = []
    prs.slides.append(slide)
    textbox(slide, .55, .30, 12.2, .65, title, 29, NAVY, True)
    textbox(slide, .57, 1.03, 12.1, .55, subtitle, 16, GRAY)
    slide.append(dict(x=.6, y=1.66, w=12.1, h=.035, fill=BLUE))
    textbox(slide, .6, 7.05, 11.4, .25,
            "2026-10-31 修订草案｜仅 FLUX 可训练主干｜代码已实现，CPU通过；GPU尚未验收", 10, GRAY)
    textbox(slide, 12.0, 7.02, .7, .3, str(len(prs.slides)), 12, GRAY, center=True)
    return slide


def note(slide, text):
    # The same details live in the tracked research-plan Markdown; no old notes
    # are copied into the newly generated presentation.
    pass


def build(output):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333333), Inches(7.5)
    props = prs.core_properties
    props.title = "FLUX-only：VLM条件恢复 energy 重做方案"
    props.subject = "用户明确禁用外挂CNN与Restormer；新方案尚未实现"
    props.author = "Project research plan"
    props.created = props.modified = datetime(2026, 10, 31, 12, 0, 0)
    props.revision = 2

    s = slide_base(prs, "只用 FLUX：恢复、判断、继续修正", "用户最新决定：不是把旧 CNN 换个名字；这是重新设计的实验。")
    panel(s, .6, 1.95, 7.6, 3.55, "唯一可训练图像主干：预训练 FLUX.2 Klein 4B",
          "用 FLUX 内部图像特征产生 energy\n用 FLUX 内部 LoRA 学习判断和修正\n直接训练修正后的真实图像质量", size=23)
    panel(s, 8.45, 1.95, 4.25, 3.55, "明确禁止", "外挂 CNN energy / scorer\nSpatialConditionalEnergy\nTailQualityEnergy\nRestormer、像素 refiner及其蒸馏", RED, 18)
    textbox(s, .72, 5.75, 11.9, .87, "边界：原生冻结 VAE 与固定 LPIPS 保留；不是额外恢复器。\n本版不是原导师确认版本；旧 PPT 在 Git 历史完整保留。", 18, GRAY)
    note(s, "详细执行边界见 reports/foundir_flux_only_energy_redesign_2026-10-31.md。原生VAE与LPIPS内部含卷积，不宣称整个图零卷积。没有新增GPU任务。")

    s = slide_base(prs, "第一原则：要学会有用的修正，而不只是打分", "旧结果支持继续研究，但不能推出‘FLUX 更大，所以一定容易成功’。")
    panel(s, .6, 1.95, 5.9, 3.65, "已经有的证据", "旧 CNN：8 个训练来源检查\nPSNR：+1.06034 dB，已过单项门槛\nLPIPS：只降 2.49%，目标 9.93%\n3 / 8 张图 LPIPS 变差", size=22)
    panel(s, 6.75, 1.95, 5.95, 3.65, "有依据的两项改变", "表示：复用 FLUX 预训练图像特征\n目标：直接监督实际修后的图像\n保留已验证的 FLUX 尾段修正位置\n不重训 4B，不再从零训练小 CNN", TEAL, 22)
    textbox(s, .75, 5.86, 11.8, .8, "还没解决：LPIPS 的有效方向、未见来源泛化、二阶反传成本。\n旧 GT 多轮成功，不等于当前单步 energy 已经成功。", 20, GRAY)

    s = slide_base(prs, "架构：同一个 FLUX 底座，两种前向模式", "生成能力不被悄悄换掉；新增学习发生在 FLUX Transformer 内部。")
    panel(s, .6, 1.95, 5.9, 3.65, "生成模式 G：固定的比较基线", "预训练 FLUX + 原 Noise LoRA\n权重全部冻结，保持原50步流程\n新增 energy LoRA 关闭\n生成初始图与最终尾段恢复图", size=22)
    panel(s, 6.75, 1.95, 5.95, 3.65, "energy 模式 E：本次训练对象", "共享 FLUX 底座 + energy LoRA\n图像 tokens → 线性读出 → energy\nVLM等级 → 线性投影 → 条件token\n没有第二个 CNN 或独立恢复主干", TEAL, 22)
    textbox(s, .75, 5.88, 11.8, .72, "只训练：energy LoRA + 条件投影 + 线性读出。\n原生成 LoRA 不更新；不能在反传图中原地切换或覆盖权重。", 20, GRAY)
    note(s, "初始建议rank16，沿现有50组attention/linear注入点。现有LoRALinear只有一套adapter，双模式尚需实现。不是冻结整个FLUX后仅训练裸线性head。")

    s = slide_base(prs, "Energy 具体看什么？", "输入仍是这张图的信息；不在推理时偷看 GT，也不把数据集类别当 VLM 判断。")
    panel(s, .6, 1.95, 5.9, 3.8, "送入 FLUX energy 模式", "当前候选 latent + 真实时间 t\n原 degraded image reference\n固定初步恢复图 reference\n逐图 VLM 五个退化等级", size=23)
    panel(s, 6.75, 1.95, 5.95, 3.8, "从 FLUX 内部读出判断", "Transformer 处理候选与条件\n读取 final_layer 前的图像 tokens\n每个 token 线性读出局部分数\n最终汇总为同一个标量 energy", TEAL, 22)
    textbox(s, .75, 5.99, 11.8, .58, "五等级投影 ≠ 已接入原始 VLM hidden tokens；小集合条件变化不足，不能先宣称 VLM 有效。", 18, GRAY)

    s = slide_base(prs, "Energy 梯度直接引导 FLUX 尾段", "从同一个第48步状态开始，真正比较 0 / 1 / 2 / 3 次修正。")
    stages = [("状态 z", "保存的第48步 latent"), ("FLUX energy", "判断 + 对 latent 求梯度"),
              ("修正状态", "沿 energy 下降方向更新"), ("FLUX 尾两步", "生成实际恢复图")]
    for i, (title, body) in enumerate(stages):
        x = .6 + 3.08*i
        panel(s, x, 2.2, 2.75, 1.7, title, body, size=16)
        if i < 3:
            textbox(s, x+2.77, 2.72, .28, .5, "→", 23, BLUE, center=True)
    textbox(s, .85, 4.25, 11.6, .8, "z_next = z − η × normalized_gradient(E)\n每次重新判断，再走同一冻结尾段；不是在 RGB 上加一个修复网络。", 23, center=True)
    textbox(s, .78, 5.5, 11.8, 1.12, "步长与幅度规则在训练侧冻结；不看检查图 GT 选幅度。\n不沿用 25 候选搜索；真实报告前向、反传、解码及总耗时。\n最终同时报告固定3步结果与energy自选结果，分别检验修正和自验证。", 19, GRAY)

    s = slide_base(prs, "直接训练‘修完以后更好’，不是只训练分数", "完整展开三步；训练梯度必须回到 FLUX 的 energy LoRA。")
    panel(s, .6, 1.95, 7.6, 3.8, "主目标：动作排序 + 每一步实际图像损失", "MSE：减少像素误差，对齐 PSNR\nLPIPS：保留细节与感知质量\n变差惩罚：分别惩罚两个指标退步\n真实修后损失 → energy 梯度 → energy LoRA", size=23)
    panel(s, 8.45, 1.95, 4.25, 3.8, "不能再省略", "不能只拟合评分缓存\n不能 detach 修正方向\n不能用1步代替3步验收\n软惩罚不保证推理无伤害", RED, 20)
    textbox(s, .75, 5.98, 11.8, .68, "训练用连续图像损失；评测严格保留双 uint8。\n原 PPT 的 L1 在本版明确改为 MSE 以对齐 PSNR，不再暗换目标。", 18, GRAY)

    s = slide_base(prs, "恢复动作对比：全部来自 FLUX 或明确构造", "不再把同一残差上的几个幅度，说成完整动作对比。")
    panel(s, .6, 1.95, 5.9, 3.75, "同一输入，比较不同恢复行为", "正确 / 更优的 FLUX 恢复候选\n真实错误 FLUX 专家输出\n未恢复、欠恢复、过恢复\n重复 FLUX 恢复、细节损坏反例", size=22)
    panel(s, 6.75, 1.95, 5.95, 3.75, "终态与中间状态不能混淆", "最终图：VAE 编码，以 t=0 排序\n尾段状态：真实 t48 + 尾段图像标签\n两指标都明确更好的候选优先排序\nGT 仅用于训练与离线评价", TEAL, 21)
    textbox(s, .75, 5.97, 11.8, .64, "禁止 Restormer/CNN 候选、蒸馏或旧 CNN 权重初始化。\n不把最终图的 VAE latent 冒充第48步生成状态。", 19, RED)

    s = slide_base(prs, "重做顺序：可训练 → 能拟合 → 能泛化", "小集合不是正式数据规模，也不是用来证明论文成功。")
    rows = [
        ("① 实现 / 工程资格", "核对模型白名单、全3步二阶梯度、512裁图、显存与恢复；建议25步。"),
        ("② 独立200步拟合", "原24个训练来源；25/50/100/200检查0/1/2/3次修正，不复用资格权重。"),
        ("③ 扩合规数据训练", "有稳定双指标信号后，扩Noise训练池；冻结来源隔离的开发协议。"),
        ("④ 未见来源 / 扩类型", "Noise泛化通过后再到Haze及其他退化，不同时恢复MoE等旧分支。"),
    ]
    for i, (title, body) in enumerate(rows):
        y = 1.95 + i*1.08
        textbox(s, .75, y, 3.2, .72, title, 22, BLUE, True)
        textbox(s, 4.0, y, 8.4, .78, body, 20)
    textbox(s, .75, 6.38, 11.8, .35, "每个完整step保存模型、优化器、随机状态和采样位置；真实resume，不仅加载权重。", 16, GRAY)

    s = slide_base(prs, "标准不变：收益必须属于 energy", "所有比较使用同环境、同来源、同裁图、同量化与原 FLUX 部署基线。")
    panel(s, .6, 1.95, 5.9, 3.85, "最终联合门槛", "PSNR：相对原 FLUX 至少 +0.30 dB\nLPIPS：相对下降至少 9.9281%\n最终必须在未参与训练/选参的来源成立\n报告单图受损率，不只报告平均数", size=22)
    panel(s, 6.75, 1.95, 5.95, 3.85, "必须能区分的贡献", "原 FLUX vs. energy 1/2/3步\n固定3步 vs. energy自选\n同计算预算、无energy的额外FLUX采样\n主方法有效后：去VLM / 排序 / 修后损失", TEAL, 20)
    textbox(s, .75, 6.0, 11.8, .62, "原基线含 70% 输入 + 30% FLUX 的固定混合，不是 Restormer；不能换弱基线。\n禁止 official / sealed / internal / parent-test 像素训练、调参或选 checkpoint。", 17, GRAY)

    s = slide_base(prs, "什么已确定，什么还没有？", "本版是明确的重做计划，不是对成功率、成本或创新性的保证。")
    panel(s, .6, 1.95, 5.9, 3.8, "已确定", "只用预训练 FLUX 可训练主干\n不再用外挂 CNN 或 Restormer\n保留原恢复能力、基线和评价口径\n直接训练修后质量，不重复旧诊断", size=22)
    panel(s, 6.75, 1.95, 5.95, 3.8, "仍需实测", "完整3步二阶反传能否装下显存\nFLUX energy 能否学会 LPIPS 方向\n更大数据与新来源能否保留收益\n速度 / GPU-hours / VLM实际贡献", TEAL, 22)
    textbox(s, .75, 5.98, 11.8, .72, "状态：FLUX-only 25步入口已实现，9项CPU测试通过，未启动 GPU。\n单机4/8张H100/H200；512图、三次更新、全局batch8。显存/速度仍需实测。", 19, RED)

    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))
    assert len(prs.slides) == 10
    texts = []
    for slide in prs.slides:
        for shape in slide:
            assert shape['x'] >= 0 and shape['y'] >= 0
            assert Inches(shape['x'] + shape['w']) <= prs.slide_width + 2
            assert Inches(shape['y'] + shape['h']) <= prs.slide_height + 2
            if 'text' in shape:
                texts.append(shape['text'])
    all_text = "\n".join(texts)
    for phrase in ("明确禁止", "Restormer", "FLUX 内部 LoRA", "1 / 2 / 3", "未启动 GPU", "9.9281%"):
        assert phrase in all_text, phrase
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert not any(name.startswith("ppt/media/") for name in archive.namelist())
        names = set(archive.namelist())
        for name in names:
            if name.endswith(('.xml', '.rels')):
                root = ET.fromstring(archive.read(name))
                if name.endswith('.rels'):
                    base = posixpath.dirname(posixpath.dirname(name))
                    for rel in root:
                        if rel.get('TargetMode') != 'External':
                            target = posixpath.normpath(posixpath.join(base, rel.attrib['Target']))
                            assert target in names, (name, target)
        slide_names = [f'ppt/slides/slide{i}.xml' for i in range(1, 11)]
        serialized_text = '\n'.join(t.text or '' for name in slide_names
                                    for t in ET.fromstring(archive.read(name)).iter(f'{{{A}}}t'))
        assert serialized_text == all_text
    return {"path": str(output), "slides": len(prs.slides), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "validation": "XML_relationships_text_bounds_and_CRC_only_not_rendered"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
