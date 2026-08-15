const path = require('path');
// Use global node_modules path since local install is not present
const globalModules = 'C:\\Users\\zhai_\\AppData\\Roaming\\npm\\node_modules';
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  VerticalAlign, LevelFormat
} = require(path.join(globalModules, 'docx'));
const fs = require('fs');

// ── helpers ──────────────────────────────────────────────────────────
const border = { style: BorderStyle.SINGLE, size: 4, color: "AAAAAA" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 140, right: 140 };

function hCell(text, widthDXA, opts = {}) {
  return new TableCell({
    borders,
    width: { size: widthDXA, type: WidthType.DXA },
    shading: { fill: "1F4E79", type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 18, font: "Arial" })]
    })],
    ...opts
  });
}

function dCell(text, widthDXA, opts = {}) {
  return new TableCell({
    borders,
    width: { size: widthDXA, type: WidthType.DXA },
    margins: cellMargins,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text: String(text), size: 18, font: "Arial" })]
    })],
    ...opts
  });
}

function dCellBold(text, widthDXA, fill = "E8F0FE") {
  return new TableCell({
    borders,
    width: { size: widthDXA, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: String(text), bold: true, size: 18, font: "Arial" })]
    })]
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, bold: true, size: 36, font: "Arial", color: "1F4E79" })],
    spacing: { before: 360, after: 180 }
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true, size: 28, font: "Arial", color: "2E74B5" })],
    spacing: { before: 240, after: 120 }
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    children: [new TextRun({ text, size: 22, font: "Arial", ...opts })]
  });
}

function codePara(text) {
  return new Paragraph({
    spacing: { before: 40, after: 40 },
    indent: { left: 600 },
    children: [new TextRun({ text, font: "Courier New", size: 18, color: "444444" })]
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, size: 22, font: "Arial" })]
  });
}

function spacer() {
  return new Paragraph({ children: [new TextRun("")], spacing: { before: 60, after: 60 } });
}

// ── table dimensions (A4, 2cm margins → content = ~11626 DXA) ──────
// distribution table columns
const W_TOTAL = 11626;
const W = [2400, 1200, 600, 600, 600, 700, 2526]; // 7 cols, sum = 8626? let me recalc
// recalc: 2400+1200+600+600+600+700+2526 = 8626 (too narrow, use proportional)
// Use proportions out of 11626
const WW = [2800, 1400, 700, 700, 700, 800, 4526]; // sum = 11626

// CSV detail table columns
const CW = [900, 900, 1600, 3000, 2100, 700]; // sum = 9200 → pad last col
const CW_TOTAL = 11626;
const CW_LAST = CW_TOTAL - CW[0] - CW[1] - CW[2] - CW[3] - CW[4]; // 11626-8300=3326 → use 1126 for label
// Actually: 900+900+1600+3000+2100+1126 = 9626 → still off, use fixed
const CWFIX = [900, 900, 1800, 3500, 3000, 1526]; // sum = 11626

// ── distribution data ─────────────────────────────────────────────
const distRows = [
  ["食物蛋白过敏",   "婴儿",    2,  3,  11, 16,  "氨基酸配方"],
  ["乳蛋白过敏高风险","婴儿",  11,  0,   0, 11,  "乳蛋白部分水解配方"],
  ["乳糖不耐受_婴儿","婴儿",  13,  1,   0, 14,  "无乳糖配方"],
  ["早产",           "婴儿",  15,  5,   0, 20,  "早产/低出生体重婴儿配方"],
  ["苯丙酮尿症_婴儿","婴儿",   1,  0,  16, 17,  "氨基酸代谢障碍配方"],
  ["补充蛋白质",     "1岁以上",13, 51,   0, 64,  "蛋白质（氨基酸）组件"],
  ["消化吸收障碍",   "1岁以上",56, 51,   0,107,  "全营养配方食品"],
  ["肿瘤",           "1岁以上", 1, 56,   0, 57,  "特定全营养配方食品"],
  ["腹泻脱水",       "1岁以上", 2, 51,   0, 53,  "电解质配方"],
  ["吞咽障碍",       "1岁以上", 1, 56,   0, 57,  "增稠组件"],
  ["苯丙酮尿症",     "1岁以上",51,  0,   0, 51,  "非全营养配方食品"],
];

function makeDistTable() {
  const headerRow = new TableRow({
    tableHeader: true,
    children: [
      hCell("病症",    WW[0]),
      hCell("年龄段",  WW[1]),
      hCell("S",       WW[2]),
      hCell("A",       WW[3]),
      hCell("ban",     WW[4]),
      hCell("小计",    WW[5]),
      hCell("首选类别",WW[6]),
    ]
  });

  const dataRows = distRows.map((r, i) => {
    const fill = i % 2 === 0 ? "FFFFFF" : "F2F7FC";
    return new TableRow({ children: [
      new TableCell({ borders, width: { size: WW[0], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill, type: ShadingType.CLEAR },
        children: [new Paragraph({ children: [new TextRun({ text: r[0], size: 18, font: "Arial" })] })] }),
      new TableCell({ borders, width: { size: WW[1], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill, type: ShadingType.CLEAR },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: r[1], size: 18, font: "Arial" })] })] }),
      new TableCell({ borders, width: { size: WW[2], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill: r[2] > 0 ? "E2EFDA" : fill, type: ShadingType.CLEAR },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: String(r[2]), size: 18, font: "Arial", bold: r[2] > 0 })] })] }),
      new TableCell({ borders, width: { size: WW[3], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill: r[3] > 0 ? "FFF2CC" : fill, type: ShadingType.CLEAR },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: String(r[3]), size: 18, font: "Arial" })] })] }),
      new TableCell({ borders, width: { size: WW[4], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill: r[4] > 0 ? "FCE4D6" : fill, type: ShadingType.CLEAR },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: String(r[4]), size: 18, font: "Arial" })] })] }),
      new TableCell({ borders, width: { size: WW[5], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill, type: ShadingType.CLEAR },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: String(r[5]), size: 18, font: "Arial", bold: true })] })] }),
      new TableCell({ borders, width: { size: WW[6], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill, type: ShadingType.CLEAR },
        children: [new Paragraph({ children: [new TextRun({ text: r[6], size: 18, font: "Arial" })] })] }),
    ]});
  });

  // total row
  const totalRow = new TableRow({ children: [
    new TableCell({ borders, width: { size: WW[0], type: WidthType.DXA }, margins: cellMargins,
      shading: { fill: "D6E4F7", type: ShadingType.CLEAR },
      children: [new Paragraph({ children: [new TextRun({ text: "合计", bold: true, size: 18, font: "Arial" })] })] }),
    new TableCell({ borders, width: { size: WW[1], type: WidthType.DXA }, margins: cellMargins,
      shading: { fill: "D6E4F7", type: ShadingType.CLEAR },
      children: [new Paragraph({ children: [new TextRun({ text: "", size: 18, font: "Arial" })] })] }),
    new TableCell({ borders, width: { size: WW[2], type: WidthType.DXA }, margins: cellMargins,
      shading: { fill: "D6E4F7", type: ShadingType.CLEAR },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "166", bold: true, size: 18, font: "Arial" })] })] }),
    new TableCell({ borders, width: { size: WW[3], type: WidthType.DXA }, margins: cellMargins,
      shading: { fill: "D6E4F7", type: ShadingType.CLEAR },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "274", bold: true, size: 18, font: "Arial" })] })] }),
    new TableCell({ borders, width: { size: WW[4], type: WidthType.DXA }, margins: cellMargins,
      shading: { fill: "D6E4F7", type: ShadingType.CLEAR },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "27", bold: true, size: 18, font: "Arial" })] })] }),
    new TableCell({ borders, width: { size: WW[5], type: WidthType.DXA }, margins: cellMargins,
      shading: { fill: "D6E4F7", type: ShadingType.CLEAR },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "467", bold: true, size: 18, font: "Arial" })] })] }),
    new TableCell({ borders, width: { size: WW[6], type: WidthType.DXA }, margins: cellMargins,
      shading: { fill: "D6E4F7", type: ShadingType.CLEAR },
      children: [new Paragraph({ children: [new TextRun({ text: "", size: 18, font: "Arial" })] })] }),
  ]});

  return new Table({
    width: { size: W_TOTAL, type: WidthType.DXA },
    columnWidths: WW,
    rows: [headerRow, ...dataRows, totalRow]
  });
}

// ── column structure table ────────────────────────────────────────
function makeColTable() {
  const CW2 = [2200, 3000, 6426]; // col, type, desc
  const headerRow = new TableRow({ tableHeader: true, children: [
    hCell("字段名",   CW2[0]),
    hCell("类型",     CW2[1]),
    hCell("说明",     CW2[2]),
  ]});
  const rows = [
    ["condition",  "字符串", "病症/需求场景，如「食物蛋白过敏」、「肿瘤」"],
    ["age_group",  "字符串", "适用年龄段：「婴儿」或「1岁以上」"],
    ["注册证号",   "字符串", "国食注字 TY……，产品唯一标识"],
    ["产品名称",   "字符串", "中文全称，含「特殊医学用途」字样"],
    ["产品类别",   "字符串", "如「氨基酸配方」、「全营养配方食品」等，与 CLINICAL_KB 对齐"],
    ["label",      "枚举",   "S = 首选 / A = 可选 / ban = 禁用"],
  ];
  const dataRows = rows.map((r, i) => {
    const fill = i % 2 === 0 ? "FFFFFF" : "F5F5F5";
    return new TableRow({ children: [
      new TableCell({ borders, width: { size: CW2[0], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill, type: ShadingType.CLEAR },
        children: [new Paragraph({ children: [new TextRun({ text: r[0], font: "Courier New", size: 18, bold: true })] })] }),
      new TableCell({ borders, width: { size: CW2[1], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill, type: ShadingType.CLEAR },
        children: [new Paragraph({ children: [new TextRun({ text: r[1], font: "Arial", size: 18 })] })] }),
      new TableCell({ borders, width: { size: CW2[2], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill, type: ShadingType.CLEAR },
        children: [new Paragraph({ children: [new TextRun({ text: r[2], font: "Arial", size: 18 })] })] }),
    ]});
  });
  return new Table({ width: { size: W_TOTAL, type: WidthType.DXA }, columnWidths: CW2, rows: [headerRow, ...dataRows] });
}

// ── label table ───────────────────────────────────────────────────
function makeLabelTable() {
  const LW = [1400, 2600, 7626];
  const headerRow = new TableRow({ tableHeader: true, children: [
    hCell("Label值", LW[0]), hCell("含义", LW[1]), hCell("临床解释", LW[2])
  ]});
  const rows = [
    ["S",   "首选（Strongly recommended）", "产品类别与该病症的 preferred 完全匹配，应优先推荐"],
    ["A",   "可选（Alternative）",           "产品类别属于 alternative，可作为替代或辅助选择"],
    ["ban", "禁用（Contraindicated）",       "产品类别在 contraindicated 列表中，推荐系统必须过滤"],
  ];
  const fills = ["E2EFDA", "FFF2CC", "FCE4D6"];
  const dataRows = rows.map((r, i) => new TableRow({ children: [
    new TableCell({ borders, width: { size: LW[0], type: WidthType.DXA }, margins: cellMargins,
      shading: { fill: fills[i], type: ShadingType.CLEAR },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: r[0], bold: true, size: 20, font: "Courier New" })] })] }),
    new TableCell({ borders, width: { size: LW[1], type: WidthType.DXA }, margins: cellMargins,
      shading: { fill: fills[i], type: ShadingType.CLEAR },
      children: [new Paragraph({ children: [new TextRun({ text: r[1], size: 18, font: "Arial", bold: true })] })] }),
    new TableCell({ borders, width: { size: LW[2], type: WidthType.DXA }, margins: cellMargins,
      children: [new Paragraph({ children: [new TextRun({ text: r[2], size: 18, font: "Arial" })] })] }),
  ]}));
  return new Table({ width: { size: W_TOTAL, type: WidthType.DXA }, columnWidths: LW, rows: [headerRow, ...dataRows] });
}

// ── clinical rule table ───────────────────────────────────────────
function makeRuleTable() {
  const RW = [2400, 2600, 2800, 3826];
  const headerRow = new TableRow({ tableHeader: true, children: [
    hCell("病症",         RW[0]),
    hCell("S（首选类别）", RW[1]),
    hCell("A（可选类别）", RW[2]),
    hCell("循证依据",     RW[3]),
  ]});
  const rules = [
    ["食物蛋白过敏",    "氨基酸配方",              "乳蛋白深度水解配方",   "ESPGHAN 2022 CMPA 指南"],
    ["乳蛋白过敏高风险","乳蛋白部分水解配方",      "—",                    "ESPGHAN 2022 预防性使用"],
    ["乳糖不耐受_婴儿", "无乳糖配方",              "低乳糖配方",           "GB 25596-2010 婴儿特医通则"],
    ["早产",            "早产/低出生体重婴儿配方", "母乳营养补充剂",       "GB 25596-2010"],
    ["苯丙酮尿症_婴儿", "氨基酸代谢障碍配方",      "—",                    "GB 29922-2013"],
    ["补充蛋白质",      "蛋白质（氨基酸）组件",    "非全营养配方食品",     "ESPEN 2019 临床营养指南"],
    ["消化吸收障碍",    "全营养配方食品",           "非全营养配方食品",     "GB 29922-2013"],
    ["肿瘤",            "特定全营养配方食品",       "全营养配方食品",       "ESPEN 2019 肿瘤营养"],
    ["腹泻脱水",        "电解质配方",              "非全营养配方食品",     "GB 29922-2013"],
    ["吞咽障碍",        "增稠组件",                "全营养配方食品",       "GB 29922-2013"],
    ["苯丙酮尿症",      "非全营养配方食品",        "—",                    "GB 29922-2013"],
  ];
  const dataRows = rules.map((r, i) => {
    const fill = i % 2 === 0 ? "FFFFFF" : "F2F7FC";
    return new TableRow({ children: [
      new TableCell({ borders, width: { size: RW[0], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill, type: ShadingType.CLEAR },
        children: [new Paragraph({ children: [new TextRun({ text: r[0], size: 18, font: "Arial", bold: true })] })] }),
      new TableCell({ borders, width: { size: RW[1], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill: "E2EFDA", type: ShadingType.CLEAR },
        children: [new Paragraph({ children: [new TextRun({ text: r[1], size: 18, font: "Arial" })] })] }),
      new TableCell({ borders, width: { size: RW[2], type: WidthType.DXA }, margins: cellMargins,
        shading: { fill: "FFF2CC", type: ShadingType.CLEAR },
        children: [new Paragraph({ children: [new TextRun({ text: r[2], size: 18, font: "Arial" })] })] }),
      new TableCell({ borders, width: { size: RW[3], type: WidthType.DXA }, margins: cellMargins,
        children: [new Paragraph({ children: [new TextRun({ text: r[3], size: 18, font: "Arial", color: "444444" })] })] }),
    ]});
  });
  return new Table({ width: { size: W_TOTAL, type: WidthType.DXA }, columnWidths: RW, rows: [headerRow, ...dataRows] });
}

// ── build document ────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 600, hanging: 300 } } } }]
    }]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2E74B5" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "2E74B5" },
        paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 16838, height: 11906 },  // A4 landscape in DXA
        margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 }  // ~2cm
      }
    },
    children: [

      // ═══════════════════════════════════════════════════════
      // Chapter 1: Ground Truth 数据概览
      // ═══════════════════════════════════════════════════════
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun({ text: "一、Ground Truth 数据概览", bold: true, size: 36, font: "Arial", color: "1F4E79" })],
        spacing: { before: 0, after: 240 }
      }),

      para("Ground Truth 数据以 CSV 格式存储于 outputs/ground_truth.csv，共 468 条 product-condition 对，用于定量评估推荐系统各路由（Route A/B/C）输出的分级准确性。"),
      spacer(),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: "1.1  列结构", bold: true, size: 28, font: "Arial", color: "2E74B5" })],
        spacing: { before: 200, after: 120 }
      }),
      makeColTable(),
      spacer(),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: "1.2  Label 含义", bold: true, size: 28, font: "Arial", color: "2E74B5" })],
        spacing: { before: 200, after: 120 }
      }),
      makeLabelTable(),
      spacer(),

      // ═══════════════════════════════════════════════════════
      // Chapter 2: Ground Truth 内容
      // ═══════════════════════════════════════════════════════
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun({ text: "二、Ground Truth 内容", bold: true, size: 36, font: "Arial", color: "1F4E79" })],
        spacing: { before: 360, after: 240 }
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: "2.1  各病症分布", bold: true, size: 28, font: "Arial", color: "2E74B5" })],
        spacing: { before: 200, after: 120 }
      }),
      para("下表展示各病症场景下 S / A / ban 三类产品数量，颜色标注：绿色（S列有值）、黄色（A列有值）、红色（ban列有值）。"),
      spacer(),
      makeDistTable(),
      spacer(),
      para("注：消化吸收障碍、吞咽障碍、腹泻脱水、补充蛋白质四个病症共享相同的 51 条非全营养配方食品（A 级），产品在不同病症下重复出现，但含义各自独立。", { color: "666666", italics: true }),

      spacer(),
      spacer(),

      // ═══════════════════════════════════════════════════════
      // Chapter 3: Ground Truth 来源
      // ═══════════════════════════════════════════════════════
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun({ text: "三、Ground Truth 来源与测试逻辑", bold: true, size: 36, font: "Arial", color: "1F4E79" })],
        spacing: { before: 360, after: 240 }
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: "3.1  构建方案：从 CLINICAL_KB 反推（方案一）", bold: true, size: 28, font: "Arial", color: "2E74B5" })],
        spacing: { before: 200, after: 120 }
      }),
      para("采用方案一（从临床规则反推），逻辑完全自洽：直接以 route_a.py 的 CLINICAL_KB 作为金标准，无需医生人工标注，也无需 LLM 生成。"),
      spacer(),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: "3.2  数据来源分层", bold: true, size: 28, font: "Arial", color: "2E74B5" })],
        spacing: { before: 200, after: 120 }
      }),

      new Paragraph({
        heading: HeadingLevel.HEADING_3,
        children: [new TextRun({ text: "第一层：临床规则（CLINICAL_KB）", bold: true, size: 24, font: "Arial" })],
        spacing: { before: 160, after: 80 }
      }),
      para("定义了 11 个病症对应的产品类别等级，例如："),
      codePara("食物蛋白过敏 → preferred      = [\"氨基酸配方\"]"),
      codePara("              alternative   = [\"乳蛋白深度水解配方\"]"),
      codePara("              contraindicated = [\"乳蛋白部分水解配方\"]"),
      spacer(),
      para("规则来源："),
      bullet("ESPGHAN 2022  牛奶蛋白过敏（CMPA）管理指南"),
      bullet("ESPEN 2019    临床营养实践指南"),
      bullet("GB 29922-2013  特殊医学用途配方食品通则"),
      bullet("GB 25596-2010  特殊医学用途婴儿配方食品通则"),
      spacer(),

      new Paragraph({
        heading: HeadingLevel.HEADING_3,
        children: [new TextRun({ text: "第二层：产品数据库（data/result2.xlsx）", bold: true, size: 24, font: "Arial" })],
        spacing: { before: 160, after: 80 }
      }),
      para("将临床规则中的产品类别与数据库中实际注册产品对号入座，由 build_ground_truth.py 自动完成："),
      bullet("产品类别 ∈ preferred        → label = S"),
      bullet("产品类别 ∈ alternative      → label = A"),
      bullet("产品类别 ∈ contraindicated  → label = ban"),
      spacer(),
      makeRuleTable(),
      spacer(),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: "3.3  生成逻辑", bold: true, size: 28, font: "Arial", color: "2E74B5" })],
        spacing: { before: 200, after: 120 }
      }),
      para("构建流程可表示为："),
      spacer(),
      codePara("CLINICAL_KB（病症规则）  ×  result2.xlsx（产品库）"),
      codePara("         ↓"),
      codePara("     ground_truth.csv（468 条 product-condition 对）"),
      spacer(),

      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: "3.4  用途与局限", bold: true, size: 28, font: "Arial", color: "2E74B5" })],
        spacing: { before: 200, after: 120 }
      }),
      para("用途："),
      bullet("计算 Precision / Recall：S 级产品是否被推荐出来、ban 产品是否被过滤"),
      bullet("评估 Route B（营养计算层）和 Route C（LLM 层）是否破坏了 Route A 的正确分级"),
      bullet("纵向对比三条路由：Route A / A+B / A+B+C 各自的准确率"),
      spacer(),
      para("局限："),
      bullet("无法评估 Route B 营养计算本身是否精确，仅评估推荐分级是否正确"),
      bullet("Ground truth 与 CLINICAL_KB 完全同源，若规则本身有误，测试无法发现"),
      bullet("实际临床场景可能存在个体差异，Ground truth 代表的是群体层面的标准答案"),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = 'outputs/ground_truth_report.docx';
  fs.writeFileSync(outPath, buffer);
  console.log('OK: ' + outPath);
}).catch(e => { console.error(e); process.exit(1); });
