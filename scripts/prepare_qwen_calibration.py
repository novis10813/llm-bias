#!/usr/bin/env python
"""Build controlled English, Simplified-Chinese, and mixed calibration corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("data/calibration/qwen3.5-4b")

TOPICS: list[dict[str, Any]] = [
    {
        "domain": "river_management",
        "subject_en": "the management of a river shared by several towns",
        "subject_zh": "多个城镇共同使用的一条河流的管理",
        "facts_en": [
            "water levels change sharply between the wet and dry seasons",
            "cargo boats and fishing families depend on predictable access",
            "upstream construction can alter water quality downstream",
            "officials publish measurements before changing release schedules",
        ],
        "facts_zh": [
            "丰水期和枯水期的水位变化很大",
            "货船和渔民家庭依赖稳定可预期的通航条件",
            "上游建设可能改变下游的水质",
            "管理部门会在调整放水计划前公布监测数据",
        ],
    },
    {
        "domain": "public_library",
        "subject_en": "a public library renovating its oldest reading room",
        "subject_zh": "一家正在翻修最老阅览室的公共图书馆",
        "facts_en": [
            "rare maps must be protected from dust and direct sunlight",
            "students still need quiet desks while construction is under way",
            "digital copies reduce handling of fragile documents",
            "the final layout should remain accessible to older visitors",
        ],
        "facts_zh": [
            "珍贵地图必须避免灰尘和阳光直射",
            "施工期间学生仍然需要安静的阅读座位",
            "数字副本可以减少对脆弱文献的直接翻阅",
            "最终布局应当方便老年访客使用",
        ],
    },
    {
        "domain": "urban_transit",
        "subject_en": "a city redesigning a busy bus corridor",
        "subject_zh": "一座正在改造繁忙公交走廊的城市",
        "facts_en": [
            "delays are concentrated near three crowded intersections",
            "dedicated lanes improve reliability but remove some parking",
            "night workers need service after the usual evening peak",
            "the transport agency will compare travel times every month",
        ],
        "facts_zh": [
            "延误主要集中在三个拥堵的路口附近",
            "公交专用道能提高准点率但会减少部分停车位",
            "夜班员工需要晚高峰之后的公共交通服务",
            "交通部门将每月比较一次通行时间",
        ],
    },
    {
        "domain": "agriculture",
        "subject_en": "a farming cooperative testing a new irrigation plan",
        "subject_zh": "一个正在测试新灌溉方案的农业合作社",
        "facts_en": [
            "soil moisture varies even between neighboring fields",
            "drip lines use less water than open channels",
            "farmers record crop growth before changing fertilizer levels",
            "the trial keeps one field under the previous routine as a control",
        ],
        "facts_zh": [
            "即使相邻田地的土壤湿度也可能不同",
            "滴灌管线比开放式水渠更节约用水",
            "农户会在调整施肥量前记录作物生长情况",
            "试验保留一块继续采用原方案的田地作为对照",
        ],
    },
    {
        "domain": "weather_station",
        "subject_en": "a mountain weather station replacing its sensors",
        "subject_zh": "一座正在更换传感器的山地气象站",
        "facts_en": [
            "strong winds can distort readings from exposed instruments",
            "ice accumulates quickly on equipment during winter storms",
            "overlapping old and new sensors reveals calibration differences",
            "daily observations are transmitted to villages in the valley",
        ],
        "facts_zh": [
            "强风可能使暴露在外的仪器读数失真",
            "冬季暴风雪期间设备会迅速结冰",
            "让新旧传感器同时工作可以发现校准差异",
            "每日观测结果会发送给山谷中的村庄",
        ],
    },
    {
        "domain": "community_clinic",
        "subject_en": "a community clinic reorganizing appointment times",
        "subject_zh": "一家正在重新安排预约时段的社区诊所",
        "facts_en": [
            "morning appointments fill faster than afternoon appointments",
            "urgent cases require space that is not booked in advance",
            "reminder messages reduce missed visits",
            "staff review waiting times without storing unnecessary personal data",
        ],
        "facts_zh": [
            "上午的预约时段比下午更快约满",
            "紧急病例需要预留不能提前预约的空间",
            "提醒信息可以减少患者爽约",
            "工作人员会在不保存多余个人数据的前提下分析等候时间",
        ],
    },
    {
        "domain": "manufacturing",
        "subject_en": "a factory investigating small defects in metal parts",
        "subject_zh": "一家正在调查金属零件微小缺陷的工厂",
        "facts_en": [
            "most defects appear after the machines have run for several hours",
            "temperature and tool wear are recorded for every production batch",
            "inspectors label samples without knowing which machine made them",
            "maintenance is scheduled only after repeated evidence is found",
        ],
        "facts_zh": [
            "多数缺陷出现在机器连续运行数小时之后",
            "每个生产批次都会记录温度和刀具磨损情况",
            "检验员在不知道零件来自哪台机器的情况下标记样本",
            "只有发现重复证据后才会安排设备维护",
        ],
    },
    {
        "domain": "astronomy",
        "subject_en": "an observatory planning a month of telescope time",
        "subject_zh": "一座正在规划一个月望远镜使用时间的天文台",
        "facts_en": [
            "some targets are visible for only a short period before dawn",
            "cloud cover makes flexible scheduling valuable",
            "different instruments are suited to different wavelengths",
            "research teams share processed observations after quality checks",
        ],
        "facts_zh": [
            "有些观测目标只在黎明前短暂可见",
            "云层变化使灵活排期具有重要价值",
            "不同仪器适合观测不同波段",
            "研究团队会在质量检查后共享处理过的观测数据",
        ],
    },
    {
        "domain": "marine_ecology",
        "subject_en": "a coastal team monitoring the recovery of a wetland",
        "subject_zh": "一个正在监测滨海湿地恢复情况的团队",
        "facts_en": [
            "young fish use shallow vegetation as shelter",
            "salinity changes after both storms and long dry periods",
            "bird counts are taken at the same hour each week",
            "nearby residents help report unusual pollution events",
        ],
        "facts_zh": [
            "幼鱼会把浅水区的植被当作庇护场所",
            "暴风雨和长期干旱都会改变水体盐度",
            "鸟类数量在每周相同的时间进行统计",
            "附近居民会协助报告异常污染事件",
        ],
    },
    {
        "domain": "school_experiment",
        "subject_en": "a science class comparing how seeds germinate",
        "subject_zh": "一个比较种子发芽条件的科学课堂",
        "facts_en": [
            "each group changes only one condition at a time",
            "containers receive equal amounts of water",
            "students measure roots with the same ruler every morning",
            "unexpected results are recorded rather than removed",
        ],
        "facts_zh": [
            "每个小组一次只改变一个条件",
            "所有容器得到相同数量的水",
            "学生每天早晨使用同一把尺子测量根部",
            "意外结果会被如实记录而不是删除",
        ],
    },
    {
        "domain": "archaeology",
        "subject_en": "an archaeological team documenting a hillside site",
        "subject_zh": "一个正在记录山坡遗址的考古团队",
        "facts_en": [
            "soil layers are photographed before any object is moved",
            "small fragments can reveal how a vessel was made",
            "rainwater must be diverted away from open trenches",
            "local historians contribute names found in old records",
        ],
        "facts_zh": [
            "移动任何物品前都要拍摄土层照片",
            "细小碎片也可能揭示器皿的制作方式",
            "必须把雨水从开放的探沟旁引走",
            "当地历史研究者会提供旧档案中发现的名称",
        ],
    },
    {
        "domain": "renewable_energy",
        "subject_en": "a region considering battery storage for wind power",
        "subject_zh": "一个正在考虑为风电配置电池储能的地区",
        "facts_en": [
            "electricity production rises when demand is sometimes low",
            "storage can reduce sudden changes sent to the grid",
            "battery lifetime depends on temperature and charging patterns",
            "planners compare cost, safety, and recycling requirements",
        ],
        "facts_zh": [
            "发电量上升时用电需求有时反而较低",
            "储能可以减少输送到电网的突然波动",
            "电池寿命取决于温度和充电模式",
            "规划人员会比较成本、安全性和回收要求",
        ],
    },
    {
        "domain": "software_reliability",
        "subject_en": "a software team preparing a critical service update",
        "subject_zh": "一个正在准备关键服务更新的软件团队",
        "facts_en": [
            "automated tests cover common requests and known failures",
            "a small group of servers receives the update first",
            "engineers monitor latency and error rates during deployment",
            "the previous version remains available for a rapid rollback",
        ],
        "facts_zh": [
            "自动化测试覆盖常见请求和已知故障",
            "少量服务器会率先接收更新",
            "工程师在部署期间监测延迟和错误率",
            "旧版本会保留以便快速回滚",
        ],
    },
    {
        "domain": "supply_chain",
        "subject_en": "a distributor reviewing delays in a cold supply chain",
        "subject_zh": "一家正在检查冷链延误情况的经销商",
        "facts_en": [
            "temperature logs accompany each shipment",
            "road closures can force trucks onto longer routes",
            "warehouses need notice before several deliveries arrive together",
            "managers distinguish isolated incidents from repeated bottlenecks",
        ],
        "facts_zh": [
            "每批货物都附带温度记录",
            "道路封闭可能迫使货车改走更长路线",
            "多批货物同时到达前仓库需要提前收到通知",
            "管理人员会区分偶发事件和反复出现的瓶颈",
        ],
    },
    {
        "domain": "community_festival",
        "subject_en": "a neighborhood organizing an annual street festival",
        "subject_zh": "一个正在筹备年度街区节日的社区",
        "facts_en": [
            "food stalls need access to clean water and waste collection",
            "musicians share a stage with school performances",
            "temporary signs keep emergency routes clear",
            "organizers survey residents after the event",
        ],
        "facts_zh": [
            "食品摊位需要清洁用水和垃圾收集服务",
            "音乐人和学校表演会共用一个舞台",
            "临时标志确保应急通道保持畅通",
            "组织者会在活动结束后调查居民意见",
        ],
    },
    {
        "domain": "market_analysis",
        "subject_en": "an analyst reviewing a company after its quarterly report",
        "subject_zh": "一名在季度报告发布后研究某家公司的分析师",
        "facts_en": [
            "revenue growth differs across business segments",
            "one-time expenses should be separated from recurring costs",
            "management forecasts are compared with independent demand data",
            "the final note states both uncertainty and possible downside risks",
        ],
        "facts_zh": [
            "不同业务板块的收入增长并不一致",
            "一次性支出应当与经常性成本分开",
            "管理层预测会与独立需求数据进行比较",
            "最终报告会同时说明不确定性和潜在下行风险",
        ],
    },
]

STYLES = [
    ("field_report", (
        "A field report examines {subject}. It notes that {f1}. The report also "
        "explains that {f2}. Before reaching a conclusion, the authors compare "
        "these observations with evidence that {f3}.",
        "一份实地报告研究了{subject}。报告指出，{f1}。报告还说明，{f2}。在得出"
        "结论之前，作者把这些观察结果与另一项证据进行比较，即{f3}。",
    )),
    ("planning_note", (
        "A planning group met to discuss {subject}. The first concern was that "
        "{f1}. Participants then considered how {f2}. Their provisional plan "
        "will be reviewed after they confirm that {f3}.",
        "一个规划小组开会讨论了{subject}。首先需要关注的是，{f1}。与会者随后"
        "考虑了{f2}这一情况。他们将在确认{f3}之后重新审查临时方案。",
    )),
    ("cause_and_effect", (
        "When studying {subject}, the team avoided treating every change as an "
        "isolated event. Because {f1}, they tracked whether {f2}. This made it "
        "easier to judge the practical importance of the finding that {f3}.",
        "在研究{subject}时，团队没有把每次变化都当作孤立事件。由于{f1}，他们"
        "持续追踪{f2}这一情况。这有助于判断{f3}这一发现的实际重要性。",
    )),
    ("procedure", (
        "The procedure for reviewing {subject} has three stages. Staff first "
        "verify that {f1}. They next document how {f2}. Finally, a second group "
        "checks the records and decides whether the evidence that {f3} is reliable.",
        "审查{subject}的流程分为三个阶段。工作人员首先核实{f1}。随后记录{f2}"
        "的具体情况。最后由另一个小组检查记录，并判断有关{f3}的证据是否可靠。",
    )),
    ("comparison", (
        "Two teams produced different accounts of {subject}. Both agreed that "
        "{f1}, but they assigned different importance to the fact that {f2}. A "
        "joint review focused on shared measurements showing that {f3}.",
        "两个团队对{subject}提出了不同看法。双方都同意{f1}，但对{f2}这一事实"
        "的重要性判断不同。联合审查重点分析了共同测量结果所显示的{f3}。",
    )),
    ("public_briefing", (
        "At a public briefing about {subject}, a specialist began with the "
        "observation that {f1}. A resident asked whether {f2}. The specialist "
        "answered cautiously and emphasized the available evidence that {f3}.",
        "在一次关于{subject}的公开说明会上，一位专家首先指出{f1}。一名居民"
        "询问了{f2}这一情况。专家谨慎作答，并强调现有证据表明{f3}。",
    )),
    ("historical_record", (
        "Records about {subject} were collected over several years. Early notes "
        "already mentioned that {f1}. Later entries described how {f2}. The most "
        "recent summary adds stronger evidence that {f3}.",
        "有关{subject}的记录持续收集了数年。早期笔记已经提到{f1}。后来的记录"
        "描述了{f2}。最近的总结又补充了更有力的证据，说明{f3}。",
    )),
    ("decision_review", (
        "Before making a decision about {subject}, reviewers listed assumptions "
        "that could change the result. They confirmed that {f1}, tested the claim "
        "that {f2}, and recorded remaining uncertainty about whether {f3}.",
        "在对{subject}作出决定之前，审查人员列出了可能改变结果的假设。他们"
        "确认了{f1}，检验了{f2}这一说法，并记录了对{f3}是否成立的剩余不确定性。",
    )),
]


def _records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    english: list[dict[str, Any]] = []
    chinese: list[dict[str, Any]] = []
    for topic_index, topic in enumerate(TOPICS):
        for style_index, (style, templates) in enumerate(STYLES):
            rotated = [
                (style_index + offset) % len(topic["facts_en"])
                for offset in range(3)
            ]
            pair_id = f"{topic['domain']}-{style}"
            common = {
                "pair_id": pair_id,
                "domain": topic["domain"],
                "style": style,
            }
            english.append(
                {
                    **common,
                    "id": f"en-{topic_index:02d}-{style_index:02d}",
                    "language": "en",
                    "text": templates[0].format(
                        subject=topic["subject_en"],
                        **{
                            f"f{offset + 1}": topic["facts_en"][fact_index]
                            for offset, fact_index in enumerate(rotated)
                        },
                    ),
                }
            )
            chinese.append(
                {
                    **common,
                    "id": f"zh-{topic_index:02d}-{style_index:02d}",
                    "language": "zh-CN",
                    "text": templates[1].format(
                        subject=topic["subject_zh"],
                        **{
                            f"f{offset + 1}": topic["facts_zh"][fact_index]
                            for offset, fact_index in enumerate(rotated)
                        },
                    ),
                }
            )
    return english, chinese


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def build(output_dir: Path, *, model_slug: str | None = None, model_config: dict[str, Any] | None = None, tokenizer_name: str | None = None) -> dict[str, Any]:
    english, chinese = _records()
    mixed = []
    for topic_index in range(len(TOPICS)):
        for style_index in range(len(STYLES)):
            index = topic_index * len(STYLES) + style_index
            source = english if (topic_index + style_index) % 2 == 0 else chinese
            mixed.append(source[index])
    if not len(english) == len(chinese) == len(mixed) == 128:
        raise AssertionError("each calibration condition must contain 128 records")
    for name, rows in (
        ("english", english),
        ("chinese_simplified", chinese),
        ("mixed", mixed),
    ):
        texts = [row["text"] for row in rows]
        if len(texts) != len(set(texts)):
            raise AssertionError(f"{name} contains duplicate text")
    if sum(row["language"] == "en" for row in mixed) != 64:
        raise AssertionError("mixed calibration must contain exactly 64 English rows")
    for domain in {row["domain"] for row in mixed}:
        domain_rows = [row for row in mixed if row["domain"] == domain]
        if sum(row["language"] == "en" for row in domain_rows) != 4:
            raise AssertionError(f"mixed domain is not language-balanced: {domain}")
    for style in {row["style"] for row in mixed}:
        style_rows = [row for row in mixed if row["style"] == style]
        if sum(row["language"] == "en" for row in style_rows) != 8:
            raise AssertionError(f"mixed style is not language-balanced: {style}")

    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        name: _write_jsonl(output_dir / f"{name}.jsonl", rows)
        for name, rows in (
            ("english", english),
            ("chinese_simplified", chinese),
            ("mixed", mixed),
        )
    }
    tokenizer_stats = {}
    if tokenizer_name:
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        lengths = [len(tokenizer(row["text"], add_special_tokens=False).input_ids) for row in english + chinese + mixed]
        tokenizer_stats = {"tokenizer_name_or_path": getattr(tokenizer, "name_or_path", tokenizer_name), "count": len(lengths), "min_tokens": min(lengths), "max_tokens": max(lengths), "mean_tokens": sum(lengths) / len(lengths), "chat_template_sha256": hashlib.sha256(str(getattr(tokenizer, "chat_template", "")).encode()).hexdigest()}
    manifest = {
        "schema_version": 1,
        "design": "parallel_domain_style_matched",
        "conditions": {
            "english": {"count": 128, "languages": {"en": 128}},
            "chinese_simplified": {"count": 128, "languages": {"zh-CN": 128}},
            "mixed": {"count": 128, "languages": {"en": 64, "zh-CN": 64}},
        },
        "domains": [topic["domain"] for topic in TOPICS],
        "styles": [style for style, _templates in STYLES],
        "sha256": hashes,
        "model_slug": model_slug or output_dir.parent.name,
        "model_config": model_config or {},
        "selection_basis": "inherited_qwen3.5_4b_operational_winner" if (model_slug or output_dir.parent.name) == "qwen3.6-27b" else "not_applicable",
        "tokenizer_stats": tokenizer_stats,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-slug", default=None)
    parser.add_argument("--model-config-json", type=Path, default=None)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()
    config = json.loads(args.model_config_json.read_text()) if args.model_config_json else None
    manifest = build(args.output_dir, model_slug=args.model_slug, model_config=config, tokenizer_name=args.tokenizer)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
