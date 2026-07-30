#!/usr/bin/env python
"""Build a bilingual paired holdout set for Jacobian-lens candidate selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_OUTPUT = Path(
    "data/evaluations/qwen3.5-4b/bilingual_intermediate_holdout.jsonl"
)

PAIRS = [
    ("carnival_ocean", "The ocean beside the country most famous for Carnival is the", "狂欢节最著名的国家濒临的海洋是", "Brazil", "巴西", "Atlantic", "大西洋"),
    ("amazon_language", "The main language of the country where the Amazon River reaches the sea is", "亚马孙河入海口所在国家的主要语言是", "Brazil", "巴西", "Portuguese", "葡萄牙语"),
    ("web_animal_legs", "The number of legs on the animal that spins webs is", "会织网的动物有多少条腿，答案是", "spider", "蜘蛛", "eight", "八"),
    ("fourth_planet_color", "The usual color associated with the fourth planet from the Sun is", "与太阳系第四颗行星最常联系的颜色是", "Mars", "火星", "red", "红色"),
    ("paper_continent", "The continent containing the country where paper was invented is", "发明造纸术的国家所在的大洲是", "China", "中国", "Asia", "亚洲"),
    ("boot_country_capital", "The capital of the European country shaped like a boot is", "那个形状像靴子的欧洲国家的首都是", "Italy", "意大利", "Rome", "罗马"),
    ("eiffel_currency", "The currency used in the country containing the Eiffel Tower is the", "拥有埃菲尔铁塔的国家所使用的货币是", "France", "法国", "euro", "欧元"),
    ("pyramid_river", "The river flowing past the country famous for the pyramids is the", "流经以金字塔闻名的国家的河流是", "Egypt", "埃及", "Nile", "尼罗河"),
    ("wool_animal", "The farm animal whose coat is commonly turned into wool is a", "身上的毛通常被制成羊毛的农场动物是", "sheep", "羊", "sheep", "羊"),
    ("honey_insect", "The insect that visits flowers and stores honey is a", "采花并储存蜂蜜的昆虫是", "bee", "蜂", "bee", "蜜蜂"),
    ("barking_animal", "The domestic animal most closely associated with barking is a", "最常与吠叫联系在一起的家养动物是", "dog", "狗", "dog", "狗"),
    ("purring_animal", "The domestic animal known for purring when comfortable is a", "感到舒适时会发出呼噜声的家养动物是", "cat", "猫", "cat", "猫"),
    ("trunk_animal", "The very large land animal that uses a trunk to lift objects is an", "用长鼻子卷起物品的大型陆地动物是", "elephant", "象", "elephant", "大象"),
    ("shell_animal", "The slow animal protected by a hard shell is a", "受到坚硬外壳保护并且行动缓慢的动物是", "turtle", "龟", "turtle", "乌龟"),
    ("howling_animal", "The wild animal that lives in packs and is famous for howling is a", "成群生活并以嚎叫闻名的野生动物是", "wolf", "狼", "wolf", "狼"),
    ("hibernating_animal", "The large animal often described as hibernating through winter is a", "经常被描述为冬季冬眠的大型动物是", "bear", "熊", "bear", "熊"),
    ("antler_animal", "The woodland animal whose males commonly grow antlers is a", "雄性通常会长出鹿角的林地动物是", "deer", "鹿", "deer", "鹿"),
    ("saddle_animal", "The animal traditionally fitted with a saddle for riding is a", "传统上会装上马鞍供人骑乘的动物是", "horse", "马", "horse", "马"),
    ("dairy_animal", "The farm animal most commonly associated with producing milk is a", "最常与生产牛奶联系在一起的农场动物是", "cow", "牛", "cow", "奶牛"),
    ("quacking_animal", "The water bird whose call is usually written as quack is a", "叫声通常被写成嘎嘎声的水鸟是", "duck", "鸭", "duck", "鸭子"),
    ("prey_bird", "The large bird of prey often used as a national symbol is an", "经常被用作国家象征的大型猛禽是", "eagle", "鹰", "eagle", "鹰"),
    ("frozen_water", "When liquid water freezes, the solid material formed is", "液态水冻结后形成的固体物质是", "ice", "冰", "ice", "冰"),
    ("condensed_vapor", "When ordinary vapor cools and condenses, the liquid formed is", "普通水蒸气冷却凝结后形成的液体是", "water", "水", "water", "水"),
    ("opposite_small", "The ordinary opposite of the word small is", "“小”这个词通常对应的反义词是", "big", "大", "big", "大"),
    ("opposite_hot", "The ordinary opposite of the word hot is", "“热”这个词通常对应的反义词是", "cold", "冷", "cold", "冷"),
    ("season_after_summer", "The season that follows summer is", "夏季之后的季节是", "autumn", "秋", "autumn", "秋季"),
    ("sunrise_direction", "The compass direction where the Sun appears to rise is", "太阳升起时所在的方向是", "east", "东", "east", "东方"),
    ("opposite_north", "The compass direction opposite north is", "与北方相反的方向是", "south", "南", "south", "南方"),
    ("clear_sky_color", "The color most often used to describe a clear daytime sky is", "最常用来形容晴朗白天天空的颜色是", "blue", "蓝", "blue", "蓝色"),
    ("banana_color", "The typical color of a ripe banana is", "成熟香蕉通常呈现的颜色是", "yellow", "黄", "yellow", "黄色"),
    ("grass_color", "The typical color of healthy grass is", "健康草地通常呈现的颜色是", "green", "绿", "green", "绿色"),
    ("snow_color", "The color conventionally associated with fresh snow is", "通常与新雪联系在一起的颜色是", "white", "白", "white", "白色"),
]


def build(output: Path) -> dict[str, object]:
    rows = []
    for index, (
        name,
        prompt_en,
        prompt_zh,
        concept_en,
        concept_zh,
        target_en,
        target_zh,
    ) in enumerate(PAIRS):
        for language, prompt, native, cross, target in (
            ("en", prompt_en, concept_en, concept_zh, target_en),
            ("zh-CN", prompt_zh, concept_zh, concept_en, target_zh),
        ):
            rows.append(
                {
                    "id": f"{language}-{index:02d}",
                    "pair_id": name,
                    "language": language,
                    "prompt": prompt,
                    "native_intermediate": native,
                    "crosslingual_intermediate": cross,
                    "target": target,
                }
            )
    if len(rows) != 64 or len({row["id"] for row in rows}) != 64:
        raise AssertionError("holdout must contain 64 unique bilingual rows")
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "design": "32_semantically_paired_bilingual_intermediate_prompts",
        "count": 64,
        "languages": {"en": 32, "zh-CN": 32},
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
