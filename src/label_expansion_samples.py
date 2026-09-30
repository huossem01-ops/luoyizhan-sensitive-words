"""Label the 240 category-balanced targeted-expansion review samples."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "expansion_review_samples.csv"
OUTPUT = ROOT / "reports" / "expansion_review_samples_labeled.csv"
SUMMARY = ROOT / "reports" / "expansion_review_summary.json"

KEEP = {
    "青春感官与毕业": {
        "晚霞", "晨光", "铃声", "雨伞", "晨曦", "朝霞", "霞光", "彩霞",
        "彩霞满天", "送别", "惜别", "晨露", "暮色", "夜色", "落日", "日落",
        "微光", "夜幕", "远别", "日光", "月光", "秋色", "细雨", "伞下",
        "月色", "曙光", "晴空", "离去", "天边", "夜空", "日出",
    },
    "校园空间与路线": {
        "篮球场", "图书馆", "体育馆", "广播站", "小卖部", "走廊", "看台",
        "公交站", "公交车站", "图书室", "运动场", "小卖店", "球场", "体育场",
        "足球场", "观众台", "长廊", "阅览室", "廊道", "田径场", "站牌",
        "实验楼", "学生宿舍", "林荫道", "绿荫",
    },
    "校园日常互动": {
        "等车", "合影留念", "美照", "后座", "拍照", "生活照", "大头照",
        "照相", "留影", "留念", "分享照片", "同车",
    },
    "校园外貌与信物": {
        "刘海", "马尾", "发梢", "发卡", "酒窝", "红脸", "发丝", "发夹", "系扣", "发根",
        "头发", "按扣", "额发", "发际", "发髻", "暗扣", "秀发", "卷发",
        "直发", "脸颊", "脸型", "发质", "橡皮筋", "梳发", "鬓发", "发卷",
        "圆脸", "发色", "鹅蛋脸", "辫子", "瓜子脸",
    },
    "约会物件与仪式": {
        "玫瑰", "奶茶", "影院", "儿童乐园", "新年礼物", "乐园", "纪念",
        "礼盒", "放映厅", "电影城", "游乐区", "影剧院", "鲜花", "送礼",
        "主题乐园", "游乐", "戏院", "影城", "咖啡", "欢乐谷", "电影票",
        "剧院", "游园", "奶盖", "观影", "两周年", "摩天轮",
    },
    "纸面心意与暗号": {
        "纸条", "信封", "书信", "书签", "明信片", "日记本", "暗号", "一封信", "信件",
        "信笺", "写信", "字条", "日记", "来信", "信函", "纸片", "寄信",
        "回信", "密信", "便条", "纸卷", "信中", "纸张", "笔记本", "稿纸",
        "送信", "写日记",
    },
}

BORDERLINE = {
    "青春感官与毕业": {"铃响", "闹铃", "晨钟", "雨衣", "毕业生", "天空", "电话铃"},
    "校园空间与路线": {"公交站点", "公交", "公交车", "体育中心", "地铁站", "走道", "公交线", "场馆", "汽车站", "火车站", "站点", "公交线路", "车站", "比赛场地"},
    "校园日常互动": {"上车", "下车时", "上下车", "车里", "车上", "打车", "让座", "拍张", "晒图"},
    "校园外貌与信物": {"白脸", "铜扣", "大脸"},
    "约会物件与仪式": {"礼品", "花朵", "饮品", "纪念活动", "儿童节", "中秋节", "除夕"},
    "纸面心意与暗号": {"小纸", "纸包", "本子", "家书", "纸袋"},
}


def main() -> None:
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 240:
        raise RuntimeError(f"expected at least 240 rows, got {len(rows)}")
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        category, term = row["target_category"], row["term"]
        if term in KEEP.get(category, set()):
            decision = "keep"
            comment = "真实腾讯词表词条，属于该遗漏概念节点的稳定词或常用短语"
        elif term in BORDERLINE.get(category, set()):
            decision = "borderline"
            comment = "与主题存在场景联系，但恋爱/校园联想依赖上下文"
        else:
            decision = "reject"
            comment = "泛词、碎片、错误变体或从种子产生的非目标语义漂移"
        row["decision"], row["comment"] = decision, comment
        counts[category][decision] += 1
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "total_reviewed": len(rows),
        "decision_distribution": dict(Counter(row["decision"] for row in rows)),
        "by_category": {category: dict(count) for category, count in counts.items()},
        "policy": "Legacy generated data supplied concept gaps only; every reviewed candidate is present in Tencent vocabulary and absent from the original Top-5000 pool.",
    }
    with SUMMARY.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
