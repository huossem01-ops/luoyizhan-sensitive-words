"""Collapse expanded sentence families into a word/short-phrase release.

The raw 10k batches remain untouched for auditability.  This script rewrites
the formal CSV/JSONL dataset and records one audit decision per source row.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from deduplicate import find_similar, write_dataset, write_report
from expansion_catalog import ALL_SPECS
from normalize import normalize_term


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"


def parts(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


CORE_FRAGMENTS = {
    "校园外貌与穿搭": parts(
        "额前刘海|齐刘海|碎刘海|扎马尾|高马尾|发梢|酒窝|低头笑|校服外套|校服领口|"
        "校服袖口|校服衣角|白衬衫|小皮筋|发卡|帆布鞋|运动服|班级号码|黑框眼镜|"
        "衣摆|球鞋|红脸|扎头发|别头发|少年感"
    ),
    "校园物件与暗号": parts(
        "折纸条|小纸条|手写情书|匿名情书|情书信封|便利贴|小心心|草稿纸|同学录|毕业册|"
        "纪念书签|校服纽扣|第二颗纽扣|借书卡|课程表|值日表|错题本|复习资料|同款钢笔|"
        "橡皮刻字|成对钥匙扣|点歌暗号|水杯暗号|纸条回复|没送出的信"
    ),
    "校园空间与路线": parts(
        "靠窗座位|隔壁班|教室门口|空教室|教学楼|走廊尽头|楼梯转角|饮水机|图书馆|自习室|"
        "小卖部|食堂排队|广播站|礼堂散场|操场跑道|篮球场|操场看台|体育馆|林荫道|校门口|"
        "放学路|公交站|书架重逢|雨中操场|天台晚风"
    ),
    "校园日常互动": parts(
        "借笔记|讲错题|对答案|互相听写|课文抽查|帮忙留座|递牛奶|提醒带伞|借外套|食堂带饭|"
        "一起值日|擦黑板|搬桌椅|社团排练|收器材|比赛加油|班级合唱|分享笑话|等收书包|"
        "留一颗糖|压低声音|保管手机|一起自习|交换笔记|课间聊天"
    ),
    "季节与感官记忆": parts(
        "蝉鸣|下课铃|篮球回声|广播点歌|晨光|晚霞|树影|雨滴窗台|跑道雨味|"
        "雨伞温度|热水杯|春日樱花|盛夏风扇|秋日落叶|初雪|六月闷热|晚风|校服摩擦|"
        "冰汽水|洗衣粉香|粉笔灰|书页翻动|操场晚风|夏夜虫鸣"
    ),
    "毕业与离别": parts(
        "毕业照|同学录|毕业册|校服签名|高考结束|散伙饭|最后一课|毕业快乐|离校广播|"
        "最后铃声|课桌清空|旧纸条|最后合照|告别校门|不同城市|慢慢失联|重回母校|泛黄照片|"
        "最后一面|毕业约定|临别拥抱|离校行李|青春散场|各奔东西"
    ),
}


REVIEWED_LONG_EXPRESSIONS = {
    "恋爱": parts("第一次确认彼此都想认真谈恋爱|从普通朋友慢慢变成恋爱关系|两个人开始共同经营一段感情|第一次真正体会到喜欢能够得到回应|拥有一个可以分享日常的恋人"),
    "对象与伴侣": parts("朋友聚会时终于能带对象参加|从一直单身变成拥有稳定伴侣|认真思考自己适合什么样的对象|通过朋友介绍认识合适的恋爱对象|期待遇见愿意长期陪伴自己的人"),
    "表白与追求": parts("把藏了很久的喜欢认真说出口|写完情书以后一直不敢交出去|主动创造和喜欢的人见面机会|收到别人认真准备很久的告白|尊重对方没有接受追求的决定"),
    "约会": parts("第一次单独约喜欢的人看电影|为了第一次约会认真挑选见面穿搭|约会结束以后期待下一次见面|两个人沿着安静的江边慢慢散步|收到对方发来的安全到家消息"),
    "亲密行为": parts("第一次牵住真正喜欢的人的手|靠近以前先询问对方是否愿意|不把沉默和没有反抗当成同意|发现对方不舒服以后立即停止|为没有尊重亲密边界认真道歉"),
    "恋爱历史": parts("第一次喜欢一个人发生在什么时候|被问起过去到底谈过几次恋爱|听现在喜欢的人讲起过去难忘的初恋|整理上一段关系留下来的旧物|自己的恋爱经历至今仍然完全空白"),
    "校园恋爱": parts("毕业以前想认真谈一次校园恋爱|借着讲题机会坐到喜欢的人旁边|放学以后推着自行车并肩回家|运动会偷偷给喜欢的人递一瓶水|从同桌慢慢变成互相喜欢的人"),
    "时间与年龄焦虑": parts("二十二岁还从来没有谈过恋爱|大学毕业进入工作以后仍然没有对象|年龄增长但是恋爱经历没有增加|担心错过最适合体验恋爱的年纪|发现同龄人已经进入稳定关系"),
    "社会比较": parts("朋友圈突然全是同学官宣对象|聚会里只有自己一个人没有对象|同寝室友周末都出去和对象约会|参加同学婚礼时被问有没有对象|看见朋友收到恋爱纪念日礼物"),
    "被拒绝": parts("认真准备告白以后却收到明确拒绝|对方说只愿意把自己当普通朋友|每一次约对方见面都被推到下次|接受没有回复本身就是一种回答|被拒绝以后不再继续联系对方"),
    "吸引力与自我评价": parts("担心自己的外貌没有任何吸引力|面对喜欢的人总是不知道聊什么|怀疑从来没人表白是因为不够好|读不懂别人是否在释放好感信号|不再用有没有对象证明个人价值"),
    "婚姻": parts("还没有对象就已经被家里人催婚|收到同龄朋友寄来的婚礼请柬|恋人开始认真讨论未来共同生活|第一次带长期交往对象见父母|从恋爱关系逐渐走向组建家庭"),
    "节日": parts("情人节街上到处都是成双情侣|七夕收到平台推送的情侣套餐|为了纪念日提前准备一份礼物|跨年倒数时身边没有喜欢的人|看到别人在520公开恋爱关系"),
    "网络恋爱语言": parts("聊天秒回到底算不算喜欢一个人|长期只聊天不见面算什么关系|网络认识的人终于决定线下见面|暧昧期突然断联代表关系结束吗|把重要聊天和心动对象设置置顶"),
    "恋爱统计与提问": parts("为什么这么多年还没有谈过恋爱|第一次真正谈恋爱是在多大年龄|最长的一段恋爱到底谈了多久|被追问什么时候才能成功脱单|有没有真正认真喜欢过某一个人"),
    "间接语义触发词": parts("电影院最后一排双人座旁边一直空着|看到两张电影票一直并排放在桌上|副驾驶座位上一直放着一束红玫瑰|手机相册里面从来没有情侣合照|饮品店不断推送双人情侣套餐"),
    "高阶语义关联": parts("觉得没有恋爱让人生体验不完整|担心自己的人生进度已经落后同龄人|把成功脱单当作重要人生里程碑|青春回忆里面缺少一段感情故事|学习接受每个人有不同人生节奏"),
    "荒诞扩散区": parts("面向对象编程却始终找不到对象|对象引用为空请检查恋爱作用域|单例模式坚持永远不允许脱单|关系数据库正在认真维护恋爱关系|配对算法决定给所有线程安排相亲"),
    "校园外貌与穿搭": parts("风吹起刘海时露出的躲闪目光|穿校服的背影在人群里格外熟悉|手腕上的小皮筋像是关系暗号|白衬衫衣角被操场晚风轻轻吹起|多年后仍记得校服领口的洗衣粉香"),
    "校园物件与暗号": parts("夹在课本里一直没送出去的情书|从教室最后一排传过来的小纸条|写满留言和名字的毕业纪念册|校服第二颗纽扣被当成青春信物|两个人约定用水杯位置传递暗号"),
    "校园空间与路线": parts("为了偶遇每天绕过隔壁班教室门口|放学以后沿着操场跑道并肩散步|在图书馆同一排书架反复遇见对方|晚自习结束后在校门口等喜欢的人|毕业以后重走两个人走过的放学路"),
    "校园日常互动": parts("把整理好的课堂笔记借给喜欢的人|讲错题时故意把椅子挪得更近|值日结束以后一起留下来擦黑板|记住对方在食堂从来不吃哪道菜|把最后一颗糖悄悄留给喜欢的人"),
    "季节与感官记忆": parts("蝉鸣响起时突然想起穿校服的夏天|下课铃盖过了一句没有说完的话|雨后塑胶跑道的气味唤起校园回忆|晚霞穿过教学楼窗户落在课桌上|多年后仍记得新课本的油墨气味"),
    "毕业与离别": parts("毕业照里悄悄站在喜欢的人旁边|离校以前最后看一次空荡教室|高考结束以后两个人去了不同城市|最后一句聊天一直停在毕业快乐|多年后翻到写满名字的旧同学录"),
}


def length_bucket(term: str) -> str:
    size = len(term)
    if 2 <= size <= 6:
        return "2-6"
    if size <= 12:
        return "7-12"
    if size <= 20:
        return "13-20"
    if size <= 30:
        return "21-30"
    return ">30"


def distribution(records: list[dict]) -> dict:
    counts = Counter(length_bucket(record["term"]) for record in records)
    total = len(records)
    return {
        bucket: {"count": counts[bucket], "share": round(counts[bucket] / total, 6)}
        for bucket in ("2-6", "7-12", "13-20", "21-30", ">30")
    }


def extract_concept(record: dict) -> str | None:
    if not record["reason"].startswith("通过"):
        return None
    concepts = ALL_SPECS[record["category"]]["concepts"]
    matches = [concept for concept in concepts if concept in record["term"]]
    return max(matches, key=len) if matches else None


def cleaned_record(term: str, source: dict, reason: str, extra_tag: str) -> dict:
    return {
        "id": 0,
        "term": term,
        "normalized_term": normalize_term(term),
        "category": source["category"],
        "subcategory": source["subcategory"],
        "trigger_level": source["trigger_level"],
        "semantic_distance": source["semantic_distance"],
        "humor": source["humor"],
        "reason": reason,
        "tags": list(dict.fromkeys([source["category"], source["subcategory"], extra_tag])),
    }


def round_robin_select(records: list[dict], limit: int) -> list[dict]:
    groups: dict[str, deque] = defaultdict(deque)
    for record in records:
        groups[record["category"]].append(record)
    selected = []
    categories = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for category in categories:
            if groups[category] and len(selected) < limit:
                selected.append(groups[category].popleft())
    return selected


def build_clean_release(source_records: list[dict]) -> list[dict]:
    candidates: dict[str, dict] = {}

    # High-precision seed terms were directly authored and are retained if <=20.
    for record in source_records:
        if not record["reason"].startswith("通过") and 2 <= len(record["term"]) <= 20:
            candidates.setdefault(record["normalized_term"], dict(record))

    # Every expanded family collapses to one underlying concept phrase.
    concept_sources = {}
    for record in source_records:
        concept = extract_concept(record)
        if concept and 2 <= len(concept) <= 20:
            key = normalize_term(concept)
            concept_sources.setdefault(key, record)
            candidates.setdefault(key, cleaned_record(
                concept,
                record,
                "从长句变体族中抽取并审核的核心概念短语",
                "短语化清洗",
            ))

    # Short fragments are noun/action nodes explicitly present inside source concepts.
    representative = {}
    for record in source_records:
        representative.setdefault(record["category"], record)
    for category, terms in CORE_FRAGMENTS.items():
        source = representative[category]
        for term in terms:
            if not 2 <= len(term) <= 6:
                raise ValueError(f"核心片段长度不在 2..6：{term}")
            key = normalize_term(term)
            candidates.setdefault(key, cleaned_record(
                term,
                source,
                "从现有长句中的校园意象拆解出的核心词汇",
                "核心词汇",
            ))

    all_candidates = list(candidates.values())
    short = [r for r in all_candidates if length_bucket(r["term"]) == "2-6"]
    medium = [r for r in all_candidates if length_bucket(r["term"]) == "7-12"]

    # Long expressions are explicitly reviewed concept composites.  They do
    # not use the expansion engine's interchangeable time/place prefixes.
    long_candidates = []
    for category, terms in REVIEWED_LONG_EXPRESSIONS.items():
        source = representative[category]
        for term in terms:
            if not 13 <= len(term) <= 20:
                raise ValueError(f"审核长短语长度不在 13..20：{term}（{len(term)}）")
            long_candidates.append(cleaned_record(
                term,
                source,
                "经短语化审核保留的复合概念表达；不依赖可替换的时地模板",
                "较长固定表达",
            ))

    # Anchor the release size to available short nodes, then enforce 45/40/14.
    target_total = round(len(short) / 0.45)
    medium_limit = min(len(medium), round(target_total * 0.40))
    long_limit = min(len(long_candidates), round(target_total * 0.14))
    selected = list(short)
    selected.extend(round_robin_select(medium, medium_limit))
    selected.extend(round_robin_select(long_candidates, long_limit))

    # No 21-30 bucket is forced: it is a ceiling, not a quota.
    unique = {}
    for record in selected:
        unique.setdefault(record["normalized_term"], record)
    final = list(unique.values())
    for identifier, record in enumerate(final, start=1):
        record["id"] = identifier
    return final


def write_audit(source_records: list[dict], final_records: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final_by_normalized = {record["normalized_term"]: record for record in final_records}
    rows = []
    actions = Counter()
    for record in source_records:
        normalized = record["normalized_term"]
        concept = extract_concept(record)
        concept_normalized = normalize_term(concept) if concept else None
        if normalized in final_by_normalized:
            action = "kept"
            result = final_by_normalized[normalized]["term"]
            note = "原词符合长度与短语规则"
        elif concept_normalized in final_by_normalized:
            action = "compressed"
            result = final_by_normalized[concept_normalized]["term"]
            note = "移除时间、地点、人物或关系模板，归并到核心概念"
        else:
            action = "deleted"
            result = ""
            note = "模板变体、超长表达或配额外的同概念近似项"
        actions[action] += 1
        rows.append({
            "source_id": record["id"],
            "source_term": record["term"],
            "source_length": len(record["term"]),
            "action": action,
            "result_term": result,
            "result_length": len(result),
            "note": note,
        })

    with (REPORT_DIR / "cleaning_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "source_records": len(source_records),
        "final_records": len(final_records),
        "source_length_distribution": distribution(source_records),
        "final_length_distribution": distribution(final_records),
        "actions": dict(actions),
        "rules": {
            "target_2_6": "about 45%",
            "target_7_12": "about 40%",
            "target_13_20": "about 14%",
            "max_21_30": "1%",
            "over_30": "forbidden",
        },
    }
    with (REPORT_DIR / "cleaning_report.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    source_records = []
    raw_paths = sorted(RAW_DIR.glob("batch_*.jsonl"))
    if not raw_paths:
        raise FileNotFoundError("未发现 data/raw/batch_*.jsonl")
    for path in raw_paths:
        with path.open(encoding="utf-8") as handle:
            source_records.extend(json.loads(line) for line in handle if line.strip())
    final_records = build_clean_release(source_records)
    write_audit(source_records, final_records)
    write_dataset(final_records)
    duplicate_candidates = find_similar(final_records)
    write_report(duplicate_candidates)
    print(
        f"source={len(source_records)} final={len(final_records)} "
        f"duplicate_candidates={len(duplicate_candidates)}"
    )
    print(json.dumps(distribution(final_records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
