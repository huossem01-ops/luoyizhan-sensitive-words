"""Second-stage semantic graph for the 10,000-record release.

The expansion is based on concept cards rather than surface paraphrases.  Each
card combines a compatible time/scene node with an event node, then expresses
four distinct semantic relations: occurrence, memory, social comparison, and
absence/counterfactual experience.
"""

from __future__ import annotations

from collections.abc import Iterator


def split(value: str) -> list[str]:
    return [part.strip() for part in value.split("|") if part.strip()]


CAMPUS_RELATIONS = (
    ("场景共现", "{context}{concept}"),
    ("回忆联想", "{context}{concept}，成了后来常想起的画面"),
    ("社会比较", "看到别人也在{context}{concept}"),
    ("缺席体验", "遗憾校园时光里没有过{context}{concept}的画面"),
)

GENERAL_RELATIONS = (
    ("场景共现", "{context}{concept}"),
    ("回忆联想", "{context}{concept}，后来常被想起"),
    ("他人触发", "听别人说起{concept}，便联想到{context}自己的反应"),
    ("缺席体验", "{context}意识到自己一直没体验过{concept}"),
)


CAMPUS_CONTEXTS = split(
    "早读晨光里|课间走廊上|午休醒来后|体育课自由活动时|"
    "晚自习散场后|放学人群里|周末前最后一节课后|雨落窗台的下午|"
    "暑气未散的放学路上|毕业倒计时里"
)


def spec(subcategories: str, concepts: str, contexts: str, level: int, distance: int, *tags: str, humor: bool = False) -> dict:
    return {
        "subcategories": split(subcategories),
        "concepts": split(concepts),
        "contexts": split(contexts),
        "trigger_level": level,
        "semantic_distance": distance,
        "tags": list(tags),
        "humor": humor,
    }


# Seven campus-centred categories receive 800 records each in the 10k release.
CAMPUS_SPECS: dict[str, dict] = {
    "校园恋爱": spec(
        "课堂心动|课间靠近|放学同行|校园关系确认",
        "和喜欢的人隔着课桌对视|发现同桌悄悄看向自己|把喜欢藏在课堂笔记里|"
        "借讲题的机会坐得更近|在走廊等喜欢的人经过|课间绕路去隔壁班门口|"
        "替心动对象去小卖部带饮料|在人群中寻找熟悉的校服背影|放学后并肩走出校门|"
        "推着自行车陪对方慢慢走|一起等最后一班公交车|约好第二天在校门口见|"
        "在操场看台说出喜欢|把暗恋告诉最信任的同学|收到对方认真回应的好感|"
        "从同桌变成互相喜欢的人|在班级起哄中偷偷脸红|交换只有两个人懂的暗号|"
        "第一次用恋人的身份说晚安|约定毕业以后也继续联系",
        "|".join(CAMPUS_CONTEXTS), 5, 1, "青春", "校园关系"
    ),
    "校园外貌与穿搭": spec(
        "头发与神态|校服细节|随身装饰|少年感印象",
        "风吹起她额前的刘海|看见她把刘海别到耳后|注意到阳光落在她的发梢|"
        "记住她低头笑时的酒窝|发现她见面前认真扎好头发|看见她穿着宽大的校服外套|"
        "认出人群里熟悉的校服衣角|替对方整理歪掉的校服领口|注意到校服袖口写着名字|"
        "看见白衬衫被晚霞染成金色|发现手腕上多了一根小皮筋|把对方送的小皮筋戴在手腕|"
        "认出书包上新挂的钥匙扣|留意她每天更换的发卡|记住帆布鞋踩过积水的声音|"
        "看见运动服背后的班级号码|注意到对方紧张时攥住衣角|记住少年骑车时鼓起的衣摆|"
        "发现镜片后躲闪的目光|在人群中认出那双熟悉的球鞋",
        "|".join(CAMPUS_CONTEXTS), 3, 3, "刘海", "校服", "小皮筋", "外貌意象"
    ),
    "校园物件与暗号": spec(
        "纸面信息|交换信物|学习用品|秘密媒介",
        "递来一张折好的小纸条|在纸条背面写下回复|把没送出的情书夹进书里|"
        "收到一封没有署名的情书|在便利贴上画一颗小心心|把名字写在同一张草稿纸上|"
        "交换一根代表好感的小皮筋|把校服第二颗纽扣留作纪念|送出成对的钥匙扣|"
        "互换写着祝福的书签|借出一支舍不得收回的笔|在橡皮上刻下对方的缩写|"
        "把复习资料装订得整整齐齐|在错题本角落留下鼓励|用同一本练习册对答案|"
        "把借书卡藏进喜欢的书里|用值日表制造一起留下的机会|在课程表上圈出共同的空课|"
        "用广播点歌传递含蓄心意|约定用窗台上的水杯当暗号",
        "|".join(CAMPUS_CONTEXTS), 3, 3, "小纸条", "情书", "信物", "暗号"
    ),
    "校园空间与路线": spec(
        "教室内部|公共校园空间|运动区域|放学路线",
        "在靠窗的座位等她出现|经过对方班级的教室门口|在空教室里多停留一会儿|"
        "从教学楼走廊远远看见她|在楼梯转角意外遇见喜欢的人|绕路经过她常去的饮水机|"
        "在图书馆同一排书架重逢|在自习室替对方留一个座位|在小卖部门口假装偶遇|"
        "在食堂排队时站到她身后|在校园广播站等一首点歌|在礼堂散场后寻找对方|"
        "沿着操场跑道陪她慢慢走|在篮球场边递上一瓶水|在看台最高一排并肩坐着|"
        "在体育馆门口等训练结束|沿林荫道推着自行车同行|在校门口犹豫要不要告别|"
        "在公交站陪对方等车|绕远路走过两个人常走的小巷",
        "|".join(CAMPUS_CONTEXTS), 3, 3, "教室", "操场", "路线", "空间意象"
    ),
    "校园日常互动": spec(
        "学习协作|生活照顾|集体活动|含蓄靠近",
        "把课堂笔记借给喜欢的人|耐心讲一道做错的题|一起背诵第二天要抽查的课文|"
        "互相检查听写答案|替对方保管暂时没收的手机|给趴在桌上的人留一盒牛奶|"
        "提醒喜欢的人带伞|把自己的外套借给对方|记住她不吃的食堂菜|"
        "替迟到的人悄悄留门|在值日时一起擦黑板|一起搬运动会用的桌椅|"
        "在社团排练后帮忙收器材|为对方的比赛偷偷加油|在班级合唱时站到她旁边|"
        "假装随口询问周末安排|把好笑的事情第一个告诉她|聊天时等对方收拾完书包|"
        "看见她睡着后压低说话声音|把最后一颗糖留给喜欢的人",
        "|".join(CAMPUS_CONTEXTS), 4, 2, "校园日常", "照顾", "含蓄好感"
    ),
    "季节与感官记忆": spec(
        "声音记忆|光影记忆|气味与温度|四季校园",
        "听见窗外一阵很长的蝉鸣|听见下课铃盖过没说完的话|记住篮球落地的回声|"
        "听见广播里放过的那首歌|看见晨光落在她的课桌上|看见晚霞穿过教学楼窗户|"
        "记住树影在校服上轻轻晃动|看见雨水沿着教室玻璃滑下|闻到新课本混着油墨的气味|"
        "记住雨后塑胶跑道的气味|感到共用雨伞时靠近的温度|记住冬天热水杯传来的暖意|"
        "等春天樱花落在她肩上|在盛夏风扇声里偷偷心动|看秋叶堆满放学的小路|"
        "在初雪那天想和她一起走|记住六月毕业季的闷热|在傍晚风里听见校服摩擦声|"
        "尝到小卖部冰汽水的甜味|记住她经过时淡淡的洗衣粉香",
        "|".join(CAMPUS_CONTEXTS), 2, 3, "蝉鸣", "季节", "感官", "青春记忆"
    ),
    "毕业与离别": spec(
        "毕业仪式|未说出口|分别时刻|多年回望",
        "在毕业册上写下一句祝福|把同学录最后一页留给她|在毕业照里悄悄站得更近|"
        "在校服背面签下名字|把没送出的情书带回家|错过毕业前最后一次告白|"
        "把喜欢藏进高考结束的夏天|约定收到录取通知后再见|在散伙饭上忍住没说的话|"
        "听完最后一次校园广播|收拾课桌时找到旧纸条|离校前再走一遍熟悉的操场|"
        "在校门口说出以后常联系|目送对方坐上不同方向的车|发现最后一句聊天停在毕业快乐|"
        "因为去往不同城市慢慢失联|多年后翻到泛黄的毕业照|重回母校时寻找当年的座位|"
        "听见校歌突然想起喜欢的人|意识到那次挥手已经是最后一面",
        "|".join(CAMPUS_CONTEXTS), 4, 2, "毕业", "离别", "青春遗憾"
    ),
}


COMMON_CONTEXTS = split("初次认识时|关系升温后|朋友突然问起时|一个人失眠时|周末单独见面时|刷到相似故事时|多年后重新想起时")


# Existing categories get 256/259 records. Concepts are category-specific and
# contexts alter time, witness, medium, and retrospective stance.
GENERAL_SPECS: dict[str, dict] = {
    "恋爱": spec("关系建立|情感体验", "第一次确认恋爱关系|感受到双向喜欢|学会经营一段感情|拥有可以分享日常的人|期待稳定的情感回应|从朋友走向恋人|认真讨论彼此的需要|发现关系正在升温|体验被人惦记的感觉|允许恋爱自然发生", "|".join(COMMON_CONTEXTS), 5, 1, "恋爱体验"),
    "对象与伴侣": spec("伴侣状态|寻找伴侣", "终于有了稳定对象|介绍身边的人是另一半|从单身状态顺利毕业|朋友介绍了合适的人|认真思考理想伴侣|遇到愿意互相陪伴的人|开始主动扩大社交圈|期待有人一起规划周末|不再独自参加朋友聚会|寻找能够长期相处的人", "|".join(COMMON_CONTEXTS), 5, 1, "伴侣", "脱单"),
    "表白与追求": spec("表达心意|追求过程", "鼓起勇气说出喜欢|把暗恋写进一封信|试探对方是否也有好感|主动创造单独见面的机会|收到别人认真的告白|准备一份有意义的礼物|请朋友帮忙判断好感信号|决定不再隐藏真实心意|尊重对方对追求的回应|把模糊关系说清楚", "|".join(COMMON_CONTEXTS), 5, 1, "表白", "追求"),
    "约会": spec("约会安排|相处氛围", "计划第一次正式约会|邀请喜欢的人一起吃饭|挑选适合聊天的咖啡店|一起看期待很久的电影|沿着安静的小路散步|为两个人安排周末行程|见面前认真选择穿搭|约会结束后舍不得分别|等待对方发来安全到家|期待下一次单独见面", "|".join(COMMON_CONTEXTS), 5, 1, "约会", "双人活动"),
    "亲密行为": spec("亲密互动|边界与同意", "第一次牵住喜欢的人的手|得到同意后拥抱对方|在靠近前先询问意愿|注意到对方不舒服的信号|明确拒绝后立即停止|不把沉默误认为同意|共同商量可以接受的边界|为越界接触认真道歉|尊重对方改变主意|在安全感中逐渐靠近", "|".join(COMMON_CONTEXTS), 5, 1, "亲密", "明确同意", "边界"),
    "恋爱历史": spec("过去关系|经验盘点", "回忆自己的第一次心动|谈起已经结束的上一段关系|被问到过去有几次恋爱|发现恋爱履历仍然空白|听对方说起难忘的初恋|整理前任留下的旧物|思考过去关系里的问题|接受曾经喜欢的人已走远|意识到自己从没被表白|不再用经历数量评价自己", "|".join(COMMON_CONTEXTS), 4, 1, "初恋", "前任", "经历"),
    "时间与年龄焦虑": spec("年龄节点|人生节点", "发现又长一岁仍然单身|担心毕业前没有恋爱经历|看着同龄人进入稳定关系|把生日变成脱单倒计时|害怕错过年轻时的恋爱|觉得感情进度落后别人|工作以后才开始学习约会|怀疑现在恋爱是否太晚|计算自己单身了多少年|提醒自己没有统一人生时钟", "|".join(COMMON_CONTEXTS), 5, 1, "年龄", "进度焦虑"),
    "社会比较": spec("同龄比较|社交展示", "看到室友突然宣布脱单|发现聚会里只有自己单身|刷到朋友晒情侣合照|听同学讨论各自的前任|参加婚礼时被问有没有对象|看别人收到纪念日礼物|发现朋友周末都去约会|听见同龄人开始谈婚论嫁|把自己的经历和别人比较|提醒自己不必追赶他人进度", "|".join(COMMON_CONTEXTS), 5, 1, "同龄人", "社会比较"),
    "被拒绝": spec("拒绝表达|拒绝后复盘", "告白后收到明确拒绝|听见我们还是做朋友吧|邀约连续几次没有回应|发现对方正在逐渐疏远|接受喜欢没有得到回应|停止追问拒绝的具体理由|被拒绝后不再继续打扰|重新建立交往边界|删除没有发出的解释消息|把没有回复也视作答案", "|".join(COMMON_CONTEXTS), 5, 1, "拒绝", "关系结束"),
    "吸引力与自我评价": spec("外貌自评|社交自评", "担心自己的外貌没有吸引力|见喜欢的人前认真整理头发|怀疑自己不擅长聊天|面对心动对象突然紧张|读不懂别人释放的好感|想知道有没有人喜欢自己|把异性缘归因于颜值|学习更坦诚地表达感受|不再靠恋爱证明个人价值|相信自己也值得被选择", "|".join(COMMON_CONTEXTS), 4, 1, "魅力", "自我评价"),
    "婚姻": spec("婚姻仪式|长期计划", "收到朋友发来的婚礼请柬|看见恋人交换订婚戒指|讨论以后在哪座城市生活|考虑要不要带对象见父母|从恋爱走向共同生活|规划两个人的长期未来|被家人追问什么时候结婚|参加同学婚礼独自坐一桌|听见别人叫彼此老公老婆|还没有对象就开始被催婚", "|".join(COMMON_CONTEXTS), 4, 2, "婚姻", "长期关系"),
    "节日": spec("恋爱节日|单身节日", "在情人节准备一束玫瑰|七夕预订双人晚餐|在520发出含蓄告白|圣诞夜邀请喜欢的人见面|为恋爱纪念日挑选礼物|零点发送周年祝福|看见街边摆满情侣商品|独自度过恋爱主题节日|收到平台推送的情侣套餐|跨年倒数时想起喜欢的人", "|".join(COMMON_CONTEXTS), 4, 2, "节日", "纪念日"),
    "网络恋爱语言": spec("网络关系|线上互动", "把心动对象称作crush|在朋友圈正式官宣关系|和朋友一起磕一对CP|判断彼此是否处于暧昧期|讨论situationship算什么关系|换上成套的情侣头像|每天给对方分享短视频|把重要聊天设置为置顶|从线上聊天走到线下见面|因为断联猜测关系是否结束", "|".join(COMMON_CONTEXTS), 4, 2, "网络用语", "线上关系"),
    "恋爱统计与提问": spec("经历提问|状态追问", "被问以前谈过几个对象|回答自己有没有恋爱经历|被追问为什么一直单身|听见什么时候脱单的问题|解释目前有没有喜欢的人|回答第一次恋爱是在几岁|被问最长一段关系多久|猜测今年是否能够脱单|面对她是不是你女朋友的追问|拒绝回答过度私人的感情问题", "|".join(COMMON_CONTEXTS), 5, 1, "提问", "恋爱统计"),
    "间接语义触发词": spec("场景联想|物件联想", "看见电影院里的双人座|路过摆满玫瑰的花店|发现饮品店推出情侣套餐|看见两件并排挂着的外套|注意到共享耳机的两个人|经过夜晚亮灯的摩天轮|看到桌上放着两张电影票|发现相册里只有单人照片|收到适合约会的地点推荐|看见副驾驶放着一束花", "|".join(COMMON_CONTEXTS), 3, 3, "间接场景", "物件联想"),
    "高阶语义关联": spec("人生叙事|归属体验", "把恋爱视作人生进度|担心青春缺少重要体验|觉得自己落后于同龄人|害怕永远错过亲密关系|希望成为某个人的例外|渴望获得稳定情感回应|学习不靠伴侣定义完整|接受每个人节奏不同|允许遗憾留在过去|相信未来仍会建立连接", "|".join(COMMON_CONTEXTS), 4, 2, "人生阶段", "青春遗憾"),
    "荒诞扩散区": spec("技术双关|跨域联想", "面向对象时找不到对象|对象存储突然容量不足|单例模式坚持保持单身|配对算法开始安排相亲|关系数据库认真维护关系|握手协议负责成功牵手|线程脱单后进入双线程|空对象拒绝返回女朋友|恋爱脑被当作神经网络|朋友圈在图论里形成闭环", "|".join(COMMON_CONTEXTS), 1, 4, "技术双关", "荒诞联想", humor=True),
}


ALL_SPECS = {**CAMPUS_SPECS, **GENERAL_SPECS}

CAMPUS_CATEGORIES = set(CAMPUS_SPECS)
TARGET_COUNTS = {
    category: (800 if category in CAMPUS_CATEGORIES else (256 if category == "荒诞扩散区" else 259))
    for category in ALL_SPECS
}
assert sum(TARGET_COUNTS.values()) == 10_000


NEW_SEED_TERMS = {
    "校园外貌与穿搭": "刘海|校服|小皮筋|白衬衫|马尾|发卡|校服衣角|帆布鞋|黑框眼镜|运动服",
    "校园物件与暗号": "小纸条|情书|便利贴|同学录|毕业册|书签|第二颗纽扣|借书卡|草稿纸|课程表",
    "校园空间与路线": "靠窗座位|教室走廊|楼梯转角|操场看台|学校天台|图书馆书架|校园小卖部|放学路|校门口|公交站",
    "校园日常互动": "借笔记|讲题|一起值日|帮忙留座|递一瓶水|分享耳机|替对方带饭|课间聊天|互相对答案|一起搬书",
    "季节与感官记忆": "蝉鸣|下课铃|晚霞|晨光|雨后跑道|风扇声|粉笔灰|新书油墨味|洗衣粉香|冬日热水杯",
    "毕业与离别": "毕业照|毕业快乐|散伙饭|最后一课|毕业签名|离校广播|高考结束|录取通知书|各奔东西|重回母校",
}


def iter_new_seeds() -> Iterator[dict]:
    for category, packed_terms in NEW_SEED_TERMS.items():
        entry = ALL_SPECS[category]
        subcategory = entry["subcategories"][0]
        for term in split(packed_terms):
            yield {
                "term": term,
                "category": category,
                "subcategory": subcategory,
                "trigger_level": entry["trigger_level"],
                "semantic_distance": entry["semantic_distance"],
                "humor": False,
                "reason": f"可独立触发“{category}”联想的高精度校园意象种子",
                "tags": [category, subcategory, "高精度种子"],
            }


def iter_expansion_candidates(category: str) -> Iterator[dict]:
    entry = ALL_SPECS[category]
    concepts = entry["concepts"]
    subcategories = entry["subcategories"]
    relations = CAMPUS_RELATIONS if category in CAMPUS_CATEGORIES else GENERAL_RELATIONS
    for relation, pattern in relations:
        for context in entry["contexts"]:
            for concept_index, concept in enumerate(concepts):
                subcategory_index = min(
                    len(subcategories) - 1,
                    concept_index * len(subcategories) // len(concepts),
                )
                subcategory = subcategories[subcategory_index]
                level = entry["trigger_level"]
                distance = entry["semantic_distance"]
                if relation == "回忆联想":
                    distance = min(4, distance + 1)
                    level = max(1, level - 1)
                elif relation in {"社会比较", "他人触发", "缺席体验"}:
                    distance = min(4, max(2, distance))
                    level = max(2, level - 1) if not entry["humor"] else 1
                yield {
                    "term": pattern.format(context=context, concept=concept),
                    "category": category,
                    "subcategory": subcategory,
                    "trigger_level": level,
                    "semantic_distance": distance,
                    "humor": entry["humor"],
                    "reason": f"通过“{relation}”关系连接“{subcategory}”与{category}主题",
                    "tags": list(dict.fromkeys([category, subcategory, relation, *entry["tags"]])),
                }
