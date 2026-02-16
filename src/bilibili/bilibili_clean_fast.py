"""
Bilibili 数据清洗脚本 (高速版)
使用 orjson + 多进程并行处理
"""

import argparse
import html
import os
import re
from typing import Dict, Any, List
from multiprocessing import Pool, cpu_count
from functools import partial
import time

try:
    import orjson
    def json_loads(s):
        return orjson.loads(s)
    def json_dumps(obj):
        return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE).decode('utf-8')
except ImportError:
    import json
    def json_loads(s):
        return json.loads(s)
    def json_dumps(obj):
        return json.dumps(obj, ensure_ascii=False) + '\n'

# ============== 配置 ==============

CATEGORY_WHITELIST = {
    '动漫杂谈', '动画', '漫画', '轻小说', '游戏',
    '手游', '单机游戏', '网络游戏', 'Vtuber',
    '综合', '影视杂谈', '番剧', 'MAD·AMV',
    '国产动画', '新番导视', '旧番补档', '动画综合',
    '单机联机', 'MUGEN', '电子竞技', 'GMV', 'SS',
}

ACGN_KEYWORDS = {
    '番剧', '动漫', '动画', '漫画', '轻小说', 'galgame', '视觉小说',
    '二次元', 'acg', 'acgn', 'ova', 'oad', '剧场版',
    '特摄', '假面骑士', '奥特曼', '战队',
    'bilibili', 'b站', '京阿尼', '骨头社', 'ufotable', 'mappa', 'a1',
    'key社', 'type-moon', 'fate', '原神', '崩坏', '明日方舟', '米哈游',
    '任天堂', '索尼', '微软', 'steam', 'epic', 'ubisoft',
    '新番', '补番', '追番', '入坑', '安利', '神作', '萌', '燃', '治愈', '致郁',
    '声优', 'cv', '作画', '演出', '脚本', '监督', '分镜',
    '角色', '人设', 'cp', '本命', '推', 'waifu', '老公', '老婆',
    'cos', 'cosplay', '同人', '圣地巡礼', '痛车', '手办', '模型',
    '鬼灭', '咒术', '进击的巨人', '间谍过家家', '电锯人',
    '孤独摇滚', 're0', '无职转生', '辉夜', '五等分',
    '刀剑神域', '魔法禁书目录', '某科学的超电磁炮', '约会大作战',
    '初音未来', '洛天依', 'v家', '东方project', '舰c', '碧蓝航线',
}

BLOCK_KEYWORDS = {'流量卡', '壁纸', '动漫壁纸', '美图'}

# ============== 硬性最低阈值 (基于3333条标注数据优化) ==============
# 策略: Low Threshold (Words>=300 & View>=20)
# 过滤率: 15.6%, 精确率: 63.6%, 召回率: 94.2%, F1: 76.0%
ARTICLE_MIN_WORDS = 300   # 字数 >= 300
ARTICLE_MIN_VIEW = 20     # 阅读量 >= 20
ARTICLE_MIN_LIKE = 0      # 点赞 >= 0 (不限制)
ARTICLE_MAX_NEWLINE_RATIO = 0.5

# 预编译正则表达式
RE_IMG_ALT = re.compile(r'<img[^>]*alt=["\']([^"\']*)["\'][^>]*>')
RE_IMG = re.compile(r'<img[^>]*>')
RE_LINK = re.compile(r'<a[^>]*>([^<]*)</a>')
RE_TAG = re.compile(r'<[^>]+>')
RE_NEWLINES = re.compile(r'\n{3,}')
RE_SPACES = re.compile(r' {2,}')

def clean_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = RE_IMG_ALT.sub(r'\1', text)
    text = RE_IMG.sub('', text)
    text = RE_LINK.sub(r'\1', text)
    text = RE_TAG.sub('', text)
    text = RE_NEWLINES.sub('\n\n', text)
    text = RE_SPACES.sub(' ', text)
    return text.strip()

def parse_tags(tags_str: str) -> List[str]:
    if not tags_str:
        return []
    try:
        tags = json_loads(tags_str)
        if isinstance(tags, list):
            return [t.get('name', t) if isinstance(t, dict) else str(t) for t in tags]
    except:
        pass
    return []

def is_acgn_content(title: str, tags: List[str], category: str) -> bool:
    if category in CATEGORY_WHITELIST:
        return True
    text_to_check = (title + ' ' + ' '.join(tags)).lower()
    for keyword in ACGN_KEYWORDS:
        if keyword.lower() in text_to_check:
            return True
    return False

def is_spam_content(title: str, content: str) -> bool:
    text_to_check = (title + ' ' + content[:500]).lower()
    for keyword in BLOCK_KEYWORDS:
        if keyword in text_to_check:
            return True
    return False

def process_line(line: str, min_words: int, min_view: int, min_like: int, require_acgn: bool) -> tuple:
    """处理单行，返回 (result_json_or_none, filter_reason)"""
    line = line.strip()
    if not line:
        return None, 'empty'

    try:
        item = json_loads(line)
    except:
        return None, 'json_error'

    content = item.get('content', '')
    words = item.get('words', 0)
    title = item.get('title', '')
    category = item.get('category_name', '')
    tags = parse_tags(item.get('tags', ''))

    like = item.get('like', 0) or 0
    view = item.get('view', 0) or 0

    # 1. 字数过滤 (words >= 300)
    if words < min_words:
        return None, 'short'

    # 2. 阅读量过滤 (view >= 20)
    if view < min_view:
        return None, 'low_view'

    # 3. 点赞过滤 (like >= 0)
    if like < min_like:
        return None, 'low_like'

    # 4. ACGN 相关性
    if require_acgn and not is_acgn_content(title, tags, category):
        return None, 'non_acgn'

    # 5. 垃圾/广告
    if is_spam_content(title, content):
        return None, 'spam'

    # 清洗内容
    clean_content = clean_html(content)
    clean_len = len(clean_content)

    # 6. 清洗后长度
    if clean_len < min_words // 2:
        return None, 'short_after_clean'

    # 7. 换行符密度
    if clean_len > 0 and clean_content.count('\n') / clean_len > ARTICLE_MAX_NEWLINE_RATIO:
        return None, 'high_newline'

    # 更新内容
    item['content'] = clean_content
    item['words'] = clean_len

    return json_dumps(item), 'passed'

def process_chunk(chunk: List[str], min_words: int, min_view: int, min_like: int, require_acgn: bool) -> Dict:
    """处理一批行"""
    results = []
    stats = {'total': 0, 'passed': 0, 'short': 0, 'low_view': 0, 'low_like': 0, 'non_acgn': 0, 'spam': 0, 'other': 0, 'chars': 0}

    for line in chunk:
        stats['total'] += 1
        result, reason = process_line(line, min_words, min_view, min_like, require_acgn)

        if result:
            results.append(result)
            stats['passed'] += 1
            stats['chars'] += len(result)
        elif reason == 'short' or reason == 'short_after_clean':
            stats['short'] += 1
        elif reason == 'low_view':
            stats['low_view'] += 1
        elif reason == 'low_like':
            stats['low_like'] += 1
        elif reason == 'non_acgn':
            stats['non_acgn'] += 1
        elif reason == 'spam':
            stats['spam'] += 1
        else:
            stats['other'] += 1

    return {'results': results, 'stats': stats}

def main():
    parser = argparse.ArgumentParser(description='Bilibili 数据清洗工具 (高速版)')
    parser.add_argument('--input', type=str, default='data/bilibili/raw/merged_articles.jsonl')
    parser.add_argument('--output', type=str, default='data/bilibili/cleaned/articles_cleaned.jsonl')
    parser.add_argument('--min-words', type=int, default=ARTICLE_MIN_WORDS, help='最小字数 (默认: 300)')
    parser.add_argument('--min-view', type=int, default=ARTICLE_MIN_VIEW, help='最小阅读量 (默认: 20)')
    parser.add_argument('--min-like', type=int, default=ARTICLE_MIN_LIKE, help='最小点赞数 (默认: 0)')
    parser.add_argument('--acgn-filter', action='store_true', default=False)
    parser.add_argument('--workers', type=int, default=0, help='Worker processes (0=auto)')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Lines per chunk')
    args = parser.parse_args()

    num_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    print(f"使用 {num_workers} 个进程并行处理")
    print(f"每批处理 {args.chunk_size} 行")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"硬性阈值: words>={args.min_words}, view>={args.min_view}, like>={args.min_like}")
    print("=" * 50)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    total_stats = {'total': 0, 'passed': 0, 'short': 0, 'low_view': 0, 'low_like': 0, 'non_acgn': 0, 'spam': 0, 'other': 0, 'chars': 0}

    process_fn = partial(process_chunk,
                         min_words=args.min_words,
                         min_view=args.min_view,
                         min_like=args.min_like,
                         require_acgn=args.acgn_filter)

    start_time = time.time()

    with open(args.input, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(args.output, 'w', encoding='utf-8') as f_out, \
         Pool(num_workers) as pool:

        chunk = []
        chunks_to_process = []

        for line in f_in:
            chunk.append(line)
            if len(chunk) >= args.chunk_size:
                chunks_to_process.append(chunk)
                chunk = []

                # 每积累 num_workers 个 chunk 就并行处理
                if len(chunks_to_process) >= num_workers:
                    results = pool.map(process_fn, chunks_to_process)
                    for r in results:
                        for key in total_stats:
                            total_stats[key] += r['stats'][key]
                        for result_line in r['results']:
                            f_out.write(result_line)

                    elapsed = time.time() - start_time
                    speed = total_stats['total'] / elapsed if elapsed > 0 else 0
                    print(f"\r已处理: {total_stats['total']:,} | 通过: {total_stats['passed']:,} | 速度: {speed:.0f} 行/秒", end='', flush=True)
                    chunks_to_process = []

        # 处理剩余
        if chunk:
            chunks_to_process.append(chunk)
        if chunks_to_process:
            results = pool.map(process_fn, chunks_to_process)
            for r in results:
                for key in total_stats:
                    total_stats[key] += r['stats'][key]
                for result_line in r['results']:
                    f_out.write(result_line)

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"处理完成! 耗时: {elapsed:.1f} 秒")
    print(f"  总计: {total_stats['total']:,}")
    print(f"  通过: {total_stats['passed']:,} ({total_stats['passed']/total_stats['total']*100:.1f}%)" if total_stats['total'] > 0 else "  通过: 0")
    print(f"  过滤-过短: {total_stats['short']:,}")
    print(f"  过滤-低阅读量: {total_stats['low_view']:,}")
    print(f"  过滤-无点赞: {total_stats['low_like']:,}")
    print(f"  过滤-非ACGN: {total_stats['non_acgn']:,}")
    print(f"  过滤-垃圾: {total_stats['spam']:,}")
    print(f"  总字符数: {total_stats['chars']:,} ({total_stats['chars']/1024/1024:.1f} MB)")
    print("=" * 50)

if __name__ == '__main__':
    main()
