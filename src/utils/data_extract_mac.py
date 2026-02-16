import json
import re

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def process_moegirl(data_dir, output_path):
    with open(data_dir, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in tqdm(f)]
    samples = []
    for line in tqdm(lines, desc='Processing moegirl articles'):
        if not line['title'].startswith('[') or line['title'].find(']') == -1:
            continue
        if line['text'].startswith('#重定向 [['):
            continue
        line['title'] = line['title'][line['title'].find(']') + 1:]
        sample = {'text': f'URL: https://zh.moegirl.org.cn/{line["title"].replace(" ", "_")}' + '\n\n' + line['title'] + '\n\n' + line['text']}
        samples.append(sample)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(samples, desc='Writing to file'):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def process_zhihu(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in tqdm(f)]
    samples = []
    for line in tqdm(lines, desc='Processing zhihu articles'):
        meta_data = json.loads(line['METADATA'])
        url = meta_data['url']
        title = line['INSTRUCTION']
        text = line['RESPONSE']
        sample = {'text': f'URL: {url}' + '\n\n' + title + '\n\n' + text}
        samples.append(sample)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(samples, desc='Writing to file'):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def process_novels(input_path, output_path, max_len, min_len):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in tqdm(f)]
    samples = []
    tokenizer = AutoTokenizer.from_pretrained('../../models/Qwen3-8B')
    fiter_words = ['电子书搜索下载', '录入：', '24小时内删除', '公众号', 'QQ', '微信', '如果你不知道读什么书', '免费电子书', '多看阅读器', '资源分享', '幻剑书盟', '轻之国度']
    for line in tqdm(lines, desc='Processing novels'):
        if line['words'] < 4096:
            continue
        title = line['title']
        texts = line['text']
        curr_length = len(tokenizer.encode(title + '\n\n'))
        curr_splits = []
        for text in texts:
            if len(text) < 1024 and any(word in text for word in fiter_words):
                continue
            splits = text.split('\n')
            for split in splits:
                if any(word in split for word in fiter_words):
                    # 如果存在章节名，去除前面的字符
                    pattern = r'第[零一二三四五六七八九十]+章'
                    res = re.search(pattern, split)
                    if res:
                        split = split[res.start():]
                    else:
                        continue
                encoded = tokenizer.encode(split)
                if curr_splits and curr_length + len(encoded) > max_len:
                    sample = {'text': title + '\n\n' + '\n'.join(curr_splits).strip()}
                    samples.append(sample)
                    curr_length = len(tokenizer.encode(title + '\n\n'))
                    curr_splits = []
                curr_length += len(encoded)
                if len(encoded) > 0:
                    curr_splits.append(split)
            if curr_length >= min_len:
                sample = {'text': title + '\n\n' + '\n'.join(curr_splits).strip()}
                samples.append(sample)
                curr_length = len(tokenizer.encode(title + '\n\n'))
                curr_splits = []
            elif curr_splits:
                curr_splits.append('')
        if curr_splits:
            sample = {'text': title + '\n\n' + '\n'.join(curr_splits).strip()}
            samples.append(sample)
    with  open(output_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(samples, desc='Writing to file'):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def concatenate_files(input_paths, output_path):
    lines = []
    for input_path in input_paths:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines.extend([json.loads(line) for line in tqdm(f)])
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in tqdm(lines, desc='Writing to file'):
            sample = {'text': line['text']}
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

def process_test(input_path, output_path) -> None:
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    output_lines = []
    current_block = {}
    in_question = False
    in_options = False

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith("题目："):
            # 保存前一个题目
            if current_block:
                if 'options' not in current_block:
                    current_block['options'] = []
                output_lines.append(current_block)

            # 开始新题目
            current_block = {
                "question": line[3:].strip(),
                "options": [],
                "answer": ""
            }
            in_question = True
            in_options = False

        elif line == "选项：":
            in_options = True

        elif line.startswith("答案："):
            current_block["answer"] = line[3:].strip()
            in_options = False

        else:
            if in_question:
                if in_options:
                    current_block["options"].append(line)
                else:
                    # 处理没有"选项："标记的情况
                    if not current_block["options"]:
                        current_block["options"] = []
                    if not line.startswith("答案："):  # 避免答案行被错误加入
                        current_block["options"].append(line)

    # 添加最后一个题目
    if current_block:
        if 'options' not in current_block:
            current_block['options'] = []
        output_lines.append(current_block)

    for idx in range(2, len(output_lines)):
        prompt = '\n\n'.join([line['question'] + '\n' + '\n'.join(line['options']) + '\n' + f'答案：{line["answer"]}' for line in output_lines[:2]])
        output_lines[idx] = {'text': prompt + '\n\n' + output_lines[idx]['question'] + '\n' + '\n'.join(output_lines[idx]['options']) + '\n' + '答案：',
                             'label': output_lines[idx]['answer']}
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in output_lines[2:]:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def compute_score(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in tqdm(f)]
    score = 0
    for line in tqdm(lines):
        predict = line['predict'].split('\n')[0].replace('《','').replace('》','').replace(' ','').replace('　','').replace('·', "").strip()
        label = line['label']
        if predict == label:
            score += 1
    print(f'Accuracy: {score / len(lines):.4f}')

if __name__ == '__main__':
    # process_zhihu('../data/zhihu/raw/zhihu.jsonl', '../data/zhihu/cleaned/zhihu.jsonl')
    # process_novels('../data/novels/filtered/ja.jsonl', '../data/novels/cleaned/ja.jsonl', 4096, 512)
    # concatenate_files(
    #     [
    #         '../data/novels/cleaned/ja.jsonl',
    #         '../data/novels/cleaned/en.jsonl',
    #         '../data/novels/cleaned/zh.jsonl',
    #         '../data/bilibili/cleaned/bilibili.jsonl',
    #         '../data/moegirl/cleaned/moegirl.jsonl',
    #         '../data/zhihu/cleaned/zhihu.jsonl',],
    #     '../data/250607.jsonl'
    # )
    # process_test('../data/test/test.txt', '../data/test/test.jsonl')
    compute_score('../data/test/base-preds-saimple.jsonl')
    compute_score('../data/test/kawaii-preds-simple.jsonl')
