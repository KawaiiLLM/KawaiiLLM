import asyncio
import collections
import csv
import json
import os
import re
import shutil
from typing import List

import jieba
import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from jieba.posseg import re_num
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from janome.tokenizer import Tokenizer
from langdetect import detect

# 初始化各语言处理器
jieba.initialize()
nltk_stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()
janome_tokenizer = Tokenizer()


import regex

def count_novel_words(text):
    # 统计英文单词（使用拉丁字母的单词）
    en_pattern = r'\b\p{Script=Latin}+\b'
    en_words = regex.findall(en_pattern, text, regex.IGNORECASE)
    en_word_count = len(en_words)

    # 移除已匹配的英文单词
    remaining_text = regex.sub(en_pattern, '', text, flags=regex.IGNORECASE)

    # 统计中日文字符（汉字、平假名、片假名），排除标点和空格
    cjk_pattern = r'[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]'
    cjk_chars = regex.findall(cjk_pattern, remaining_text)
    cjk_char_count = len(cjk_chars)

    # 总字数为英文单词数 + 中日文字符数
    total = en_word_count + cjk_char_count
    return total


def language_detect(text):
    """语言检测函数"""
    try:
        lang = detect(text)
    except:
        lang = 'en'  # 默认英语
    return lang

def process_ja(text):
    tokens = janome_tokenizer.tokenize(text)
    return [token.base_form or token.surface for token in tokens
            if token.part_of_speech.split(',')[0] not in ['助詞', '助動詞', '記号']]

def process_zh(text):
    words = jieba.cut(text)
    return [w for w in words if w.strip()]

def process_en(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    return [stemmer.stem(w) for w in words if w not in nltk_stopwords]

async def multilingual_preprocess(text, lang=None, executor=None):
    """多语言预处理管道"""
    loop = asyncio.get_event_loop()

    if lang is None:
        lang = language_detect(text)

    # 按语言处理
    if lang in ['zh', 'zh-cn', 'zh-tw']:
        return await loop.run_in_executor(
            executor,
            partial(process_zh, text)
        )

    elif lang == 'ja':
        return await loop.run_in_executor(
            executor,
            partial(process_ja, text)
        )

    else:  # 英语及其他语言
        return await loop.run_in_executor(
            executor,
            partial(process_en, text)
        )

def create_minhash(features, num_perm):
    """生成MinHash签名"""
    m = MinHash(num_perm=num_perm)
    for word in features:
        m.update(word.encode('utf-8'))
    return m


async def find_duplicates(folder_path, lang=None, batch_size=8, num_perm=128, threshold=0.85):
    """主处理函数"""
    # 读取文件
    filename_to_feature = {}
    filenames = []
    task_list = []
    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        for filename in tqdm(os.listdir(folder_path), desc="Processing files", unit="file"):
            if filename.endswith('.txt'):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    filenames.append(filename)
                    task_list.append(multilingual_preprocess(text, lang, executor))
                    if len(task_list) < batch_size:
                        continue
                    else:
                        features = await asyncio.gather(*task_list)
                        for name, feature in zip(filenames, features):
                            filename_to_feature[name] = feature
                        filenames = []
                        task_list = []

    # 创建LSH索引
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}

    # 生成MinHash并插入索引
    for doc_id, features in tqdm(filename_to_feature.items(), desc="Inserting MinHash", unit="file"):
        mh = create_minhash(features, num_perm)
        minhashes[doc_id] = mh
        lsh.insert(doc_id, mh)

    # 查询相似文档
    duplicates = []
    for doc_id in tqdm(filename_to_feature.keys(), desc="Querying LSH", unit="file"):
        duplicates.append({doc_id: []})
        # LSH查询
        result = lsh.query(minhashes[doc_id])
        similar_docs = [x for x in result if x != doc_id]

        if similar_docs:
            # 精确验证（可选BM25计算）
            main_doc = set(filename_to_feature[doc_id])
            for other_id in similar_docs:
                main_set = set(main_doc)
                other_set = set(filename_to_feature[other_id])

                union = main_set | other_set
                if not union:  # 两文档均为空
                    similarity = 1.0  # 定义空文档为100%相似
                else:
                    intersection = main_set & other_set
                    similarity = len(intersection) / len(union)
                if similarity >= threshold:
                    duplicates[-1][doc_id].append((other_id, round(similarity, 2)))
    return duplicates

# BM25实现（用于二次验证）
class BM25:
    def __init__(self, corpus):
        self.corpus = corpus
        self.doc_lengths = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_lengths) / len(corpus)
        self.k1 = 1.5
        self.b = 0.75

    def _calc_idf(self, term):
        n_qi = sum(1 for doc in self.corpus if term in doc)
        return math.log((len(self.corpus) - n_qi + 0.5) / (n_qi + 0.5) + 1)

    def similarity(self, doc_a, doc_b):
        # 转换为BM25相关性评分
        common_terms = set(doc_a) & set(doc_b)
        score = 0
        for term in common_terms:
            idf = self._calc_idf(term)
            tf_a = doc_a.count(term)
            tf_b = doc_b.count(term)
            score += idf * (tf_a * (self.k1 + 1) / (tf_a + self.k1 * (1 - self.b + self.b * len(doc_a)/self.avgdl))) * \
                     (tf_b * (self.k1 + 1) / (tf_b + self.k1 * (1 - self.b + self.b * len(doc_b)/self.avgdl)))
        return score

def save_results(unique_files, output_file):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in unique_files:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def score_text(texts: List[str]) -> List[float]:
    positive_targets = {'ISBN', '©', '(C)', '(c)'}
    scores = [0] * len(texts)
    for index, text in enumerate(texts):
        for target in positive_targets:
            scores[index] += (text.find(target) != -1)
    # newline_counts = [text.count('\n') for text in texts]
    # for index, count in enumerate(newline_counts):
    #     scores[index] += (count - min(newline_counts)) / (max(newline_counts) // 4)
    words_counts = [count_novel_words(text) for text in texts]
    for index, words in enumerate(words_counts):
        if words == 0:
            continue
        scores[index] += (words) / (max(words_counts) // 4)
    return scores

def filter_duplicates(duplicate_file_path, index_path, folder_path, output_path, threshold):
    """过滤重复文件"""
    f = open(index_path, 'r', encoding='utf-8')
    reader = csv.DictReader(f)
    total_count = sum(1 for _ in reader)
    f.seek(0)  # Reset the file pointer to the beginning after counting
    progress_bar = tqdm(reader, total=total_count, desc="Processing rows", unit="row")
    next(reader)
    filename_to_title = {}
    for row in progress_bar:
        filename_to_title[row['zlibrary_id']] = row['title']

    with open(duplicate_file_path, 'r', encoding='utf-8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
    reserved_filenames = set()
    removed_filenames = set()
    data = []
    for line in tqdm(lines):
        filename, similar_files = list(line.keys())[0], list(line.values())[0]
        if filename in reserved_filenames or filename in removed_filenames:
            continue
        to_select_filenames = [filename]
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            to_select_text = [text]
        for similar_file in similar_files:
            filename, similarity = similar_file
            if similarity < threshold:
                continue
            if filename in reserved_filenames or filename in removed_filenames:
                continue
            to_select_filenames.append(filename)
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                to_select_text.append(text)
        if len(to_select_filenames) > 1:
            scores = score_text(to_select_text)
            selected_index = scores.index(max(scores))
        else:
            selected_index = 0

        for index, filename in enumerate(to_select_filenames):
            if index == selected_index:
                reserved_filenames.add(filename)
            else:
                removed_filenames.add(filename)
        text = to_select_text[selected_index].replace("这个div不要改动\n", "")

        # Determine language
        sample_text = text[len(text)//2: len(text)//2+2000] if len(text) > 2000 else text
        lang = language_detect(sample_text)

        # Fix common misclassification: Chinese detected as Korean
        if lang == 'ko':
            # Check for Hangul in the sample text
            if not regex.search(r'\p{IsHangul}', sample_text):
                lang = 'zh'

        # Prepare output data
        text_split = text.split('[PAGE_SEP]')
        texts_dict = {}
        total_words = 0

        for i, page_text in enumerate(text_split):
            page_words = count_novel_words(page_text)
            total_words += page_words
            texts_dict[str(i)] = {
                "words": page_words,
                "text": page_text
            }

        output_data = {
            "meta": {
                "id": to_select_filenames[selected_index].removesuffix('.txt'),
                "title": filename_to_title.get(to_select_filenames[selected_index].removesuffix('.txt'), "Unknown"),
                "language": lang,
                "source_filename": to_select_filenames[selected_index],
                "total_words": total_words,
                "total_pages": len(text_split)
            },
            "texts": texts_dict
        }

        # Create output directory for the language
        lang_dir = os.path.join(output_path, lang)
        os.makedirs(lang_dir, exist_ok=True)

        # Write to individual JSON file
        output_filename = to_select_filenames[selected_index].replace('.txt', '.json')
        output_file_path = os.path.join(lang_dir, output_filename)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Original files: {len(lines)}")
    print(f"Reserved files: {len(reserved_filenames)}")
    print(f"Removed files: {len(removed_filenames)}")


async def main():
    # for filename in tqdm(os.listdir('../data/novels/raw/txt')):
    #     if filename.endswith('.txt'):
    #         # if filename != '21215649.txt':
    #         #     continue
    #         with open(os.path.join('../data/novels/raw/txt', filename), 'r', encoding='utf-8') as f:
    #             text = f.read()
    #             if len(text) > 2000:
    #                 text = text[len(text)//2:][:2000]
    #             text = text.replace('[PAGE_SEP]', '')
    #             lang = language_detect(text)
    #         if lang == 'ja':
    #             shutil.move(os.path.join('../data/novels/raw/txt', filename), os.path.join('../data/novels/raw/txt/ja', filename))
    #         elif lang in ['zh', 'zh-cn', 'zh-tw']:
    #             shutil.move(os.path.join('../data/novels/raw/txt', filename), os.path.join('../data/novels/raw/txt/zh', filename))
    #         elif lang == 'en':
    #             shutil.move(os.path.join('../data/novels/raw/txt', filename), os.path.join('../data/novels/raw/txt/en', filename))
    #         else:
    #             print(f"Unknown language: {lang}, text: {text[:100]}")
    #             shutil.move(os.path.join('../data/novels/raw/txt', filename), os.path.join('../data/novels/raw/txt/zh', filename))

    # for filename in tqdm(os.listdir('../data/novels/raw/txt/ja')):
    #     if filename.endswith('.txt') and os.path.exists(os.path.join('../data/novels/raw/txt', filename)):
    #         try:
    #             os.remove(os.path.join('../data/novels/raw/txt/ja', filename))
    #             shutil.copyfile(os.path.join('../data/novels/raw/txt', filename),
    #                             os.path.join('../data/novels/raw/txt/ja', filename))
    #         except  Exception as e:
    #             print(f"Error for processing file {filename}: {e}")



    # for lang in ['ja', 'en']:
    #     duplicate_files = await find_duplicates(f"../data/novels/raw/txt/{lang}", lang, 16, 128, 0.5)
    #     save_results(duplicate_files, f"../data/novels/similar_novels_{lang}.txt")

    languages = ['en', 'ja', 'zh']
    for lang in languages:
        similar_novels_path = f"../data/novels/similar_novels_{lang}.txt"
        raw_txt_path = f"../data/novels/raw/txt/{lang}"

        if os.path.exists(similar_novels_path) and os.path.exists(raw_txt_path):
            print(f"Processing {lang}...")
            filter_duplicates(similar_novels_path,
                              '../data/novels/raw/selected_index.txt',
                              raw_txt_path,
                              "../data/novels/deduped/novels",
                              0.5)
        else:
            print(f"Skipping {lang}: Input files not found.")


if __name__ == "__main__":
    asyncio.run(main())