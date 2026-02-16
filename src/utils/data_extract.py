import asyncio
import csv
import os
import re
import shutil
import html
from functools import partial
from typing import Optional
import json

import pymysql
import ebooklib
import pandas as pd
import chardet
import regex
from ebooklib import epub
from bs4 import BeautifulSoup, Tag
from nltk.corpus import words
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import os
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET

def fix_epub_xmlns(epub_path, output_path):
    """
    修复 EPUB 文件中 OPF 的 xmlns 属性拼写错误
    :param epub_path: 输入 EPUB 文件路径
    :param output_path: 修复后的 EPUB 保存路径
    """
    # 创建临时工作目录
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 解压 EPUB 到临时目录
        with zipfile.ZipFile(epub_path, 'r') as zf:
            zf.extractall(tmp_dir)

        # 查找 OPF 文件（示例路径为 OPS/fb.opf，按实际调整）
        opf_path = os.path.join(tmp_dir, 'OPS', 'fb.opf')
        if not os.path.exists(opf_path):
            raise FileNotFoundError("OPF file not found in EPUB package")

        # 解析并修复 XML
        tree = ET.parse(opf_path)
        root = tree.getroot()

        # 检查并替换错误属性
        if 'mlns' in root.attrib:
            root.attrib['xmlns'] = root.attrib['mlns']
            del root.attrib['mlns']
            print(f"Fixed xmlns attribute in {opf_path}")
        else:
            print("No xmlns typo found, file is correct")

        # 保存修复后的 OPF
        tree.write(opf_path, encoding='utf-8', xml_declaration=True)

        # 重新打包为 EPUB
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as new_zf:
            for root_dir, _, files in os.walk(tmp_dir):
                for file in files:
                    file_path = os.path.join(root_dir, file)
                    arcname = os.path.relpath(file_path, tmp_dir)
                    new_zf.write(file_path, arcname)

    print(f"Repaired EPUB saved to: {output_path}")

def extract_text_from_epub(epub_file_path) -> Optional[str]:
    # 打开EPUB文件
    try:
        book = epub.read_epub(epub_file_path, {"ignore_ncx": True})
    except AttributeError as e:
        try:
            print(f"Repairing EPUB file...")
            filename = epub_file_path.split("\\")[-1]
            output_path = f'../data/tmp/{filename}.epub'
            fix_epub_xmlns(epub_file_path, output_path)
            book = epub.read_epub(output_path, {"ignore_ncx": True})
        except  Exception as e:
            print(f"Error repairing EPUB file {epub_file_path}: {e}")
            return None
    except Exception as e:
        print(f"Error reading EPUB file {epub_file_path}: {e}")
        return None
    all_text = ''

    # 按 spine 顺序获取内容项
    spine_items = [book.get_item_with_id(item_id[0]) for item_id in book.spine]

    for item in spine_items:
        if item is None:
            continue
        try:
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            content = item.get_content()
            if content is None:
                print(f"Warning: Empty content for item {item.get_id()}")
                continue

            # 检测编码并解码
            detected_enc = chardet.detect(content)['encoding']
            html_content = content.decode(detected_enc or 'utf-8', errors='replace')

            # 提取文本

            soup = BeautifulSoup(html_content, 'xml')
            # 检查是否存在解析错误
            if soup.find('parsererror'):
                # 存在错误，改用HTML解析器
                soup = BeautifulSoup(html_content, 'lxml')
            # text = soup.get_text(strip=False)
            body = soup.find('body')
            while len(body.find_all(recursive=False)) == 1:
                body = body.find_all(recursive=False)[0]

            # 提取所有直接子标签的文本（过滤空文本）
            BLOCK_TAGS = {
                'br',
                'p', 'div', 'h1', 'h2', 'h3',
                'pre', 'blockquote', 'section',
                'li', 'dt', 'dd'  # 常见于对话列表
            }
            def extract_clean_text(element, strip=False) -> str:
                text_parts = []
                for child in element.children:
                    if isinstance(child, Tag):
                        # 过滤干扰标签
                        if child.name in {'script', 'style', 'rt', 'rp'}:
                            continue
                        # 处理Ruby标签
                        if child.name == 'ruby':
                            ruby_text = ''.join([rb.get_text(strip=False) for rb in child if rb.name not in ['rt', 'rp']])
                            text_parts.append(ruby_text.strip())
                        else:
                            # 递归处理子元素，标记是否为块级
                            child_is_block = child.name in BLOCK_TAGS
                            child_text = extract_clean_text(child, strip)
                            if child_is_block:
                                if len(text_parts) > 0:
                                    text_parts.append('\n')
                                text_parts.append(child_text)
                            else:
                                text_parts.append(child_text)
                    else:
                        text_parts.append(child.string.strip() if strip else child.string)

                if strip is False and '\n' not in text_parts:
                    return html.unescape('\n'.join(text_parts))
                else:
                    return html.unescape(''.join(text_parts))

            text = extract_clean_text(body, True)

            if text.strip() == '':
                text = ''

            if text:
                if all_text:
                    all_text += "\n[PAGE_SEP]\n" + text
                else:
                    all_text = text

        except Exception as e:
            print(f"Error processing {item.get_id()}: {e}")
    return all_text


def select_from_index(data_dir, index_path, output_dir):
    def get_group(idx):
        if 11860000 <= idx <= 11899999:
            return 'pilimi-zlib-11860000-11899999'
        elif 18610000 <= idx <= 18699999:
            return 'pilimi-zlib2-18610000-18699999'
        elif 19330000 <= idx <= 21079999:
            return 'pilimi-zlib2-19330000-21079999'
        elif 21080000 <= idx <= 21179999:
            return 'pilimi-zlib2-21080000-21179999'
        elif 21180000 <= idx <= 21229999:
            return 'pilimi-zlib2-21180000-21229999'
        elif 21230000 <= idx <= 21319999:
            return 'pilimi-zlib2-21230000-21319999'
        elif 21320000 <= idx <= 21399999:
            return 'pilimi-zlib2-21320000-21399999'
        elif 21490000 <= idx <= 21589999:
            return 'pilimi-zlib2-21490000-21589999'
        elif 22120000 <= idx <= 22199999:
            return 'pilimi-zlib2-22120000-22199999'
        else:
            return ''

    f = open(index_path, 'r', encoding='utf-8')
    reader = csv.DictReader(f)
    total_count = sum(1 for _ in reader)
    file_count = 0
    exist_count = 0
    f.seek(0)  # Reset the file pointer to the beginning after counting
    progress_bar = tqdm(reader, total=total_count, desc="Processing rows", unit="row")
    next(reader)
    for row in progress_bar:
        total_count += 1
        group = get_group(int(row['zlibrary_id']))
        if group:
            file_path = os.path.join(data_dir, group, row['zlibrary_id'])
            output_path = os.path.join(output_dir, row['zlibrary_id'])
            if os.path.exists(file_path):
                if os.path.exists(output_path):
                    exist_count += 1
                else:
                    shutil.copyfile(file_path, output_path)
                file_count += 1
    print(f'total count: {total_count}')
    print(f'file count: {file_count}')
    print(f'exist count: {exist_count}')

def delete_from_index(data_dir, index_path):
    f = open(index_path, 'r', encoding='utf-8')
    reader = csv.DictReader(f)
    total_count = sum(1 for _ in reader)
    file_count = 0
    f.seek(0)  # Reset the file pointer to the beginning after counting
    progress_bar = tqdm(reader, total=total_count, desc="Processing rows", unit="row")
    next(reader)
    found_set = set()
    for row in progress_bar:
        total_count += 1
        file_path = os.path.join(data_dir, row['zlibrary_id'])
        if os.path.exists(file_path):
            file_count += 1
            found_set.add(file_path)
    delete_count = 0
    for file in os.listdir(data_dir):
        if os.path.join(data_dir, file) not in found_set:
            os.remove(os.path.join(data_dir, file))
            delete_count += 1
    print(f'total count: {total_count}')
    print(f'file count: {file_count}')
    print(f'delete count: {delete_count}')


async def convert_epub_to_txt(data_dir, output_dir, batch_size=8):
    loop = asyncio.get_event_loop()
    output_paths = []
    task_list = []
    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        for file_name in tqdm(os.listdir(data_dir)):
            epub_file_path = os.path.join(data_dir, file_name)
            output_txt_file_path = os.path.join(output_dir, file_name + '.txt')
            # if os.path.exists(output_txt_file_path):
            # if file_name != "18617497":
            # if file_name != "21183502":
            # if file_name != "11861063":
            #     continue
            output_paths.append(output_txt_file_path)
            task_list.append(
                loop.run_in_executor(
                    executor,
                    partial(extract_text_from_epub, epub_file_path)
                )
            )
            if len(task_list) < batch_size:
                continue
            else:
                texts = await asyncio.gather(*task_list)
                for text, output_path in zip(texts, output_paths):
                    if text is not None:
                        with open(output_path, 'w', encoding='utf-8') as outfile:
                            outfile.write(text)
                output_paths = []
                task_list  = []




def process_raw_zhihu(data_dir, output_path):
    df = pd.DataFrame()
    for file in os.listdir(data_dir):
        df = pd.concat([df, pd.read_parquet(os.path.join(data_dir, file))])
    # df = df[df['INSTRUCTION'].str.contains('动画', na=False)]
    df.to_json(output_path, orient='records', lines=True, index=False, force_ascii=False)

def process_raw_moegirl(data_dir, output_path):
    samples = []
    for file in tqdm(os.listdir(data_dir)):
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            text = f.read()
        samples.append({'title': file.removesuffix('.txt'), 'text': text})
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in samples:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')



from datetime import datetime, date
import json
from decimal import Decimal
def json_serializer(obj):
    """处理常见不可序列化类型"""
    if isinstance(obj, (datetime, date)):
        # 将 datetime/date 转为 ISO 8601 格式字符串
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        # 将 Decimal 转为 float（或 str 保留精度）
        return float(obj)  # 或 str(obj)
    elif isinstance(obj, bytes):
        # 将 bytes 转为 base64 字符串（如存储二进制数据）
        return obj.decode('utf-8', errors='ignore')  # 或使用 base64.b64encode(obj).decode()
    else:
        # 无法处理的类型抛出错误，防止遗漏
        raise TypeError(f"Type {type(obj)} not serializable")

def process_raw_bilibili(table_name, output_path, start_id, batch_size):
    query = f"""SELECT * FROM `{table_name}` WHERE rpid > %s 
               ORDER BY rpid 
               LIMIT %s"""
    conn = pymysql.connect(host='localhost',
                           user='crawler',
                           password='123456',
                           db='media_crawler',
                           charset='utf8mb4',  # 支持 Emoji 和特殊字符
                           cursorclass=pymysql.cursors.SSCursor)  # 使用流式游标减少内存占用

    with conn.cursor() as cursor:
        cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 1")
        rows = cursor.fetchall()
        if not rows:
            raise  ValueError("No data found in the table.")
        columns = [col[0] for col in cursor.description]
        last_id = start_id
        pbar = tqdm(total=261856016192, desc="Processing rows", unit="row")

        with open(output_path, 'w', encoding='utf-8') as f:
            while True:
                # 分批读取数据
                cursor.execute(query, (last_id, batch_size))
                rows = cursor.fetchall()
                if not rows:
                    break

                # 逐行处理并写入 JSONL
                for row in rows:
                    # 组合列名与数据
                    row_dict = dict(zip(columns, row))
                    # 转换为 JSON 字符串并写入文件
                    json_line = json.dumps(row_dict, default=json_serializer, ensure_ascii=False)
                    f.write(json_line + '\n')
                pbar.update(row_dict['rpid'] - last_id)
                last_id = row_dict['rpid']
    conn.close()


def process_bilibili(article_path, comment_path, output_path):
    with open(comment_path, 'r', encoding='utf-8') as f:
        comments = {json.loads(line)['rpid']: json.loads(line) for line in f}
    lines = []
    filter_words = ['流量卡', '壁纸', '动漫壁纸', '美图']
    filter_category_ids = []
    with open(article_path, 'r', encoding='utf-8') as f:
        for article in tqdm(f):
            article = json.loads(article)
            if not article['words'] or not article['content'] or article['words'] < 512:
                continue
            if article['title'] in filter_words or article['category_id'] in filter_category_ids:
                continue
            if article['tags'] != '[]':
                tags = json.loads(article['tags'])
                if any(tag in tags for tag in filter_words):
                    continue
            article_text_list = ['URL: ' + article['link'],
                                 '标题: ' + article['title'],
                                 '作者: ' + article['author_name'],
                                 '发布日期: ' + article['publish_time'],
                                 article['content'],
                                 '标签: ' + ', '.join(json.loads(article['tags'])) if article['tags'] != '[]' else '']
            article_text = '\n\n'.join(article_text_list)
            article_text = article_text.strip()


            if article['comment_ids']:
                comment_texts = []
                comment_ids = json.loads(article['comment_ids'])
                n = len(comment_ids)
                i = 0
                while i < n:
                    try:
                        comment_id = comment_ids[i]
                        comment = comments[int(comment_id)]
                        if comment['author_name'] is None:
                            comment['author_name'] = "无名"
                        if comment['content'] is None:
                            continue
                        if comment['publish_time']:
                            comment_time = ' ' + comment['publish_time'].replace('T', ' ')
                        else:
                            comment_time = ''
                        comment_text = '\n'.join([comment['author_name'] + comment_time,
                                                  comment['content']])

                        if comment['comment_ids']:
                            sub_comment_ids = json.loads(comment['comment_ids'])
                            # 修改子评论的 @用户名
                            id_to_user = {int(comment_id): comment['author_name']}
                            for sub_comment_id in sub_comment_ids:
                                sub_comment = comments[int(sub_comment_id)]
                                if sub_comment['author_name']:
                                    id_to_user[int(sub_comment_id)] = sub_comment['author_name']
                            pattern = r'@.+:'
                            for sub_comment_id in sub_comment_ids:
                                sub_comment = comments[int(sub_comment_id)]
                                if sub_comment['content'] is not None and sub_comment['parent'] in id_to_user:
                                    sub_comment['content'] = regex.sub(pattern, '@' + id_to_user[sub_comment['parent']] + ':', sub_comment['content'])
                                if not sub_comment['content'].startswith('回复 @') and sub_comment['parent'] in id_to_user:
                                    sub_comment['content'] = '回复 @' + id_to_user[sub_comment['parent']] + ':' + sub_comment['content']

                            # 将子评论 id 插入评论 id 列表中
                            comment_ids = comment_ids[:i + 1] + sub_comment_ids + comment_ids[i + 1:]
                            n = len(comment_ids)
                        comment_texts.append(comment_text)
                    except  KeyError:
                        pass
                        # print(f'Comment {comment_id} not found in comments.jsonl')
                    i += 1

                if comment_texts:
                    comment_text = '\n\n'.join(comment_texts).strip()
                    article_text += '\n\n' + comment_text
            line = {'text': article_text}
            lines.append(line)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f"Processed {len(lines)} bilibili articles")


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


async def main():
    process_bilibili('../data/bilibili/raw/articles.jsonl',
                     '../data/bilibili/raw/comments.jsonl',
                     '../data/bilibili/bilibili.jsonl')

    # process_moegirl('../data/moegirl/raw/moegirl.jsonl',
    #                 '../data/moegirl/moegirl.jsonl')

    # process_zhihu(data_dir='../data/zhihu/raw', output_path='../data/zhihu/zhihu.jsonl')
    # process_moegirl(data_dir='C:\\Users\\MakiseKurisu\\Downloads\\moegirl\\raw\\mgp_archive_2505\\archive0',
    #                 output_path='../data/moegirl/moegirl.jsonl')
    # await convert_epub_to_txt(data_dir='../data/novels/raw/epub',
    #                           output_dir='../data/novels/raw/txt',
    #                           batch_size=16)

    # export_mysql('bilibili_comment',
    #              '../data/bilibili/raw/comments.jsonl',
    #              0,
    #              10000)


# process_bilibili('../data/bilibili/raw', '../data/bilibili/test', ['动画', '番剧'])

# select_from_index(data_dir=r'\\10.1.116.113\public\Z-library',
#                   index_path='../data/selected_index.txt',
#                   output_dir='../data/epub')

# delete_from_index(data_dir='../data/epub', index_path='../data/selected_index.txt')

if __name__ == '__main__':
    asyncio.run(main())
