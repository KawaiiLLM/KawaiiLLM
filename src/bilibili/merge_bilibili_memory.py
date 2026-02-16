import json
import os
from tqdm import tqdm
import sys

# Paths
BASE_DIR = "../../datasets/bilibili-articles"
ARTICLES_PATH = os.path.join(BASE_DIR, "articles.jsonl")
COMMENTS_PATH = os.path.join(BASE_DIR, "comments.jsonl")
OUTPUT_DIR = "data/bilibili/raw"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "merged_articles.jsonl")

def load_comments_memory():
    print(f"Loading comments from {COMMENTS_PATH} into memory...")
    comments = {}

    # Estimate lines for tqdm (approximate)
    # 5.2GB file, avg line 500 bytes? ~10M lines

    try:
        with open(COMMENTS_PATH, 'r', encoding='utf-8', errors='replace') as f:
            for line in tqdm(f, desc="Loading Comments", unit="line"):
                try:
                    c = json.loads(line)
                    rpid = c.get('rpid')
                    if rpid:
                        # Optimization: Only keep necessary fields to save RAM
                        # If you have plenty of RAM, you can keep 'c' directly.
                        # comments[rpid] = c

                        # Memory optimized version:
                        comments[rpid] = {
                            'rpid': rpid,
                            'content': c.get('content'),
                            'author_name': c.get('author_name'),
                            'like': c.get('like', 0),
                            'parent': c.get('parent', 0),
                            'comment_ids': c.get('comment_ids', []) # For sub-comments
                        }
                except:
                    continue
    except MemoryError:
        print("Error: Not enough memory to load all comments!")
        sys.exit(1)

    print(f"Loaded {len(comments)} comments into memory.")
    return comments

def fetch_comments_recursive(comment_ids_input, comments_map, depth=0, max_depth=2):
    if depth > max_depth or not comment_ids_input:
        return []

    result_comments = []
    try:
        # Handle string vs list input
        if isinstance(comment_ids_input, str):
            if comment_ids_input == '[]':
                return []
            comment_ids = json.loads(comment_ids_input)
        else:
            comment_ids = comment_ids_input

        for cid in comment_ids:
            cid_int = int(cid)
            if cid_int in comments_map:
                # Create a copy to avoid modifying the global map if we were caching
                # But here we just want to attach sub-comments to the output object
                # Since comments_map values are dicts, we should copy if we modify them.
                # However, modifying the dict in comments_map would affect other references?
                # No, comments are unique per rpid. But let's be safe and copy.
                c_obj = comments_map[cid_int].copy()

                # Recursively fetch sub-comments
                sub_ids = c_obj.get('comment_ids')
                if sub_ids:
                    c_obj['replies'] = fetch_comments_recursive(sub_ids, comments_map, depth + 1, max_depth)

                result_comments.append(c_obj)
    except Exception:
        pass

    return result_comments

def merge_articles(comments_map):
    print(f"Merging articles from {ARTICLES_PATH}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(ARTICLES_PATH, 'r', encoding='utf-8', errors='replace') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Processing Articles", unit="line"):
            try:
                article = json.loads(line)

                # Get top-level comment IDs
                comment_ids_str = article.get('comment_ids')

                if comment_ids_str:
                    article['comments'] = fetch_comments_recursive(comment_ids_str, comments_map)
                else:
                    article['comments'] = []

                f_out.write(json.dumps(article, ensure_ascii=False) + '\n')

            except Exception as e:
                continue

def main():
    if not os.path.exists(COMMENTS_PATH):
        print(f"File not found: {COMMENTS_PATH}")
        return

    comments_map = load_comments_memory()
    merge_articles(comments_map)
    print(f"Done! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
