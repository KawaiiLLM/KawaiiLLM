import argparse
import sys
from tqdm import tqdm
import orjson

import re

def parse_prediction(predict_str):
    try:
        if not predict_str:
            return None, None

        # Use regex to find the pattern "digit, digit"
        # This handles extra text like <think> tags
        match = re.search(r'(\d)\s*,\s*(\d)', predict_str)
        if match:
            q = int(match.group(1))
            d = int(match.group(2))
            return q, d

        # Fallback to simple split if regex fails (though regex is safer)
        parts = predict_str.split(',')
        if len(parts) >= 2:
            q = int(parts[0].strip())
            d = int(parts[1].strip())
            return q, d
    except:
        pass
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Filter Q=1 articles and merge scores")
    parser.add_argument("--articles", required=True, help="Path to articles_cleaned_v3.jsonl")
    parser.add_argument("--predictions", required=True, help="Path to generated_predictions_full.jsonl")
    parser.add_argument("--output", required=True, help="Path to output file")

    args = parser.parse_args()

    print(f"Processing...")
    print(f"Articles: {args.articles}")
    print(f"Predictions: {args.predictions}")
    print(f"Output: {args.output}")

    count_total = 0
    count_kept = 0

    try:
        # Open files
        with open(args.articles, 'rb') as f_art, \
             open(args.predictions, 'rb') as f_pred, \
             open(args.output, 'wb') as f_out:

            # Use zip to iterate both
            # We need to handle the case where files have different lengths, but zip stops at shortest
            # Ideally they should be same length.

            # Using tqdm on the zip object might not show total if not provided,
            # but we can just iterate.

            for line_art, line_pred in tqdm(zip(f_art, f_pred)):
                count_total += 1

                try:
                    # Parse prediction first (faster than parsing full article)
                    pred_obj = orjson.loads(line_pred)
                    # The prediction file might have 'predict' field
                    predict_str = pred_obj.get('predict', '')

                    q, d = parse_prediction(predict_str)

                    # Filter: Keep only if Q == 1
                    if q == 1:
                        # Parse article
                        article_obj = orjson.loads(line_art)

                        # Add scores
                        article_obj['quality_score'] = q
                        article_obj['domain_score'] = d

                        # Write to output
                        f_out.write(orjson.dumps(article_obj) + b'\n')
                        count_kept += 1

                except Exception as e:
                    # print(f"Error on line {count_total}: {e}", file=sys.stderr)
                    continue

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Processed {count_total} lines.")
    print(f"Kept {count_kept} articles (Q=1).")
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
