import json
import os
from datetime import datetime, timedelta

from opensearchpy import OpenSearch, helpers

from config import num_train_day, num_val_day

opensearch_endpoint = "localhost:40444"
username = "Kinesis-test1"
password = "Kinesis-test1"

os_client = OpenSearch(
    [opensearch_endpoint],
    http_auth=(username, password),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
    timeout=120,
)

if __name__ == "__main__":
    today = datetime.today().isoformat()[:10]
    os_delay = 3
    for i in range(os_delay, num_train_day + num_val_day + os_delay):
        date = (datetime.fromisoformat(today) - timedelta(days=i)).isoformat()[:10]
        if os.path.exists(f"./data/dataset_{date}.jsonl"):
            print(f"dataset_{date}.jsonl already exists")
            continue
        for hour in range(6):
            start_hour = str(4 * hour)
            end_hour = str(4 * hour + 3)
            if len(start_hour) < 2:
                start_hour = "0" + start_hour
            if len(end_hour) < 2:
                end_hour = "0" + end_hour
            # print(start_hour)
            # print(end_hour)
            start_time = f"{date}T{start_hour}:00:00Z"
            end_time = f"{date}T{end_hour}:59:59Z"
            query = {
                "_source": [
                    "doc_id",
                    "twitter_tweet_text",
                    "twitter_tweet_author_detail",
                    "twitter_tweet_media_detail",
                    "llm_relevance_score",
                    "twitter_kkol_engagement_count",
                    "engagement_count",
                    "twitter_tweet_like_count",
                    "twitter_tweet_reply_count",
                    "twitter_is_from_official_account",
                    "embedding",
                ],
                "query": {
                    "bool": {
                        "must": [
                            {"exists": {"field": "embedding"}},
                            {
                                "range": {
                                    "created_at": {"gte": start_time, "lte": end_time}
                                }
                            },
                            {"term": {"twitter_tweet_is_retweet": "false"}},
                        ]
                    }
                },
            }
            scanned_data = helpers.scan(
                os_client, query=query, index="twitter", size=1000, scroll="1m"
            )
            with open(f"dataset_{date}.jsonl", "a") as f:
                for doc in scanned_data:
                    print("got doc")
                    f.write(json.dumps(doc) + "\n")
