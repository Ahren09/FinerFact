import os
import os.path as osp
import pickle

import pandas as pd
import torch

# Which fields are used in encoding user features?
fields_user_num = ['followers_count', 'friends_count', 'listed_count', 'statuses_count', 'favourites_count', 'len_name',
                   'len_screen_name', 'len_description']
fields_user_cat = ['geo_enabled', 'verified', 'has_location']

fields_user = ['followers_count', 'friends_count', 'listed_count', 'statuses_count', 'favourites_count', 'geo_enabled', 'verified', 'has_location']

def get_root_dir():
    """
    Change this to the root data directory
    :return: root directory
    """
    root = osp.join("..", "fake_news_data")
    if os.name == "posix":
        root = osp.join("..", "fake_news_data")
    else:
        root = osp.join("C:\\Workspace", "FakeNews", "fake_news_data")
    return root


def get_processed_dir(exp_name=None):
    if exp_name is not None:
        processed_dir = osp.join(get_root_dir(), "back", exp_name)
        if not osp.exists(processed_dir):
            os.mkdir(processed_dir)
        return processed_dir
    else:
        return get_root_dir()


def load_tweets(dataset):
    return torch.load(osp.join(get_root_dir(), f"{dataset}_tweets.pt"))


def save_tweets(dataset, all_tweets_d, all_replies_d, all_tweets_score_d):
    torch.save((all_tweets_d, all_replies_d, all_tweets_score_d), osp.join(get_root_dir(), f"{dataset}_tweets.pt"))


def load_users(dataset):
    all_user_feat_d = torch.load(osp.join(get_root_dir(), f"{dataset}_users.pt"))
    return all_user_feat_d


def save_users(dataset, all_user_feat_d):
    torch.save(all_user_feat_d, osp.join(get_root_dir(), f"{dataset}_users.pt"))


def load_nx_graphs(dataset):
    all_Gu = torch.load(osp.join(get_root_dir(), f"{dataset}_Gu.pt"))
    return all_Gu


def save_Gu(dataset, all_Gu):
    torch.save(all_Gu, osp.join(get_root_dir(), f"{dataset}_Gu.pt"))


def load_labels(dataset):
    processed_dir = get_root_dir()
    labels_d = torch.load(osp.join(processed_dir, f"{dataset}_labels.pt"))
    return labels_d


def save_labels(dataset, labels_d):
    torch.save(labels_d, osp.join(get_root_dir(), f"{dataset}_labels.pt"))


def load_user_feat(dataset):
    import torch
    all_Gu = torch.load(osp.join(get_root_dir(), f"{dataset}_Gu.pt"))
    return all_Gu


def read_news_article_evidence(dataset):
    """
    format: claim_evi_pair, news_article, label
    :param dataset:
    :return:
    """
    data_dir = get_root_dir()
    path = osp.join(data_dir, f"{dataset}_news_article_evidence.pkl")

    with open(path, "rb") as f:
        examples = pickle.load(f)
    return examples


def read_tweets_and_scores(dataset):
    import torch
    all_tweets_d, all_replies_d, all_tweets_score_d = torch.load(osp.join(get_root_dir(), f"{dataset}_tweets.pt"))
    return all_tweets_d, all_replies_d, all_tweets_score_d


def read_news_articles_text(global_news_article_d, dataset_name="politifact"):
    root = get_root_dir()
    with open(osp.join(root, f"{dataset_name}_news_articles.txt"), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            filename, article = line.split("\t")
            article = article.strip()
            global_news_article_d[filename] = article
        f.close()


def read_news_articles_labels(dataset_name="politifact", n_samples=0):
    KEEP_EMPTY_RETWEETS_AND_REPLIES = 1

    # if we only read the first `n_samples` samples in the dataframe
    if n_samples > 0:
        news_article_df = pd.read_csv(get_root_dir() + f"\\{dataset_name}_news_articles.tsv", sep='\t', iterator=True,
                                      header=None)
        news_article_df = news_article_df.get_chunk(n_samples)

    else:

        news_article_df = pd.read_csv(get_root_dir() + f"\\{dataset_name}_news_articles.tsv", sep='\t')

    if KEEP_EMPTY_RETWEETS_AND_REPLIES:
        news_article_cleaned_df = news_article_df[
            (news_article_df.has_tweets == 1) & (news_article_df.has_news_article == 1)]
    else:
        news_article_cleaned_df = news_article_df[
            (news_article_df.has_tweets == 1) & (news_article_df.has_news_article == 1) & (
                    news_article_df.has_retweets == 1) & (news_article_df.has_replies == 1)]
    return news_article_cleaned_df


def only_directories(path):
    return [name for name in os.listdir(path) if osp.isdir(osp.join(path, name))]


def get_data_list():
    politifact_fake = only_directories("politifact_fake")
    politifact_real = only_directories("politifact_real")
    gossipcop_fake = only_directories("gossipcop_fake")
    gossipcop_real = only_directories("gossipcop_real")

    data_list = {
        "politifact_real": politifact_real,
        "politifact_fake": politifact_fake,

        "gossipcop_fake" : gossipcop_fake,
        "gossipcop_real" : gossipcop_real
    }

    return data_list


def get_dataset_names(dataset):
    if dataset == "politifact":
        dataset_names = {
            "politifact": ["politifact_real", "politifact_fake"]
        }
    elif dataset == "gossipcop":
        dataset_names = {
            "gossipcop": ["gossipcop_real", "gossipcop_fake"]
        }
    elif dataset == "both":
        dataset_names = {
            "politifact": ["politifact_real", "politifact_fake"],
            "gossipcop" : ["gossipcop_real", "gossipcop_fake"]
        }
    else:
        raise NotImplementedError
    return dataset_names


def filter_empty_dict_entry(d, filename, log=True):
    is_empty_json = False
    new_d = {}
    for k in d:
        if d[k] != []:
            new_d[k] = d[k]
    if new_d == {}:
        if log:
            print(f"\t{filename} json empty")
        is_empty_json = True

    return new_d, is_empty_json


# For a dict of dicts, filter empty entries, which are {}
def filter_empty_nested_dict(d):
    new_d, empty_li = {}, []
    for k, v in d.items():
        if v == {}:
            empty_li += [k]
        else:
            new_d[k] = d[k]
    return new_d, empty_li


def print_results(results, epoch, dataset_split_name="Train", enable_logging=True, args=None):
    log_str = f"\n[{dataset_split_name}] Epoch {epoch}\n\tPre: {results['pre']:.4f}, Rec: {results['rec']:.4f}\n\tAcc: {results['acc']:.4f}, F1: {results['f1']:.4f}\n"
    print(log_str)
    if enable_logging:
        f = open(f"{args.outdir}/{dataset_split_name}_{args.max_len}_{args.evi_num}_results.txt", "a+")
        f.write(log_str)


def load_tweet_df(filename, dataset):
    import pandas as pd
    import numpy as np

    pd.options.display.max_columns = 20
    pd.set_option('precision', 20)

    # NOTE: reading as int64 is super important
    dtypes = {
        'root_tweet_id': np.int64,
        'tweet_id'     : np.int64,
        'root_user_id' : np.int64,
        'user_id'      : np.int64,
    }

    path = osp.join(get_root_dir(), dataset, filename, "tweets_retweets_comments.tsv")
    if not osp.exists(path):
        print(f"\t SKIP {filename}: no tweet_retweet_comment.tsv")
        return None

    tweet_df = pd.read_csv(path, sep='\t', float_precision='high')
    return tweet_df


def print_heading(dataset):
    print("-" * 30 + f"\n# Processing {dataset}\n" + "-" * 30 + "\n")
