import os
import os.path as osp

import pandas as pd
import torch
from sklearn import metrics
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def read_news_articles_text(global_news_article_d, args, dataset_name="politifact"):
    with open(osp.join(get_root_dir(), f"{dataset_name}_news_articles.txt"), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            filename, article = line.split("\t")
            article = article.strip()
            global_news_article_d[filename] = article
        f.close()


def read_news_articles_labels(dataset_name="politifact", n_samples=0):
    KEEP_EMPTY_RETWEETS_AND_REPLIES = 1

    # if we only read the first `n_samples` samples in the dataframe
    if n_samples > 0:
        news_article_df = pd.read_csv(osp.join(get_root_dir(), f"{dataset_name}_news_articles.tsv"), sep='\t',
                                      iterator=True, header=None)
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
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


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


def set_args_from_config(args, config):
    config_gat = config["KGAT"]
    args.data_dir = config_gat.get("data_dir", "../data")
    args.bert_pretrain = config_gat.get("bert_pretrain", "../bert_base")
    args.dropout = config_gat.getfloat("dropout")
    args.evi_num = config_gat.getint("evi_num")

    # Note that the following are the same
    args.num_tweets = config["gat.social"].getint("num_tweets_in_each_pair", 6)
    args.sent_num = config["gat.social"].getint("num_tweets_in_each_pair", 6)

    args.num_words_per_topic = config["pagerank"].getint("num_words_per_topic", 6)
    args.gradient_accumulation_steps = config_gat.getfloat("gradient_accumulation_steps")
    # args.kernel = config_gat.getint("kernel")
    args.learning_rate = config_gat.getfloat("learning_rate")
    args.max_len = config_gat.getint("max_len")
    args.num_train_epochs = config_gat.getint("num_train_epochs")
    args.train_batch_size = config_gat.getint("train_batch_size")
    args.valid_batch_size = config_gat.getint("valid_batch_size")
    args.cuda = config_gat.getboolean("cuda")

    args.dataset = config_gat.get("dataset")
    args.model_name = config_gat.get("model_name")
    args.mode = config_gat.get("mode")
    args.sample_ratio = config_gat.getfloat("sample_ratio")
    args.test_size = config_gat.getfloat("test_size")
    args.keep_claim = config_gat.getboolean("keep_claim", False)
    args.only_claim = config_gat.getboolean("only_claim", False)
    args.sample_suffix = f"_SAMPLE{args.sample_ratio}" if args.sample_ratio is not None else ""
    args.user_embed_dim = config_gat.getint("user_embed_dim", 64)
    args.num_users = config_gat.getint("num_users", 32)
    args.warmup_ratio = config_gat.getfloat("warmup_ratio", 0.03)
    args.enable_tensorboard = config_gat.getboolean("enable_tensorboard", False)
    args.pretrained_user_embed = config_gat.getboolean("pretrained_user_embed", False)


    args.prefix = f"{args.model_name}_{args.dataset}_{args.max_len}_{args.evi_num}"
    # args.path = os.path.join(args.data_dir, f"{args.prefix}{args.sample_suffix}")
    args.path_train = os.path.join(args.data_dir, f"Train_{args.prefix}{args.sample_suffix}.pt")
    args.path_test = os.path.join(args.data_dir, f"Test_{args.prefix}{args.sample_suffix}.pt")

    return args


def get_eval_report(labels, preds, probs, do_plot=False):
    # mcc = matthews_corrcoef(labels, preds)
    [tn, fp, fn, tp] = list(confusion_matrix(labels, preds, labels=[0, 1]).ravel())

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, pos_label=1, average="binary")
    assert p == precision_score(labels, preds, zero_division=0)
    assert r == recall_score(labels, preds, zero_division=0)
    assert f1 == f1_score(labels, preds, average='binary')
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs, pos_label=1)
    if do_plot:
        plt.plot(fpr, tpr)
        plt.show()
    auc = metrics.auc(fpr, tpr)
    results = {
        # "mcc": mcc,
        # "tp": tp,
        # "tn": tn,
        # "fp": fp,
        # "fn": fn,
        "pre": p,
        "rec": r,
        "f1": f1,
        "acc": accuracy_score(labels, preds),
        "auc": auc
    }

    return results


def print_results(results, epoch, dataset_split_name="Train", enable_logging=True, args=None, tweets_np_segment=None):
    if tweets_np_segment is None:
        log_str = f"\n[{dataset_split_name}] Epoch {epoch}\tPre: {results['pre']:.4f}, Rec: {results['rec']:.4f}\tAcc: {results['acc']:.4f}, F1: {results['f1']:.4f}\n"
    else:
        log_str = f"[#Tweets: {str(int(tweets_np_segment[0])):5} - {str(int(tweets_np_segment[-1])):5}] \tPre: {results['pre']:.4f}, Rec: {results['rec']:.4f}, Acc: {results['acc']:.4f}, F1: {results['f1']:.4f}"
    log_str += f", AUC: {results['auc']:.4f}" if 'auc' in results else ""
    print(log_str)


def get_root_dir():
    root = None
    if os.name == "posix":
        root = "../../fake_news_data"
    else:
        root = "C:\\Workspace\\FakeNews\\fake_news_data"
    return root


def save_results_to_tsv(results_train, results_eval, experiment_name, args):
    results_train_df = pd.DataFrame.from_dict(results_train).transpose()
    results_eval_df = pd.DataFrame.from_dict(results_eval).transpose()

    results_train_df.to_csv(osp.join("..", "logs", experiment_name,
                                     f"Train_{experiment_name}{'_KFold' + str(args.kfold_index) if args.kfold_index >= 0 else ''}.tsv"),
                            sep='\t')
    results_eval_df.to_csv(osp.join("..", "logs", experiment_name,
                                    f"Eval_{experiment_name}{'_KFold' + str(args.kfold_index) if args.kfold_index >= 0 else ''}.tsv"),
                           sep='\t')


def read_G_triple(args):
    suffix = args.suffix if args.suffix is not None else ""
    all_G_triple = torch.load(os.path.join(get_root_dir(), f"{args.dataset}_G_triple{suffix}.pt"))
    return all_G_triple


def print_msg(msg):
    print('-' * (len(msg) + 4) + f"\n* {msg} *\n" + '-' * (len(msg) + 4))
