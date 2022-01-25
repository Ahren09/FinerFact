import logging
import os
import os.path as osp
import pickle
import random

import torch

from kgat.data_loader import FNNDataLoader
from utils.utils_misc import get_root_dir

logger = logging.getLogger(__name__)


def read_news_article_evidence(dataset, processed_dir=None):
    data_dir = get_root_dir() if processed_dir is None else processed_dir
    path = osp.join(data_dir, f"{dataset}_news_article_evidence.pkl")

    with open(path, "rb") as f:
        examples = pickle.load(f)
    # import torch
    # all_tweet_d = torch.load(osp.join(get_root_dir(), f"{dataset}_tweets.pt"))

    return examples

def load_filenames_sampled(processed_dir, labels_d, args):
    path_filenames_sampled = osp.join(processed_dir, f"{args.dataset}_filenames{args.sample_suffix}.pt")
    if osp.exists(path_filenames_sampled):
        filenames_sampled = torch.load(path_filenames_sampled)
    else:
        filenames_sampled = labels_d.keys()
    return filenames_sampled

def read_claim_evidence_pairs(args):
    debug_suffix = "_DEBUG" if args.debug else ""
    path = osp.join(get_root_dir(), f"{args.dataset}_claim_evidence_pairs{debug_suffix}.pt")
    print(path)
    all_claim_evidence_pairs_d = torch.load(path)
    all_pr_scores_d = torch.load(
        os.path.join(get_root_dir(), f"{args.dataset}_pr_scores{debug_suffix}.pt"))
    all_metadata_d = torch.load(
        os.path.join(get_root_dir(), f"{args.dataset}_claim_evidence_pairs_metadata{debug_suffix}.pt"))

    return all_claim_evidence_pairs_d, all_pr_scores_d, all_metadata_d


def load_mr(args, processed_dir):
    path = osp.join(processed_dir, f"{args.dataset}_all_mr_d.pt")
    if osp.exists(path):
        logger.info("Loading MR ...")
        all_mr_d = torch.load(path)
    else:
        all_mr_d = {}
    return all_mr_d


def load_user_embed_and_Rs(args, processed_dir):
    path_usr = osp.join(processed_dir, f"{args.dataset}_user_embed.pt")
    path_Rs = osp.join(processed_dir, f"{args.dataset}_Rs.pt")

    if osp.exists(path_usr):
        print("Loading User Embedding ...")
        all_user_embed_d = torch.load(path_usr)
    else:
        print("-" * 5 + "User Embedding empty!!" + "-" * 5)
        all_user_embed_d = {}
    if osp.exists(path_Rs):
        print("Loading Mutual Reinforcement scores Rs ...")
        all_Rs_d = torch.load(path_Rs)
    else:
        all_Rs_d = {}

    return all_user_embed_d, all_Rs_d


def save_user_embed(all_user_embed_d, args, processed_dir):
    path = osp.join(processed_dir, f"{args.dataset}_user_embed.pt")

    if not osp.exists(path) or args.reprocess:
        print("Saving User Embedding ...")
        torch.save(all_user_embed_d, path)
    return all_user_embed_d

def save_Rs(all_Rs_d, args, processed_dir):
    path = osp.join(processed_dir, f"{args.dataset}_Rs.pt")
    if not osp.exists(path) or args.reprocess:
        print("Saving R scores ...")
        torch.save(all_Rs_d, path)
    return all_Rs_d


def merge_inputs(inputs_e, inputs_s, args):
    """
    Merge the inputs of external knowledge (including the news article)
    and inputs of social context
    """
    assert len(inputs_e) == len(inputs_s)
    inputs = []
    for i, data in enumerate(inputs_e):
        inps_e, msks_e, segs_e = data
        if inputs_s[i] != []:
            inps_s, msks_s, segs_s = inputs_s[i]
            inps_e += inps_s
            msks_e += msks_s
            segs_e += segs_s
            inps_e = inps_e[:args.evi_num]
            msks_e = msks_e[:args.evi_num]
            segs_e = segs_e[:args.evi_num]
        inputs += [[inps_e, msks_e, segs_e]]
    assert len(inputs_e) == len(inputs)
    return inputs


def get_processed_dir(exp_name=None):
    if exp_name is not None:
        processed_dir = osp.join(get_root_dir(), "back", exp_name)
        if not osp.exists(processed_dir):
            os.mkdir(processed_dir)
        return processed_dir
    else:
        return get_root_dir()


def sample_filenames(labels_d, args):
    examples_real = [filename for filename, label in labels_d.items() if label == 0]
    examples_fake = [filename for filename, label in labels_d.items() if label == 1]
    filenames_sampled = random.sample(examples_real, int(args.sample_ratio * len(labels_d) * 0.5)) + random.sample(
        examples_fake, int(args.sample_ratio * len(labels_d) * 0.5))
    return filenames_sampled


def get_train_test_readers(label_map, tokenizer, args, test=False):
    """
    Function for getting train-test split
    """
    if args.kfold_index >= 0:
        # NOTE: args.path_train changed here!!
        args.path_train = osp.join(args.data_dir, f"Train_{args.prefix}_KFold{args.kfold_index}.pt")
        args.path_test = osp.join(args.data_dir, f"Test_{args.prefix}_KFold{args.kfold_index}.pt")

    if os.path.exists(args.path_train) and os.path.exists(args.path_test):
        if not test:
            logger.info(f"Loading train files {args.path_train}")
            filenames_train, inputs_train, labels_train, aux_info_train, user_embeds_train, user_metadata_train = torch.load(args.path_train)

            # Shuffle training files
            train_shuffled = list(zip(filenames_train, inputs_train, labels_train, aux_info_train, user_embeds_train, user_metadata_train))
            random.shuffle(train_shuffled)

            filenames_train, inputs_train, labels_train, aux_info_train, user_embeds_train, user_metadata_train = zip(*train_shuffled)

        logger.info(f"Loading test files {args.path_test}")

        # Shuffle test files
        filenames_test, inputs_test, labels_test, aux_info_test, user_embeds_test, user_metadata_test = torch.load(args.path_test)

        test_shuffled = list(zip(filenames_test, inputs_test, labels_test, aux_info_test, user_embeds_test, user_metadata_test))
        random.shuffle(test_shuffled)
        filenames_test, inputs_test, labels_test, aux_info_test, user_embeds_test, user_metadata_test = zip(*test_shuffled)

    else:
        raise Exception("Error: must preprocess input files first")

    if test:
        trainset_reader = None
    else:
        # TODO: sort number of tweets for training examples here

        logger.info("loading train set")
        trainset_reader = FNNDataLoader(label_map, tokenizer, args, inputs=inputs_train,
                                        # inputs_s=inputs_s_train,
                                        filenames=filenames_train,
                                        labels=labels_train,
                                        aux_info=aux_info_train,
                                        user_embeds=user_embeds_train,
                                        user_metadata=user_metadata_train,
                                        batch_size=args.train_batch_size)
    logger.info("loading validation set")
    validset_reader = FNNDataLoader(label_map, tokenizer, args, inputs=inputs_test,
                                    # inputs_s=inputs_s_test,
                                    filenames=filenames_test,
                                    labels=labels_test,
                                    aux_info=aux_info_test,
                                    user_embeds=user_embeds_test,
                                    user_metadata=user_metadata_test,
                                    batch_size=args.valid_batch_size, test=True)

    return trainset_reader, validset_reader
