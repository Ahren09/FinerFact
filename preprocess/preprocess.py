import argparse
import configparser
import logging
import os
import os.path as osp
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import BertTokenizer

from kgat.data_loader import tok2int_list_concat, tok2int_list_claim_evi_pair
from utils.utils import fields_user, fields_user_cat, fields_user_num
from utils.utils_misc import set_args_from_config, get_root_dir, read_news_articles_text
from utils.utils_preprocess import (read_news_article_evidence, load_mr, logger, load_user_embed_and_Rs, merge_inputs)

logger = logging.getLogger(__name__)


def preprocess(tokenizer, args):
    logger.info("Creating features from dataset file at %s", args.data_dir)

    filenames, inputs_e, inputs_s, labels = [], [], [], []
    aux_info, user_embeds, user_metadata = [], [], []

    # ------------------------------------------
    # Load and cache examples
    # ------------------------------------------

    logger.info("Loading news articles and evidence ...")

    processed_dir = get_root_dir()
    examples = read_news_article_evidence(args.dataset, processed_dir)
    all_mr_d = load_mr(args, processed_dir)

    global_news_article_d = {}
    read_news_articles_text(global_news_article_d, args, dataset_name=args.dataset)
    global_news_article_d = {k: v for k, v in global_news_article_d.items() if v != "" and k not in examples}

    labels_d = torch.load(osp.join(processed_dir, f"{args.dataset}_labels.pt"))

    all_user_embed_d, all_Rs_d = load_user_embed_and_Rs(args, processed_dir)

    user_metadata_d = torch.load(osp.join(processed_dir, f"{args.dataset}_users.pt"))

    """
    Note: Here we use the full dataset through examples.keys().
    You can add sampling step here if you want to try experimenting with just part of the dataset
    """
    filenames_sampled = examples.keys()
    logger.info(f"Sampling {len(filenames_sampled)} examples for training")

    for filename in filenames_sampled:

        print(f"\t{filename}")

        if filename in examples and not examples[filename][1] == []:
            example = examples[filename]
            inps_e, msks_e, segs_e, label = tok2int_list_concat(example, filename, labels_d, tokenizer, args.max_len,
                                                                max_seq_size=args.evi_num, keep_claim=args.keep_claim,
                                                                only_claim=args.only_claim)

        elif filename in global_news_article_d:
            example = global_news_article_d[filename]
            inps_e, msks_e, segs_e, label = tok2int_list_concat(example, filename, labels_d, tokenizer, args.max_len,
                                                                max_seq_size=args.evi_num, keep_claim=args.keep_claim,
                                                                only_claim=True)
        else:
            continue

        if inps_e[0] == [0] * args.max_len:
            continue

        filenames += [filename]
        inputs_e += [[inps_e, msks_e, segs_e]]
        labels += [label]

        if filename in all_mr_d:

            Rs, sim_topics, twt_weight_mat, claim_evi_pairs, _ = all_mr_d[filename]
            # Rs, _, _, _, _ = all_mr_d[filename]

            if claim_evi_pairs is not None:
                # To fix the bug that the length of claim evidence pair is doubled
                if len(claim_evi_pairs) == 2 * args.evi_num:
                    claim_evi_pairs_new = []
                    i = 0
                    while i < args.evi_num:
                        claim_evi_pairs_new += [[claim_evi_pairs[2 * i], claim_evi_pairs[2 * i + 1]]]
                        i += 1
                    assert len(claim_evi_pairs_new) == args.evi_num
                    claim_evi_pairs = claim_evi_pairs_new

                inps_s, msks_s, segs_s = tok2int_list_claim_evi_pair(claim_evi_pairs, filename, labels_d, tokenizer,
                                                                     args.max_len, max_seq_size=args.evi_num)

                inputs_s += [[inps_s, msks_s, segs_s]]
            else:
                inputs_s += [[]]  # aux_info += [[Rs, sim_topics, twt_weight_mat]]
        else:

            inputs_s += [[]]

        """Produce user embeddings from raw metadata"""
        num_users_total = args.evi_num * args.num_users
        if filename in user_metadata_d:
            user_df = user_metadata_d[filename]
            # Sanity check
            user_df = user_df.iloc[:num_users_total]

            user_feat = user_df[fields_user]

            """Following GCAN, rolling-pad the user metadata"""
            user_feat = np.pad(user_feat, [(0, num_users_total - len(user_df)), (0, 0)], 'wrap')
            user_feats = torch.tensor(user_feat.reshape(args.evi_num, args.num_users, len(fields_user)), dtype=torch.float)

        else:
            user_feats = torch.zeros((args.evi_num, args.num_users, len(fields_user)))

        """Pre-trained user embeddings"""
        if filename in all_user_embed_d:
            user_embed = all_user_embed_d[filename]
            Rs = all_Rs_d[filename]

        else:
            user_embed = torch.zeros(size=(args.evi_num, args.num_users, args.user_embed_dim))

        user_embeds += [user_embed]
        user_metadata += [user_feats]
        aux_info += [[Rs, None, None]]
    assert len(inputs_e) == len(inputs_s) == len(user_embeds)
    assert len(inputs_e) == len(filenames)
    assert len(inputs_e) == len(labels)

    label_count = Counter(labels)
    logger.info(f"Processed Real {label_count[0]} | Fake {label_count[1]}")
    logger.info("Saving features into cached file %s", args.path_cache)

    if args.separate_inputs:
        files = (filenames, inputs_e, inputs_s, labels, aux_info, user_embeds, user_metadata)
        torch.save(files, args.path_cache)
        return files
    else:
        """By default, we merge the inputs together"""
        inputs = merge_inputs(inputs_e, inputs_s, args)
        files = (filenames, inputs, labels, aux_info, user_embeds, user_metadata)
        torch.save(files, args.path_cache)
        return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--seed", default=2021, type=int, help="Random state")
    parser.add_argument("--kfold", default=None, type=int, help="Number of folds K-fold cross validation")
    parser.add_argument("--separate_inputs", action="store_true",
                        help="Do we keep the social context inputs and external knowledge inputs as separate files? Or do we merge them?")

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(osp.join("..", "kgat", "config", args.config_file))
    set_args_from_config(args, config)

    args.path_cache = os.path.join(args.data_dir, f"{args.prefix}{args.sample_suffix}.pt")

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)

    if os.path.exists(args.path_cache) and not args.reprocess:
        logger.info("Loading features from cache %s", args.path_cache)
        files = torch.load(args.path_cache)

    else:
        files = preprocess(tokenizer, args)

    filenames, inputs, labels, aux_info, user_embeds, user_metadata = files

    labels_d = torch.load(os.path.join(get_root_dir(), f"{args.dataset}_labels.pt"))

    if args.kfold is not None:
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)


        def get_items(li, indices):
            return [li[idx] for idx in indices]


        index_loaded = set([])
        for i, (train_index, test_index) in enumerate(skf.split(filenames, labels)):
            print(f"Generating fold K={i}")
            files_train, files_test = [], []

            for file_li in [filenames, inputs, labels, aux_info, user_embeds, user_metadata]:
                files_train += [get_items(file_li, train_index)]
                files_test += [get_items(file_li, test_index)]
            files_true = [filename for i, filename in enumerate(files_test[0]) if labels_d[filename] == 0]
            files_false = [filename for i, filename in enumerate(files_test[0]) if labels_d[filename] == 1]
            print(f"\tT: {len(files_true)} | F: {len(files_false)}")
            index_loaded = set(test_index) | index_loaded
            # Assert no overlap in training & test files
            assert list(set(files_train[0]) & set(files_test[0])) == []
            torch.save(files_train, osp.join(args.data_dir, f"Train_{args.prefix}{args.sample_suffix}_KFold{i}.pt"))
            torch.save(files_test, osp.join(args.data_dir, f"Test_{args.prefix}{args.sample_suffix}_KFold{i}.pt"))
        assert len(index_loaded) == len(filenames)
    else:

        inputs_train, inputs_test, filenames_train, filenames_test, labels_train, labels_test, aux_info_train, aux_info_test, user_embeds_train, user_embeds_test, user_metadata_train, user_metadata_test = train_test_split(
            inputs, filenames, labels, aux_info, user_embeds, user_metadata, test_size=args.test_size, random_state=args.seed,
            stratify=labels)

        # Assert no overlap in training & test files
        assert list(set(filenames_train) & set(filenames_test)) == []

        print("Saving preprocessed train/test files ...")
        torch.save((filenames_train, inputs_train, labels_train, aux_info_train, user_embeds_train, user_metadata_train), args.path_train)
        torch.save((filenames_test, inputs_test, labels_test, aux_info_test, user_embeds_test, user_metadata_test), args.path_test)

    print("Done")
