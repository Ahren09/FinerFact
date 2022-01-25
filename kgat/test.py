import argparse
import configparser
import logging
import os
import os.path as osp
import numpy as np
import torch
from transformers import BertTokenizer

from bert_model import BertForSequenceEncoder
from models import inference_model
from kgat.train import correct_prediction
from kgat.analysis import analyze_results
from utils.utils_case_study import load_case_study
from utils.utils_misc import set_args_from_config, get_eval_report, print_results
from utils.utils_preprocess import get_train_test_readers

logger = logging.getLogger(__name__)


def eval_model(model, validset_reader, results_eval=None, args=None, epoch=0, writer=None, counters_test=None,
               tokens_li=None):
    model.eval()
    correct_pred = 0.0
    preds_all, labs_all, logits_all, filenames_test_all = [], [], [], []

    for index, data in enumerate(validset_reader):
        inputs, lab_tensor, filenames_test, aux_info, user_embed = data

        prob, att_score_li = model(inputs, tokens_li, user_embed)

        correct_pred += correct_prediction(prob, lab_tensor)
        preds_all += prob.max(1)[1].tolist()
        logits_all += prob.tolist()
        labs_all += lab_tensor.tolist()
        filenames_test_all += filenames_test

    preds_np = np.array(preds_all)
    labs_np = np.array(labs_all)
    logits_np = np.array(logits_all)

    if counters_test is not None:
        analyze_results(labs_np, preds_np, counters_test, filenames_test_all, epoch, args)

    results = get_eval_report(labs_np, preds_np)
    print_results(results, epoch, args=args, dataset_split_name="Eval")

    # TODO!!

    if results_eval is not None:
        results_eval[epoch] = results
    dev_accuracy = correct_pred / validset_reader.total_num

    if writer is not None:
        writer.add_pr_curve('pr_curve', labels=labs_np, predictions=np.exp(logits_np)[:, 1], global_step=epoch)
        writer.add_scalar("Acc/Test", dev_accuracy, global_step=epoch)

    return dev_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', help='train path')
    parser.add_argument('--name', help='train path')
    parser.add_argument('--test_origin_path', help='train path')
    parser.add_argument("--batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument('--outdir', default=".", help='path to output directory')
    parser.add_argument("--min_evi_num", type=int, default=4,
                        help='Minimum evidence number for reasoning on social network')
    parser.add_argument('--bert_pretrain', default="../bert_base")
    parser.add_argument('--checkpoint', default="../../fake_news_data/models/P_K11.pt")
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument("--kfold_index", default=-1, type=int,
                        help="Run this for K-fold cross validation")
    parser.add_argument("--kernel", default=11, type=int,
                        help="Number of kernels")
    parser.add_argument("--sigma", default=1e-1, type=float,
                        help="Sigma value used")
    parser.add_argument('--case_study', action='store_true', help='Case study')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(osp.join("config", args.config_file))
    args = set_args_from_config(args, config)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S')
    logger.info(args)
    logger.info('Start testing!')

    label_map = {
        'real': 0,
        'fake': 1
    }

    args.num_labels = 2
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
    logger.info("loading validation set")

    if args.case_study:
        validset_reader, tokens_li = load_case_study(tokenizer, args)

    else:
        _, validset_reader = get_train_test_readers(label_map, tokenizer, args, test=True)

    logger.info('initializing estimator model')
    bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)

    if args.cuda:
        bert_model = bert_model.cuda()
    bert_model.eval()
    model = inference_model(bert_model, args, config)

    checkpoints = torch.load(args.checkpoint)

    model.load_state_dict(checkpoints['model'])

    if args.cuda:
        model = model.cuda()
    model.eval()

    results_eval = {}
    eval_model(model, validset_reader, results_eval=results_eval, args=args)

    model.eval()
