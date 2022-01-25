import argparse
import configparser
import datetime
import logging
import math
import os
import os.path as osp
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW

# import fitlog

sys.path.append("..")

from models import inference_model
from bert_model import BertForSequenceEncoder
from analysis import analyze_results
from utils.utils_misc import get_eval_report, print_results, set_args_from_config, save_results_to_tsv
from utils.utils_preprocess import get_train_test_readers

logger = logging.getLogger(__name__)


def get_raw_dir(args):
    """
    Archived function. Ignore this for now
    """
    root = "C:\\Workspace\\FakeNews"
    if os.name == "posix":
        root = '..'
    path = osp.join(root, "Demo", "data", f"{args.dataset}", "raw")

    return path


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def correct_prediction(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct


def cuda_memory_info():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print(f'total    : {t}')
    print(f'free     : {f}')
    print(f'used     : {a}')


def eval_model(model, validset_reader, results_eval=None, args=None, epoch=0, writer=None, counters_test=None):
    model.eval()
    correct_pred = 0.0
    preds_all, labs_all, logits_all, probs_all, filenames_test_all = [], [], [], [], []

    for index, data in enumerate(validset_reader):
        inputs, lab_tensor, filenames_test, aux_info, user_metadata = data

        logits, prob = model(inputs, aux_info, user_metadata)

        correct_pred += correct_prediction(logits, lab_tensor)
        preds_all += logits.max(1)[1].tolist()
        probs_all += prob[:, 1].tolist()
        logits_all += logits.tolist()
        labs_all += lab_tensor.tolist()
        filenames_test_all += filenames_test
        # print(f"\t {str(prob)}, {str(prob.max(1)[1].tolist())}, {str(lab_tensor)}")

    preds_np = np.array(preds_all)  # .ravel()
    probs_np = np.array(probs_all)  # .ravel()
    labs_np = np.array(labs_all)  # .ravel()
    logits_np = np.array(logits_all)

    if counters_test is not None:
        analyze_results(labs_np, preds_np, counters_test, filenames_test_all, epoch, args)

    results = get_eval_report(labs_np, preds_np, probs_np)
    print_results(results, epoch, args=args, dataset_split_name="Eval")

    # TODO!!

    if results_eval is not None:
        results_eval[epoch] = results
    dev_accuracy = correct_pred / validset_reader.total_num

    if writer is not None:
        writer.add_pr_curve('pr_curve', labels=labs_np, predictions=np.exp(logits_np)[:, 1], global_step=epoch)
        writer.add_scalar("Acc/Test", dev_accuracy, global_step=epoch)

    return dev_accuracy



def train_model(model, ori_model, args, trainset_reader, validset_reader, writer, experiment_name):
    save_path = args.outdir + '/model'
    best_accuracy = 0.0
    running_loss = 0.0
    t_total = int(
        trainset_reader.total_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate)

    args.warmup_steps = math.ceil(t_total * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    global_step = 0

    # Initialize analysis tools

    # Summarize results
    counters_test = [Counter(), Counter(), Counter(), Counter()]
    counters_train = [Counter(), Counter(), Counter(), Counter()]

    # Dict to store results
    results_train, results_eval = {}, {}

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # dev_accuracy = eval_model(model, validset_reader, results_eval=results_eval, args=args, epoch=0, counters_test=counters_test)

    tr_loss, logging_loss = 0.0, 0.0
    for epoch in range(int(args.num_train_epochs)):
        model.train()
        optimizer.zero_grad()
        preds_all, labs_all, probs_all, filenames_train_all = [], [], [], []

        labels_count = Counter(trainset_reader.labels)
        # ratio = float(labels_count[1]) / labels_count[0]
        # weight_tensor = torch.tensor([ratio, 1.]).to(device)

        for index, data in enumerate(trainset_reader):

            # inputs, lab_tensor, filenames_train, aux_info, user_embed = data
            inputs, lab_tensor, filenames_train, aux_info, user_metadata = data
            logits, prob = model(inputs, aux_info, user_metadata)

            loss = F.nll_loss(logits, lab_tensor.to(device))
            # loss = F.nll_loss(prob, lab_tensor.to(model.device), weight=weight_tensor)

            preds_all += logits.max(1)[1].tolist()
            probs_all += prob[:, 1].tolist()

            # preds_all += [prob.max(1)[1].cpu().numpy()]
            labs_all += lab_tensor.tolist()
            filenames_train_all += filenames_train

            running_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            global_step += 1
            tr_loss += loss.item()
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()  # optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # fitlog.add_loss(loss, name="Loss", step=global_step)

                logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, global_step, (running_loss / global_step)))
                if writer is not None:
                    writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar("Loss/train", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

        logger.info('Start eval!')
        analyze_results(labs_all, preds_all, counters_train, filenames_train_all, epoch, args)
        with torch.no_grad():
            dev_accuracy = eval_model(model, validset_reader, results_eval=results_eval, args=args, epoch=epoch, counters_test=counters_test)
            logger.info('Dev acc: {0}'.format(dev_accuracy))
            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy
                if not osp.exists(save_path):
                    os.mkdir(save_path)

                torch.save({
                    'epoch': epoch,
                    'model': ori_model.state_dict(),
                    'best_accuracy': best_accuracy
                }, f"{save_path}/{experiment_name}.pt")
                logger.info("Saved best epoch {0}, best accuracy {1}".format(epoch, best_accuracy))
            if (epoch + 1) % 10 == 0:
                if not osp.exists(osp.join("..", "logs", "stats")):
                    os.mkdir(osp.join("..", "logs", "stats"))
                torch.save((counters_train, counters_test),
                           osp.join("..", "logs", "stats", f"{args.prefix}{args.sample_suffix}_{epoch + 1}.pt"))
                save_results_to_tsv(results_train, results_eval, experiment_name, args)
        # ------------------------------------------
        # Get eval results
        # ------------------------------------------
        preds_all = np.hstack(preds_all)
        labs_all = np.hstack(labs_all)
        results = get_eval_report(labs_all, preds_all, probs_all)
        results_train[epoch] = results
        print_results(results, epoch, args=args, dataset_split_name="Train")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience', type=int, default=20, help='Patience')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=2)

    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')

    parser.add_argument("--eval_step", default=500, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    # ------------------------------------------
    # Additional args for FND
    # ------------------------------------------

    parser.add_argument("--min_evi_num", type=int, default=4,
                        help='Minimum evidence number for reasoning on social network')

    parser.add_argument("--seed", default=21, type=int,
                        help="Random state")
    parser.add_argument("--kernel", default=11, type=int,
                        help="Number of kernels")
    parser.add_argument("--sigma", default=1e-1, type=float,
                        help="Sigma value used")
    parser.add_argument("--root", default='../Demo', type=str,
                        help="")
    parser.add_argument("--kfold_index", default=-1, type=int,
                        help="Run this for K-fold cross validation")

    parser.add_argument('--debug', action='store_true', help='Debug')

    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--postpretrain')
    parser.add_argument('--itp', action="store_true")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # ------------------------------------------
    # Config
    # ------------------------------------------

    config = configparser.ConfigParser(allow_no_value=True)

    args = parser.parse_args()

    # train on DLTS or ITP
    if args.itp:
        os.chdir(osp.join("/home", "v-yiqiaojin", "KernelGAT", "kgat"))

    config = configparser.ConfigParser()
    config.read(osp.join("config", args.config_file))

    args = set_args_from_config(args, config)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    timestr = datetime.datetime.now().strftime("%m-%d_%H")
    experiment_name = f"{'P' if args.dataset == 'politifact' else 'G'}_{timestr}_{args.mode}_batch{args.train_batch_size}_{args.num_train_epochs}_lr{args.learning_rate}_K{args.kernel}_Sig{args.sigma}{'_KFold' + str(args.kfold_index) if args.kfold_index >= 0 else ''}_{args.sample_suffix}"

    if not osp.exists(osp.join("..", "logs", experiment_name)):
        os.mkdir(osp.join("..", "logs", experiment_name))

    handlers = [logging.FileHandler(osp.join("..", "logs", experiment_name, f"log_{experiment_name}.txt")),
                logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logger.info(f'{args.dataset} Start training!')
    logger.info(f'Using batch size {args.train_batch_size} | accumulation {args.gradient_accumulation_steps}')

    label_map = {
        'real': 0,
        'fake': 1
    }
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)

    trainset_reader, validset_reader = get_train_test_readers(label_map, tokenizer, args)

    # ------------------------------------------
    # Loading BERT model
    # ------------------------------------------

    logger.info('Initializing BERT model')
    bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
    if args.postpretrain:
        model_dict = bert_model.state_dict()
        pretrained_dict = torch.load(args.postpretrain)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
    ori_model = inference_model(bert_model, args, config, tokenizer=tokenizer)

    # ------------------------------------------
    # Visualize model
    # ------------------------------------------

    if args.enable_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        logger.info("Initializing Tensorboard")
        writer = SummaryWriter()
        vis_data = trainset_reader.next()
        writer.add_graph(ori_model.device, (vis_data[0], vis_data[2], None))
        writer.flush()
    else:
        writer = None

    # ------------------------------------------
    # Train model
    # ------------------------------------------

    # logger.info("Initializing Fitlog")
    # log_dir = "logs/"
    # if not osp.exists(log_dir):
    #     os.mkdir(log_dir)
    # fitlog.set_log_dir(log_dir)
    #
    # fitlog.add_hyper(args)
    # fitlog.add_hyper_in_file(__file__)

    if args.cuda:

        model = nn.DataParallel(ori_model)
        model = model.cuda()
    else:
        model = ori_model

    train_model(model, ori_model, args, trainset_reader, validset_reader, writer=writer,
                experiment_name=experiment_name)
