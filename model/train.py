from utils import LOG
import json
import os
import numpy as np
import data_process
import torch
import random
from torch.utils.data import DataLoader
from transformers import AutoConfig
from evaluation import extract_relation_emb, evaluate
from model import PPAG
from transformers import get_linear_schedule_with_warmup
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--margin", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--num_negsample", type=int, default=7)
parser.add_argument("--warm_up", type=float, default=0.1)
parser.add_argument("--unseen", type=int, default=15)
parser.add_argument("--dataset_path", type=str, default='../data')
parser.add_argument("--dataset", type=str, default='fewrel', choices=['fewrel', 'wikizsl'],
                    help='original dataset')
parser.add_argument("--rel2id_path", type=str, default='../data/rel2id')
parser.add_argument("--rel_split_seed", type=str, default='ori')
parser.add_argument("--relation_description", type=str, default='fewrel_property.json')
parser.add_argument("--visible_device", type=str, default='0')
parser.add_argument("--pretrained_model", type=str, default='../../BERT_MODELS/bert-base-uncased')
args = parser.parse_args()

args.ckpt_save_path = f'../ckpt/{args.dataset}_split_{args.rel_split_seed}_unseen_{str(args.unseen)}'

args.dataset_file = os.path.join(args.dataset_path, args.dataset, f'{args.dataset}_dataset.json')
args.relation_description_file = os.path.join(args.dataset_path, args.dataset, 'relation_description',
                                              args.relation_description)
args.rel2id_file = os.path.join(args.rel2id_path, f'{args.dataset}_rel2id',
                                f'{args.dataset}_rel2id_{str(args.unseen)}_{args.rel_split_seed}.json')

log = LOG(args)

log.logging('ckpt_save_path:' + args.ckpt_save_path)
log.logging('dataset_file_path : ' + args.dataset_file)
log.logging('relation_description_file_path : ' + args.relation_description_file)
log.logging('rel2id_file_path : ' + args.rel2id_file)
log.logging('margin : ' + str(args.margin))
log.logging('epochs_total : ' + str(args.epochs))
log.logging('lr : ' + str(args.lr))

os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_device


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

with open(args.rel2id_file, 'r', encoding='utf-8') as r2id:
    relation2idx = json.load(r2id)
    train_relation2idx, test_relation2idx = relation2idx['train'], relation2idx['test']
    train_idx2relation, test_idx2relation, = dict((v, k) for k, v in train_relation2idx.items()), \
        dict((v, k) for k, v in test_relation2idx.items())

train_label, test_label = list(train_relation2idx.keys()), list(test_relation2idx.keys())

# load rel_description
with open(args.relation_description_file, 'r', encoding='utf-8') as rd:
    relation_desc = json.load(rd)
    train_desc = [i for i in relation_desc if i['relation'] in train_label]
    test_desc = [i for i in relation_desc if i['relation'] in test_label]

# load data
with open(args.dataset_file, 'r', encoding='utf-8') as d:
    raw_data = json.load(d)
    training_data = [i for i in raw_data if i['relation'] in train_label]
    test_data = [i for i in raw_data if i['relation'] in test_label]

# print info
log.logging(
    f'args_config: , lr:{args.lr},seed:{args.seed},epochs:{args.epochs}')
log.logging('there are {} kinds of relation in test.'.format(len(set(test_label))))
log.logging('the lengths of test data is {} '.format(len(test_data)))

# load description
train_rel2vec, test_rel2vec = data_process.generate_attribute(args, train_desc, test_desc)

# load model
config = AutoConfig.from_pretrained(args.pretrained_model, num_labels=len(set(train_label)))
config.pretrained_model = args.pretrained_model
config.margin = args.margin
model = PPAG.from_pretrained(args.pretrained_model, config=config)
model = model.cuda()

trainset = data_process.FewRelDataset(args, 'train', training_data, train_rel2vec, train_relation2idx)
trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=data_process.create_mini_batch, shuffle=True)

# To evaluate the inference time
test_batchsize = 10 * args.unseen

testset = data_process.FewRelDataset(args, 'test', test_data, test_rel2vec, test_relation2idx)
testloader = DataLoader(testset, batch_size=test_batchsize,
                        collate_fn=data_process.create_mini_batch, shuffle=False)

train_y_attr, test_y_attr, test_y, test_y_e1, test_y_e2, train_y_e1, train_y_e2 = [], [], [], [], [], [], []

for i, test in enumerate(test_data):
    label = int(test_relation2idx[test['relation']])
    test_y.append(label)

for i in test_label:
    test_y_attr.append(test_rel2vec[i])

for i in train_label:
    train_y_attr.append(train_rel2vec[i])

train_y_attr, train_y_e1, train_y_e2 = np.array(train_y_attr), np.array(train_y_e1), np.array(train_y_e2)
test_y, test_y_attr, test_y_e1, test_y_e2 = np.array(test_y), np.array(test_y_attr), np.array(test_y_e1), np.array(
    test_y_e2)

# optimizer and scheduler
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
num_training_steps = len(trainset) * args.epochs // args.batch_size
warmup_steps = num_training_steps * args.warm_up
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

best_pt, best_rt, best_f1t = 0.0, 0.0, 0.0
test_pt, test_rt, test_f1t = 0.0, 0.0, 0.0
for epoch in range(args.epochs):
    log.logging(f'============== TRAIN ON THE {epoch + 1}-th EPOCH ==============')
    running_loss = 0.0
    out_sentence_embs = None
    e1_hs = None
    e2_hs = None
    train_y = None
    for step, data in enumerate(trainloader):
        input_ids, attention_mask, token_type_ids, mark_head_mask, mark_tail_mask, mark_relation_mask, relation_emb, labels_ids = [
            t.cuda() for t in data]

        optimizer.zero_grad()

        outputs, relation_mask, e1_mask, e2_mask = model(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids,
                                                         mark_head_mask=mark_head_mask,
                                                         mark_tail_mask=mark_tail_mask,
                                                         mark_relation_mask=mark_relation_mask,
                                                         input_relation_emb=relation_emb,
                                                         labels=labels_ids,
                                                         num_neg_sample=args.num_negsample
                                                         )

        loss = outputs.sum()
        loss = loss / args.batch_size

        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        if step % 100 == 0:
            log.logging(f'[step {step}]' + '=' * (step // 100))
            log.logging('running_loss:{}'.format(running_loss / (step + 1)))

    # if epoch == args.epochs - 1:
    log.logging('============== EVALUATION ON Test DATA ==============')

    preds, e1_hs, e2_hs = extract_relation_emb(model, testloader)
    test_pt, test_rt, test_f1t = evaluate(args, preds.cpu(), e1_hs.cpu(), e2_hs.cpu(), test_y_attr, test_y,
                                          test_idx2relation)

    if test_f1t > best_f1t:
        best_pt, best_rt, best_f1t = test_pt, test_rt, test_f1t
        torch.save(model.state_dict(), args.ckpt_save_path + f'_f1_{test_f1t}')
    log.logging(f'[test] precision: {test_pt:.4f}, recall: {test_rt:.4f}, f1 score: {test_f1t:.4f}')
    log.logging("* " * 20)
log.logging(f'[test] final precision: {best_pt:.4f}, recall: {best_rt:.4f}, f1 score: {best_f1t:.4f}')
