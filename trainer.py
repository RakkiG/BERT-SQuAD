import collections
import os
import time
import logging

import torch
from tqdm import tqdm
from transformers import BertTokenizer, get_scheduler

from models.bert_qa import BertQA
from models.data_loader import SQuADDataLoader
from models.BasicBert.BertConfig import BertConfig

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def show_result(batch_input, itos, num_show=5, y_pred=None, y_true=None):
    count = 0
    batch_input = batch_input.transpose(0, 1)
    for i in range(len(batch_input)):
        if count == num_show:
            break
        input_tokens = [itos[s] for s in batch_input[i]]
        start_pos, end_pos = y_pred[0][i], y_pred[1][i]
        answer_text = " ".join(input_tokens[start_pos:(end_pos + 1)]).replace(" ##", "")
        input_text = " ".join(input_tokens).replace(" ##", "").split('[SEP]')
        question_text, context_text = input_text[0], input_text[1]

        logger.info(f"### Question: {question_text}")
        logger.info(f"  ## Predicted answer: {answer_text}")
        start_pos, end_pos = y_true[0][i], y_true[1][i]
        true_answer_text = " ".join(input_tokens[start_pos:(end_pos + 1)])
        true_answer_text = true_answer_text.replace(" ##", "")
        logger.info(f"  ## True answer: {true_answer_text}")
        logger.info(f"  ## True answer idx: {start_pos.cpu(), end_pos.cpu()}")
        count += 1

def train(args, device):
    bert_config_path = os.path.join(args.bert_dir, "config.json")
    bert_config = BertConfig.from_json_file(bert_config_path)
    model = BertQA(args, bert_config)
    if os.path.exists(args.model_path):
        loaded_paras = torch.load(args.model_path)
        model.load_state_dict(loaded_paras)
        logger.info('Loading checkpoint from %s' % args.model_path)
    model = model.to(device)

    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(args.bert_dir).tokenize
    data_loader = SQuADDataLoader(vocab_path=os.path.join(args.bert_dir, 'vocab.txt'),
                                  tokenizer=bert_tokenize,
                                  batch_size=args.batch_size,
                                  max_sen_len=args.max_sen_len,
                                  max_query_length=args.max_query_len,
                                  max_position_embeddings=args.max_position_embeddings,
                                  is_sample_shuffle=args.is_sample_shuffle,
                                  doc_stride=args.doc_stride,
                                  with_negative=args.version_2_with_negative)
    train_iter, test_iter, val_iter = \
        data_loader.load_train_val_test_data(train_file_path=os.path.join(args.data_path, 'train-' + args.dataset),
                                             test_file_path=os.path.join(args.data_path, 'dev-' + args.dataset),
                                             only_test=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(name='linear',
                                 optimizer=optimizer,
                                 num_warmup_steps=int(len(train_iter) * 0),
                                 num_training_steps=int(args.epochs * len(train_iter)))
    max_acc = 0
    for epoch in range(args.epochs):
        losses = 0
        start_time = time.time()
        for idx, (batch_input, batch_seg, batch_label, _, _, _, _) in enumerate(train_iter):
            batch_input = batch_input.to(device)
            batch_seg = batch_seg.to(device)
            batch_label = batch_label.to(device)
            padding_mask = (batch_input == data_loader.PAD_IDX).transpose(0, 1)
            loss, start_logits, end_logits = model(input_ids=batch_input,
                                                   attention_mask=padding_mask,
                                                   token_type_ids=batch_seg,
                                                   position_ids=None,
                                                   start_positions=batch_label[:, 0],
                                                   end_positions=batch_label[:, 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            losses += loss.item()
            acc_start = (start_logits.argmax(1) == batch_label[:, 0]).float().mean()
            acc_end = (end_logits.argmax(1) == batch_label[:, 1]).float().mean()
            acc = (acc_start + acc_end) / 2
            if idx % 10 == 0:
                logger.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                            f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}, Cost time: {time.time()-start_time:6.0f}")
            if idx % 100 == 0:
                y_pred = [start_logits.argmax(1), end_logits.argmax(1)]
                y_true = [batch_label[:, 0], batch_label[:, 1]]
                show_result(batch_input, data_loader.vocab.itos,
                            y_pred=y_pred, y_true=y_true)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logger.info(f"Epoch: {epoch}, Train loss: "
                    f"{train_loss:.3f}, Epoch time = {(end_time - start_time):6.0f}s")
        if (epoch + 1) % args.model_val_per_epoch == 0:
            acc = evaluate(val_iter, model,
                           device,
                           data_loader.PAD_IDX,
                           inference=False)
            if acc > max_acc:
                max_acc = acc
            logger.info(f"Accuracy on val: {round(acc, 4)} max :{max_acc}")
            torch.save(model.state_dict(), args.model_path)


def evaluate(data_iter, model, device, PAD_IDX, inference=False):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        all_results = collections.defaultdict(list)
        for batch_input, batch_seg, batch_label, batch_qid, _, batch_feature_id, _ in tqdm(data_iter, ncols=80, desc="evaluating"):
            batch_input = batch_input.to(device)
            batch_seg = batch_seg.to(device)
            batch_label = batch_label.to(device)
            padding_mask = (batch_input == PAD_IDX).transpose(0, 1)
            _, start_logits, end_logits = model(input_ids=batch_input,
                                                              attention_mask=padding_mask,
                                                              token_type_ids=batch_seg,
                                                              position_ids=None)

            all_results[batch_qid[0]].append([batch_feature_id[0],
                                              start_logits.cpu().numpy().reshape(-1),
                                              end_logits.cpu().numpy().reshape(-1)])
            if not inference:
                acc_sum_start = (start_logits.argmax(1) == batch_label[:, 0]).float().sum().item()
                acc_sum_end = (end_logits.argmax(1) == batch_label[:, 1]).float().sum().item()
                acc_sum += (acc_sum_start + acc_sum_end)
                n += len(batch_label)
        model.train()
        if inference:
            return all_results
        return acc_sum / (2 * n)

def inference(args, device):
    bert_tokenize = BertTokenizer.from_pretrained(args.bert_dir).tokenize
    data_loader = SQuADDataLoader(vocab_path=os.path.join(args.bert_dir, 'vocab.txt'),
                                  tokenizer=bert_tokenize,
                                  batch_size=1,
                                  max_sen_len=args.max_sen_len,
                                  doc_stride=args.doc_stride,
                                  max_query_length=args.max_query_len,
                                  max_answer_length=args.max_answer_len,
                                  max_position_embeddings=args.max_position_embeddings,
                                  n_best_size=args.n_best_size,
                                  with_negative=args.version_2_with_negative)
    test_iter, all_examples = data_loader.load_train_val_test_data(test_file_path=os.path.join(args.data_path, 'dev-' + args.dataset),
                                                                   only_test=True)
    bert_config_path = os.path.join(args.bert_dir, "config.json")
    bert_config = BertConfig.from_json_file(bert_config_path)
    model = BertQA(args, bert_config)
    if os.path.exists(args.model_path):
        loaded_paras = torch.load(args.model_path)
        model.load_state_dict(loaded_paras)
        logger.info(f"Loading checkpoint from {args.model_path}, start inferencing...")
    else:
        raise ValueError(f"model {args.model_path} is not existed")

    model = model.to(device)
    all_result_logits = evaluate(test_iter, model, device,
                                 data_loader.PAD_IDX, inference=True)
    data_loader.write_prediction(test_iter, all_examples,
                                 all_result_logits, args.data_path)
