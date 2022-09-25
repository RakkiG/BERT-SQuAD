import collections
import json
import os
import six
import torch
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Vocab:
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    return Vocab(vocab_path)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


class BaseDataLoader:
    def __init__(self,
                 vocab_path='./vocab.txt',
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 split_sep='\n',
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True
                 ):
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']

        self.batch_size = batch_size
        self.split_sep = split_sep
        self.max_position_embeddings = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle


class SQuADDataLoader(BaseDataLoader):
    def __init__(self, doc_stride=64,
                 max_query_length=64,
                 n_best_size=20,
                 max_answer_length=30,
                 with_negative=False,
                 **kwargs):
        super(SQuADDataLoader, self).__init__(**kwargs)
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.with_negative = with_negative

    @staticmethod
    def get_format_text_and_word_offset(text):

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        for c in text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace: 
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c 
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        return doc_tokens, char_to_word_offset

    def preprocessing(self, filepath, is_training=True):
        """Read a SQuAD json file into a list of SquadExample."""
        with open(filepath, 'r') as f:
            input_data = json.loads(f.read())['data']
        examples = []
        for entry in tqdm(input_data, ncols=80, desc="load every paragraph"):
            for paragraph in entry['paragraphs']:
                context = paragraph['context']
                context_tokens, word_offset = self.get_format_text_and_word_offset(context)
                for qa in paragraph['qas']:
                    question_text = qa['question']
                    qas_id = qa['id']
                    is_impossible = False
                    if is_training:
                        if self.with_negative:
                            is_impossible = qa['is_impossible']
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = word_offset[answer_offset]
                            end_position = word_offset[answer_offset + answer_length - 1]
                            actual_text = " ".join(context_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(orig_answer_text.strip().split())
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.info("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""
                    else:
                        start_position = None
                        end_position = None
                        orig_answer_text = None
                    examples.append([qas_id, question_text, orig_answer_text,
                                     " ".join(context_tokens), start_position, end_position, is_impossible])
        return examples

    @staticmethod
    def improve_answer_span(context_tokens, answer_tokens, start_position, end_position):

        new_end = None
        for i in range(start_position, len(context_tokens)):
            if context_tokens[i] != answer_tokens[0]:
                continue
            for j in range(len(answer_tokens)):
                if answer_tokens[j] != context_tokens[i + j]:
                    break
                new_end = i + j
            if new_end - i + 1 == len(answer_tokens):
                return i, new_end
        return start_position, end_position

    @staticmethod
    def get_token_to_orig_map(input_tokens, origin_context, tokenizer):

        origin_context_tokens = origin_context.split()
        token_id = []
        str_origin_context = ""
        for i in range(len(origin_context_tokens)):
            tokens = tokenizer(origin_context_tokens[i])
            str_token = "".join(tokens)
            str_origin_context += "" + str_token
            for _ in str_token:
                token_id.append(i)

        key_start = input_tokens.index('[SEP]') + 1
        tokenized_tokens = input_tokens[key_start:-1]
        str_tokenized_tokens = "".join(tokenized_tokens)
        index = str_origin_context.index(str_tokenized_tokens)
        value_start = token_id[index]
        token_to_orig_map = {}
        # Handling edge cases like this： Building's gold   《==》   's', 'gold', 'dome'
        token = tokenizer(origin_context_tokens[value_start])
        for i in range(len(token), -1, -1):
            s1 = "".join(token[-i:])
            s2 = "".join(tokenized_tokens[:i])
            if s1 == s2:
                token = token[-i:]
                break

        while True:
            for j in range(len(token)):
                token_to_orig_map[key_start] = value_start
                key_start += 1
                if len(token_to_orig_map) == len(tokenized_tokens):
                    return token_to_orig_map
            value_start += 1
            token = tokenizer(origin_context_tokens[value_start])

    def data_process(self, filepath, is_training=False):

        data_path = filepath + '.pt'
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                logger.info("loading processed data {}".format(data_path))
                data = torch.load(f)
            return data

        logger.info("processing data {}".format(filepath))
        examples = self.preprocessing(filepath, is_training)
        all_data = []
        example_id, feature_id = 0, 1000000000

        for example in tqdm(examples, ncols=80, desc="load every example"):
            # [qas_id, question_text, orig_answer_text, context_tokens, start_position, end_position, impossible]
            question_tokens = self.tokenizer(example[1])

            if len(question_tokens) > self.max_query_length:
                question_tokens = question_tokens[:self.max_query_length]

            question_ids = [self.vocab[token] for token in question_tokens]
            question_ids = [self.CLS_IDX] + question_ids + [self.SEP_IDX]
            context_tokens = self.tokenizer(example[3])
            context_ids = [self.vocab[token] for token in context_tokens]

            start_position, end_position, answer_text = -1, -1, None
            if is_training:
                start_position, end_position = example[4], example[5]
                answer_text = example[2]
                if not example[-1]:
                    answer_tokens = self.tokenizer(answer_text)
                    start_position, end_position = self.improve_answer_span(context_tokens,
                                                                            answer_tokens,
                                                                            start_position,
                                                                            end_position)
            rest_len = self.max_sen_len - len(question_ids) - 1
            context_ids_len = len(context_ids)

            if context_ids_len > rest_len:  # over max_sen_len
                s_idx, e_idx = 0, rest_len
                while True:
                    # We can have documents that are longer than the maximum sequence length.
                    # To deal with this we do a sliding window approach, where we take chunks
                    # of the up to our max length with a stride of `doc_stride`.
                    tmp_context_ids = context_ids[s_idx:e_idx]
                    tmp_context_tokens = [self.vocab.itos[item] for item in tmp_context_ids]

                    input_ids = torch.tensor(question_ids + tmp_context_ids + [self.SEP_IDX])
                    input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + tmp_context_tokens + ['[SEP]']
                    seg = [0] * len(question_ids) + [1] * (len(input_ids) - len(question_ids))
                    seg = torch.tensor(seg)
                    if is_training:
                        new_start_position, new_end_position = 0, 0
                        if start_position >= s_idx and end_position <= e_idx:  # in train
                            new_start_position = start_position - s_idx
                            new_end_position = new_start_position + (end_position - start_position)

                            new_start_position += len(question_ids)
                            new_end_position += len(question_ids)

                        all_data.append([example_id, feature_id, input_ids, seg, new_start_position,
                                         new_end_position, answer_text, example[0], input_tokens])
                    else:
                        all_data.append([example_id, feature_id, input_ids, seg, start_position,
                                         end_position, answer_text, example[0], input_tokens])

                    token_to_orig_map = self.get_token_to_orig_map(input_tokens, example[3], self.tokenizer)
                    all_data[-1].append(token_to_orig_map)

                    feature_id += 1
                    if e_idx >= context_ids_len:
                        break
                    s_idx += self.doc_stride
                    e_idx += self.doc_stride

            else:
                input_ids = torch.tensor(question_ids + context_ids + [self.SEP_IDX])
                input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
                seg = [0] * len(question_ids) + [1] * (len(input_ids) - len(question_ids))
                seg = torch.tensor(seg)
                if is_training:
                    if not example[-1]:
                        start_position += (len(question_ids))
                        end_position += (len(question_ids))
                    else:
                        start_position = 0
                        end_position = 0
                token_to_orig_map = self.get_token_to_orig_map(input_tokens, example[3], self.tokenizer)
                all_data.append([example_id, feature_id, input_ids, seg, start_position,
                                 end_position, answer_text, example[0], input_tokens, token_to_orig_map])
                feature_id += 1
            example_id += 1
        #  all_data[0]: [origin id, feature id, input_ids, seg, start, end, answer text, question id, input_tokens,ori_map]
        data = {'all_data': all_data, 'max_len': self.max_sen_len, 'examples': examples}
        with open(data_path, 'wb') as f:
            torch.save(data, f)
        return data

    def generate_batch(self, data_batch):
        torch.manual_seed(2021)
        batch_input, batch_seg, batch_label, batch_qid = [], [], [], []
        batch_example_id, batch_feature_id, batch_map = [], [], []
        for item in data_batch:
            # item: [origin id, feature id, input_ids, seg, start, end, answer text, question id, input_tokens,ori_map]
            batch_example_id.append(item[0])  # origin id
            batch_feature_id.append(item[1])  # feature id
            batch_input.append(item[2])  # input_ids
            batch_seg.append(item[3])  # seg
            batch_label.append([item[4], item[5]])  # start pos, end pos
            batch_qid.append(item[7])  # question id
            batch_map.append(item[9])  # ori_map

        batch_input = pad_sequence(batch_input,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)  # [max_len,batch_size]
        batch_seg = pad_sequence(batch_seg,  # [batch_size,max_len]
                                 padding_value=self.PAD_IDX,
                                 batch_first=False,
                                 max_len=self.max_sen_len)  # [max_len, batch_size]
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        # [max_len,batch_size] , [max_len, batch_size] , [batch_size,2], [batch_size,], [batch_size,]
        return batch_input, batch_seg, batch_label, batch_qid, batch_example_id, batch_feature_id, batch_map

    def load_train_val_test_data(self, train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=True):
        data = self.data_process(filepath=test_file_path,
                                 is_training=False)
        test_data, examples = data['all_data'], data['examples']
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False,
                               collate_fn=self.generate_batch)
        if only_test:
            logger.info(f"loaded test dataset, {len(test_iter.dataset)} instances")
            return test_iter, examples

        data = self.data_process(filepath=train_file_path,
                                 is_training=True)
        train_data, max_sen_len = data['all_data'], data['max_len']
        _, val_data = train_test_split(train_data, test_size=0.3, random_state=2021)
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        train_iter = DataLoader(train_data, batch_size=self.batch_size, num_workers=0,
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        logger.info(f"loaded train dataset ({len(train_iter.dataset)}) instances、val dataset "
                    f"{len(val_iter.dataset)}) instances, test dataset {len(test_iter.dataset)} instances.")
        return train_iter, test_iter, val_iter

    @staticmethod
    def get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        # logits = [0.37203778 0.48594432 0.81051651 0.07998148 0.93529721 0.0476721
        #  0.15275263 0.98202781 0.07813079 0.85410559]
        # n_best_size = 4
        # return [7, 4, 9, 2]
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def get_final_text(self, pred_text, orig_text):
        """Project the tokenized prediction back to the original text."""

        # ref: https://github.com/google-research/bert/blob/master/run_squad.py
        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heruistic between
        # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.

        tok_text = " ".join(self.tokenizer(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

    def write_prediction(self, test_iter, all_examples, logits_data, output_dir):
        """
        :param test_iter:
        :param all_examples:
        :param logits_data:
        :return:
        """
        logger.info("Writing predictions to: %s" % (output_dir + 'best_result.json'))
        logger.info("Writing nbest to: %s" % (output_dir + 'best_n_result.json'))
        logger.info("Writing scores_diff to: %s" % (output_dir + 'scores_diff.json'))
        qid_to_example_context = {} 
        qid_to_null = {}
        for example in all_examples:
            context = example[3]
            context_list = context.split()
            qid_to_example_context[example[0]] = context_list
        _PrelimPrediction = collections.namedtuple(
            "PrelimPrediction",
            ["text", "start_index", "end_index", "start_logit", "end_logit"])
        prelim_predictions = collections.defaultdict(list)
        for b_input, _, _, b_qid, _, b_feature_id, b_map in tqdm(test_iter, ncols=80, desc="writing prediction"):
            score_null = 1000000  # large and positive
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score

            all_logits = logits_data[b_qid[0]]
            for logits in all_logits:
                if logits[0] != b_feature_id[0]:
                    continue  

                start_indexes = self.get_best_indexes(logits[1], self.n_best_size)
                # Get the index corresponding to the value with the highest probability of starting position, it may be [ 4,6,3,1]
                end_indexes = self.get_best_indexes(logits[2], self.n_best_size)
                # Get the index corresponding to the value with the highest probability of ending position, it may be [ 5,8,10,9]
                if self.with_negative:
                    feature_null_score = logits[1][0] + logits[2][0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        null_start_logit = logits[1][0]
                        null_end_logit = logits[2][0]
                for start_index in start_indexes:
                    for end_index in end_indexes: 
                        if start_index >= b_input.size(0):
                            continue  # start index is greater than the token length, ignored
                        if end_index >= b_input.size(0):
                            continue  # end index is greater than the token length, ignored
                        if start_index not in b_map[0]:
                            continue  # determine whether the index is located after [SEP]
                        if end_index not in b_map[0]:
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self.max_answer_length:
                            continue
                        token_ids = b_input.transpose(0, 1)[0]
                        strs = [self.vocab.itos[s] for s in token_ids]
                        tok_text = " ".join(strs[start_index:(end_index + 1)])
                        tok_text = tok_text.replace(" ##", "").replace("##", "")
                        tok_text = tok_text.strip()
                        tok_text = " ".join(tok_text.split())

                        orig_doc_start = b_map[0][start_index]
                        orig_doc_end = b_map[0][end_index]
                        orig_tokens = qid_to_example_context[b_qid[0]][orig_doc_start:(orig_doc_end + 1)]
                        orig_text = " ".join(orig_tokens)
                        final_text = self.get_final_text(tok_text, orig_text)

                        prelim_predictions[b_qid[0]].append(_PrelimPrediction(
                            text=final_text,
                            start_index=int(start_index),
                            end_index=int(end_index),
                            start_logit=float(logits[1][start_index]),
                            end_logit=float(logits[2][end_index])))

            if self.with_negative:
                qid_to_null[b_qid[0]] = score_null
                prelim_predictions[b_qid[0]].append(_PrelimPrediction(
                    text="",
                    start_index=int(0),
                    end_index=int(0),
                    start_logit=float(null_start_logit),
                    end_logit=float(null_end_logit)))

                if len(prelim_predictions[b_qid[0]]) == 1:
                    logger.info('No predict answer, writing empty.')
                    prelim_predictions[b_qid[0]].append(_PrelimPrediction(
                        text="empty",
                        start_index=int(0),
                        end_index=int(0),
                        start_logit=float(0.0),
                        end_logit=float(0.0)))

        for k, v in prelim_predictions.items():
            prelim_predictions[k] = sorted(prelim_predictions[k],
                                           key=lambda x: (x.start_logit + x.end_logit),
                                           reverse=True)
        best_results, all_n_best_results, scores_diff_json = {}, {}, {}
        for k, v in prelim_predictions.items():
            if not self.with_negative:
                best_results[k] = v[0].text  # the best answer
            else:
                best_non_null_entry = None
                for entry in v:
                    if not best_non_null_entry:
                        if entry.text:
                            best_non_null_entry = entry
                            break

                score_diff = qid_to_null[k] - best_non_null_entry.start_logit - best_non_null_entry.end_logit
                scores_diff_json[k] = score_diff
                if score_diff > 0.0:
                    best_results[k] = ""
                    logger.info(f'result {k} is null, the score_diff: {score_diff}')
                else:
                    best_results[k] = v[0].text
            all_n_best_results[k] = v  
        with open(os.path.join(output_dir, f"best_result.json"), 'w') as f:
            f.write(json.dumps(best_results, indent=4) + '\n')
        with open(os.path.join(output_dir, f"scores_diff.json"), 'w') as f:
            f.write(json.dumps(scores_diff_json, indent=4) + '\n')
        with open(os.path.join(output_dir, f"best_n_result.json"), 'w') as f:
            f.write(json.dumps(all_n_best_results, indent=4) + '\n')

