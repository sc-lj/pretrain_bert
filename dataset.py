import torch
import os
from torch.utils.data import DataLoader, Dataset
from enum import IntEnum
from random import shuffle
import random
import collections
import glob
from text import mask, torch_long, PAD
from sources import PretrainingDataCreator, TokenInstance, GenericPretrainingDataCreator
from sources import WikiPretrainingDataCreator
from transformers.tokenization_bert import BertTokenizer
import pickle

class BatchType(IntEnum):
    PRETRAIN_BATCH = 0


class PretrainDataType(IntEnum):
    WIKIPEDIA = 1
    VALIDATION = 2
    MIDEADATA = 3

MaskedLMInstance = collections.namedtuple(
    "MaskedLMInstance", ["index", "label"])

PretrainBatch = collections.namedtuple(
    'PreTrainBatch', ['input_ids', 'input_mask', 'sequence_ids',
                      'is_next_label', 'masked_lm_output']
)

def get_random_partition(data_directory, index):
    partitions = [os.path.join(data_directory, x)
                  for x in os.listdir(data_directory)]
    partitions = sorted(partitions)
    i = index % len(partitions)
    return partitions[i]


def map_to_torch(encoding):
    encoding = torch_long(encoding)
    encoding.requires_grad_(False)
    return encoding


def map_to_torch_float(encoding):
    encoding = torch.FloatTensor(encoding)
    encoding.requires_grad_(False)
    return encoding


def map_to_torch_half(encoding):
    encoding = torch.HalfTensor(encoding)
    encoding.requires_grad_(False)
    return encoding


def encode_sequence(seqA, seqB, max_seq_len, tokenizer):
    seqA = ["[CLS]"] + seqA + ["[SEP]"]
    seqB = seqB + ["[SEP]"]

    input_tokens = seqA + seqB
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    sequence_ids = [0]*len(seqA) + [1]*len(seqB)
    input_mask = [1]*len(input_ids)

    while len(input_ids) < max_seq_len:
        input_ids.append(PAD)
        sequence_ids.append(PAD)
        input_mask.append(PAD)

    return (map_to_torch(input_ids), map_to_torch(input_mask), map_to_torch(sequence_ids))


def truncate_input_sequence(tokens_a, tokens_b, max_num_tokens):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

class PreTrainingDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, folder: str, logger, max_seq_length, index, data_type: PretrainDataType = PretrainDataType.WIKIPEDIA, max_predictions_per_seq=20, masked_lm_prob=0.15):
        self.tokenizer = tokenizer
        self.dir_path = folder
        self.max_seq_length = max_seq_length
        self.len = 0
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.vocab_words = list(tokenizer.vocab.keys())

        path = get_random_partition(self.dir_path, index)

        logger.info(f"Loading Pretraining Data from {path}")
        if data_type == PretrainDataType.WIKIPEDIA:
            self.data = GenericPretrainingDataCreator.load(path)
        elif data_type == PretrainDataType.VALIDATION:
            self.data = WikiPretrainingDataCreator.load(path)
        else:
            self.data = []
        self.len = len(self.data)
        logger.info(
            f"Data Loading Completed for Pretraining Data from {path} with {self.len} samples.")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        i = index % self.len

        instance: TokenInstance = self.data.instances[i]
        return self.create_training_instance(instance)

    def create_training_instance(self, instance: TokenInstance):
        tokens_a, tokens_b, is_next = instance.get_values()
        # print(f'is_next label:{is_next}')
        # Create mapper
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)

            tokens.append("[SEP]")
            segment_ids.append(1)

        # Get Masked LM predictions
        tokens, masked_lm_output = self.create_masked_lm_predictions(tokens)

        # Convert to Ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(PAD)
            segment_ids.append(PAD)
            input_mask.append(PAD)
            masked_lm_output.append(-1)
        return([map_to_torch([BatchType.PRETRAIN_BATCH]), map_to_torch(input_ids), map_to_torch(input_mask), map_to_torch(segment_ids), map_to_torch([is_next]), map_to_torch(masked_lm_output)])


    def create_masked_lm_predictions(self, tokens):
        cand_indexes = []
        for i, token in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            cand_indexes.append(i)

        random.shuffle(cand_indexes)
        output_tokens = list(tokens)

        num_to_predict = min(self.max_predictions_per_seq, max(
            1, int(round(len(tokens) * self.masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% mask
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% Keep Original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% replace w/ random word
                else:
                    masked_token = self.vocab_words[random.randint(0, len(self.vocab_words) - 1)]

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLMInstance(index=index, label=tokens[index]))

        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        masked_lm_output = [-1] * len(output_tokens)
        for p in masked_lms:
            masked_lm_output[p.index] = self.tokenizer.vocab[p.label]

        return (output_tokens, masked_lm_output)

class MideaDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, folder: str, max_seq_length, shuffle_numbers,
                 max_predictions_per_seq=20, masked_lm_prob=0.15,types="train"):
        super(MideaDataset, self).__init__()
        neg_filenames = glob.glob(os.path.join(folder, "all_instances_neg/*.pkl"))
        pos_filenames = glob.glob(os.path.join(folder, "all_instances_pos/*.pkl"))
        neg_per = int(len(neg_filenames) * 0.9)
        pos_per = int(len(pos_filenames) * 0.9)
        if types=='train':
            self.pos_filenames= pos_filenames[:pos_per]
            self.neg_filenames = neg_filenames[:neg_per]
        else:
            self.pos_filenames= pos_filenames[pos_per:]
            self.neg_filenames = neg_filenames[neg_per:]

        self.tokenizer = tokenizer
        self.dir_path = folder
        self.max_seq_length = max_seq_length
        self.pos_len = len(self.pos_filenames)
        self.neg_len = len(self.neg_filenames)
        self.len = max(self.pos_len, self.neg_len)
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.vocab_words = list(tokenizer.vocab.keys())
        self.pad_id = self.tokenizer.vocab.get("[PAD]")
        self.shuffle_numbers = shuffle_numbers


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        neg_i = index % self.neg_len
        pos_i = index % self.pos_len

        instances = []
        neg_instances: TokenInstance = self.read_pkl(self.neg_filenames[neg_i])
        shuffle(neg_instances)
        neg_instances = neg_instances[:self.shuffle_numbers]
        pos_instances: TokenInstance = self.read_pkl(self.pos_filenames[pos_i])
        shuffle(pos_instances)
        pos_instances = pos_instances[:self.shuffle_numbers]
        instances.extend(neg_instances)
        instances.extend(pos_instances)
        batchs = []
        for instance in instances:
            batchs.append(self.create_training_instance(instance))
        instances = [torch.cat(batch) for batch in zip(*batchs)]
        return instances

    def read_pkl(self,filename):
        with open(filename,'rb') as f:
            data = pickle.load(f)
        return data

    def create_training_instance(self, instance: TokenInstance):
        tokens_a, tokens_b, is_next = instance.get_values()
        # print(f'is_next label:{is_next}')
        # Create mapper
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)

            tokens.append("[SEP]")
            segment_ids.append(1)

        # Get Masked LM predictions
        tokens, masked_lm_output = self.create_masked_lm_predictions(tokens)

        # Convert to Ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(PAD)
            segment_ids.append(PAD)
            input_mask.append(PAD)
            masked_lm_output.append(-1)
        return([map_to_torch([BatchType.PRETRAIN_BATCH]).reshape(1,-1), map_to_torch(input_ids).reshape(1,-1), map_to_torch(input_mask).reshape(1,-1), map_to_torch(segment_ids).reshape(1,-1), map_to_torch([is_next]).reshape(1,-1), map_to_torch(masked_lm_output).reshape(1,-1)])


    def create_masked_lm_predictions(self, tokens):
        cand_indexes = []
        for i, token in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            cand_indexes.append(i)

        random.shuffle(cand_indexes)
        output_tokens = list(tokens)

        num_to_predict = min(self.max_predictions_per_seq, max(
            1, int(round(len(tokens) * self.masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% mask
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% Keep Original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% replace w/ random word
                else:
                    masked_token = self.vocab_words[random.randint(0, len(self.vocab_words) - 1)]

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLMInstance(index=index, label=tokens[index]))

        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        masked_lm_output = [-1] * len(output_tokens)

        for p in masked_lms:
            masked_lm_output[p.index] = self.tokenizer.vocab.get(p.label,self.pad_id)

        return (output_tokens, masked_lm_output)
