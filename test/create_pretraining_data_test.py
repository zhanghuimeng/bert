import unittest
import tensorflow as tf
import random

import tokenization
from create_pretraining_data import create_training_instances
from create_pretraining_data import create_masked_lm_predictions
from create_pretraining_data import TrainingInstance


class TestCreateTrainingInstances(unittest.TestCase):
    VOCAB_FILE = "vocab.txt"
    INPUT_FILE = "sample.txt"
    # 这是参考README里Pretraining一段的参数制定出来的
    MAX_SEQ_LENGTH = 128
    DUPE_FACTOR = 5
    SHORT_SEQ_PROB = 0.1
    MASKED_LM_PROB = 0.15
    MAX_PREDICTIONS_PER_SEQ = 20

    def test_read(self):
        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.VOCAB_FILE, do_lower_case=True)

        input_files = []
        for input_pattern in self.INPUT_FILE.split(","):
            input_files.extend(tf.gfile.Glob(input_pattern))

        tf.logging.info("*** Reading from input files ***")
        for input_file in input_files:
            tf.logging.info("  %s", input_file)

        rng = random.Random(12345)  # random seed
        instances = create_training_instances(
            input_files, tokenizer, self.MAX_SEQ_LENGTH, self.DUPE_FACTOR,
            self.SHORT_SEQ_PROB, self.MASKED_LM_PROB, self.MAX_PREDICTIONS_PER_SEQ,
            rng)

        for instance in instances:
            print(instance)


class TestCreateMaskedLmPredictions(unittest.TestCase):
    VOCAB_FILE = "vocab.txt"
    MASKED_LM_PROB = 0.15
    MAX_PREDICTIONS_PER_SEQ = 20
    TOKEN_SAMPLE = [
        '[CLS]', 'we', 'are', 'releasing', 'code', 'to', 'do', '"', 'mask', '##ed',
        'l', '##m', '"', 'and', '"', 'next', 'sentence', 'prediction', '"', 'on',
        'an', 'ar', '##bit', '##rary', 'text', 'corpus', '.', '[SEP]', 'that',
        'this', 'is', 'not', 'the', 'exact', 'code', 'that', 'was', 'used', 'for',
        'the', 'paper', '.', 'but', 'this', 'code', 'does', 'generate', 'pre', '-',
        'training', 'data', 'as', 'described', 'in', 'the', 'paper', '.', 'here',
        "'", 's', 'how', 'to', 'run', 'the', 'data', 'generation', '.', 'the',
        'input', 'is', 'a', 'plain', 'text', 'file', ',', 'with', 'one', 'sentence',
        'per', 'line', '.', 'it', 'is', 'important', 'that', 'these', 'be', 'actual',
        'sentence', '##s', 'for', 'the', '"', 'next', 'sentence', 'prediction', '"',
        'task', '.', 'documents', 'are', 'del', '##imit', '##ed', 'by', 'empty',
        'lines', '.', 'the', 'output', 'is', 'a', 'set', 'of', 't', '##f', '.',
        'train', '.', 'examples', 'serial', '##ized', 'into', 't', '##fre',
        '##cor', '##d', '[SEP]']

    def test_normal(self):
        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.VOCAB_FILE, do_lower_case=True)
        vocab_words = list(tokenizer.vocab.keys())
        rng = random.Random(12345)  # random seed

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
            self.TOKEN_SAMPLE, self.MASKED_LM_PROB, self.MAX_PREDICTIONS_PER_SEQ,
            vocab_words, rng)
        # segment_ids and is_random_next is dumb
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=[0],
            is_random_next=True,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        print(instance)


if __name__ == '__main__':
    unittest.main()
