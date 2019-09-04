import unittest
import tensorflow as tf
import random
import os

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

    flags = tf.flags
    FLAGS = flags.FLAGS

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
        # need to set FLAGS.do_whole_word_mask
        self.FLAGS.do_whole_word_mask = True
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
    flags = tf.flags
    FLAGS = flags.FLAGS

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = tokenization.FullTokenizer(
            vocab_file=cls.VOCAB_FILE, do_lower_case=True)
        cls.vocab_words = list(cls.tokenizer.vocab.keys())
        cls.rng = random.Random(12345)  # random seed
        cls.FLAGS.do_whole_word_mask = True

    def test_normal(self):
        # tokenizer = tokenization.FullTokenizer(
        #     vocab_file=self.VOCAB_FILE, do_lower_case=True)
        # vocab_words = list(tokenizer.vocab.keys())
        # rng = random.Random(12345)  # random seed
        # self.FLAGS.do_whole_word_mask = True

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
            self.TOKEN_SAMPLE, self.MASKED_LM_PROB, self.MAX_PREDICTIONS_PER_SEQ,
            self.vocab_words, self.rng, "normal")
        # segment_ids and is_random_next is dumb
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=[0],
            is_random_next=True,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        print("Test normal")
        print(instance)

    def test_front_half(self):
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
            self.TOKEN_SAMPLE, self.MASKED_LM_PROB, self.MAX_PREDICTIONS_PER_SEQ,
            self.vocab_words, self.rng, "front-half")
        n = len(tokens)
        for pos in masked_lm_positions:
            if tokens[pos] == "[MASK]":
                continue
            self.assertLess(pos, 0.5 * n)

        # segment_ids and is_random_next is dumb
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=[0],
            is_random_next=True,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        print("Test front-half")
        print(instance)

    # TODO: back-half, middle, odd, even


class TestMain(unittest.TestCase):
    VOCAB_FILE = "vocab.txt"

    def test_main(self):
        result = os.system(
            "python ../create_pretraining_data.py "
            "--input_file=./sample.txt "
            "--output_file=./tf_examples.tfrecord "
            "--vocab_file=./vocab.txt")
        self.assertEqual(result, 0)

if __name__ == '__main__':
    unittest.main()
