

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
import sys

class UtteranceExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               id,
               utterance_tokens,
               query_tokens,
               answer_start_position=None,
               answer_end_position=None,
              answer_text=None,
               is_impossible=False):
    self.id = id
    self.query_tokens = query_tokens
    self.utterance_tokens = utterance_tokens
    self.answer_start_position = answer_start_position
    self.answer_end_position = answer_end_position
    self.answer_text = answer_text
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", utterance_tokens: [%s]" % (" ".join(self.utterance_tokens))
    s += ", query_tokens: [%s]" % (" ".join(self.query_tokens))
    if self.answer_start_position:
      s += ", answer_start_position: %d" % (self.answer_start_position)
    if self.answer_start_position:
      s += ", end_position: %d" % (self.answer_end_position)
    if self.is_impossible:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s

import csv 

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def build_char_to_word(text):
    prev_is_whitespace = True
    char_to_word_offset = []
    tokens = []
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
            else:
                tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(tokens) - 1)
    return tokens, char_to_word_offset

def read_csv_examples(input_file, is_training):
    """Read an csv file containing an utterance per line into a list of UtteranceExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        examples = []
        
        csvreader = csv.reader(reader, delimiter=',')
        for entry in csvreader:
            utterance_tokens, utterance_char_to_word_offset = build_char_to_word(entry[0].replace('"',""))
            query_tokens, query_char_to_word_offset = build_char_to_word(entry[1].replace('"',""))
            answer_text = entry[2].replace('"',"")
            answer_start = int(entry[3].replace('"',""))
            answer_end = int(entry[4].replace('"',""))

            def can_find(text, offset, length, tokens, char_to_word_offset):
                start_position = char_to_word_offset[offset]
                end_position = char_to_word_offset[offset + length - 1]
                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
                actual_text = " ".join(tokens[start_position:(end_position + 1)])
                cleaned_answer_text = " ".join(tokenization.whitespace_tokenize(text))
                if actual_text.find(cleaned_answer_text) == -1:
                    tf.logging.warning("Could not find answer: '%s' vs. '%s'",actual_text, cleaned_answer_text)
                    return None,None
                return start_position,end_position

            if not (answer_start == -1 and answer_end == -1):
                answer_start_position, answer_end_position = can_find(answer_text, answer_start, answer_end - answer_start, utterance_tokens, utterance_char_to_word_offset)
                if answer_start_position is None or answer_end_position is None:
                    continue
                else:
                    answer_start_position = -1
                    answer_end_position = -1
                    answer_text = ""
                    if answer_start_position is None:
                        print(paragraph)
                        raise Exception()
            example = UtteranceExample(
                id=id,
                utterance_tokens=utterance_tokens,
                query_tokens=query_tokens,
                answer_start_position=answer_start_position,
                answer_end_position=answer_end_position,
                answer_text=answer_text,
                is_impossible=answer_start_position==-1)
            examples.append(example)
            print("EXAMPLES {0}".format(len(examples)))
        return examples
