from pathlib import Path
from typing import NamedTuple
import math
import json

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Tag(NamedTuple):
#     tag: str
#     word_score: float
#     caption_score: float


class Tag:
    def __init__(self, tag_, word_score_, caption_score_):
        self.tag = tag_
        self.caption_score = caption_score_
        self.word_score = word_score_


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: captions, scores for captions, scores for individual words in captions
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    # img = imread(image_path)
    # if len(img.shape) == 2:
    #     img = img[:, :, np.newaxis]
    #     img = np.concatenate([img, img, img], axis=2)
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = np.array(img)
    # img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' scores per word; now they're just 0
    top_k_scores_per_word = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()
    complete_seqs_scores_per_word = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores_for_individual_words = decoder.fc(h)  # (s, vocab_size)
        scores_for_individual_words = F.log_softmax(scores_for_individual_words, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores_for_individual_words) + scores_for_individual_words  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        # prev_word_inds = top_k_words / vocab_size  # (s)
        prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='trunc')  # fix for pytorch > 0.4.0
        next_word_inds = top_k_words % vocab_size  # (s)

        top_k_scores_individual_word = scores_for_individual_words.view(-1)[top_k_words]

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        top_k_scores_per_word = torch.cat([top_k_scores_per_word[prev_word_inds], top_k_scores_individual_word.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            complete_seqs_scores_per_word.extend(top_k_scores_per_word[complete_inds].tolist())
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        top_k_scores_per_word = top_k_scores_per_word[incomplete_inds]
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    # i = complete_seqs_scores.index(max(complete_seqs_scores))
    # seq = complete_seqs[i]
    # alphas = complete_seqs_alpha[i]

    rev_word_map = {v: k for k, v in word_map.items()}

    sentences = [[rev_word_map[i] for i in seq] for seq in complete_seqs]

    return sentences, list(map(lambda x: x.tolist(), complete_seqs_scores)), complete_seqs_scores_per_word


def get_tags_from_captions(captions, scores_per_caption, scores_per_word):
    """
    takes the captions extracted from the image and converts them to tags
    :param captions: the different captions (list[list[str]])
    :param scores_per_caption: the log score of the captions (list[float])
    :param scores_per_word:  the log score of the individual words in the captions (list[list[float]])
    :return:
    """

    non_wanted_words = {'<start>', '<end>',
                        'a', 'with', 'of', 'on', 'the', 'to', 'and', 'in', 'is', 'her', 'his', 'there', 'that', 'an',
                        'next', 'who', 'while', 'at', 'he', 'she', 'up', 'using', 'each', 'other', 'are', 'something'}

    tags = dict()
    for words, word_scores, caption_score in zip(captions, scores_per_word, scores_per_caption):
        caption_score = math.exp(caption_score)
        for word, word_score in zip(words, word_scores):
            if word not in non_wanted_words:
                word_score = math.exp(word_score)
                if word not in tags:
                    tags[word] = Tag(word, word_score, caption_score)
                else:
                    tags[word].word_score = max(word_score, tags[word].word_score)
                    tags[word].caption_score = max(caption_score, tags[word].caption_score)

    return tags


if __name__ == '__main__':
    ROOT = Path(__file__).parent

    image_file = ROOT / 'images/test_03__001_2021_11_insp_vatertag_thementeaser_querformat_125642.jpg'
    model_checkpoint_file = ROOT / 'checkpoint/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    word_map_file = ROOT / 'checkpoint/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
    beam_size = 105


    # Load model
    checkpoint = torch.load(model_checkpoint_file, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    word_map = json.loads(word_map_file.read_text())
    # rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    captions, scores, scores_per_word = caption_image_beam_search(encoder, decoder, image_file, word_map, beam_size)
    tags = get_tags_from_captions(captions, scores, scores_per_word)
    pass
