import torch
import torch.nn.functional as F
import numpy as np
import json, cv2
import argparse
from utils.dataloader import data_turn
from PIL import Image

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=5):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # 图片读取以及预处理过程
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片由BGR转灰度
    img = data_turn(img, resize=True)  # 图片预处理
    image = torch.FloatTensor(img).to(device)

    with torch.no_grad():
        # Encode
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 3, 256, 256)
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(2), encoder_out.size(3)
        encoder_dim = encoder_out.size(1)  # 这里和普通的resnet输出的不同，resnet是最后一个维度是C

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

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, enc_image_size[0], enc_image_size[1]).to(
            device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        # h, c = decoder.init_hidden_state(encoder_out)
        h = decoder.init_hidden_state(encoder_out)

        # s <= k,一旦输出<end>就会跳出该过程
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            # awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            alpha = alpha.view(-1, enc_image_size[0], enc_image_size[1])  # (s, enc_image_size, enc_image_size)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            # h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            h = decoder.decode_step(torch.cat([embeddings, awe], dim=1), h)  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # 对于第一步，所有k个点都有相同的分数 (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # 展开并找到最高分数及其展开的索引
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # 将展开的索引转换为实际的分数索引
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # 把新的单词加入到序列中, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                   dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # 处理未结束的序列
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            # c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            print('step', step)
            if step > 160:
                break
            step += 1

        complete_seqs_scores = np.array(complete_seqs_scores)
        i = np.argmax(complete_seqs_scores)
        # i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]

    return seq, alphas
    # return seq


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    # image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]
    print(words)
    print(alphas.shape)

    # for t in range(len(words)):
    #     if t > 50:
    #         break
    #     plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

    #     plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
    #     plt.imshow(image)
    #     current_alpha = alphas[t, :]
    #     if smooth:
    #         alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
    #     else:
    #         alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
    #     if t == 0:
    #         plt.imshow(alpha, alpha=0)
    #     else:
    #         plt.imshow(alpha, alpha=0.8)
    #     plt.set_cmap(cm.Greys_r)
    #     plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', default='./data/CROHME/images/images_train/TrainData2_15_sub_33.png',
                        help='path to image')
    parser.add_argument('--model', '-m', default='BEST_checkpoint_CROHME.pth.tar', help='path to model')
    parser.add_argument('--word_map', '-wm', default='./data/CROHME/vocab.json', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=3, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    print(seq)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
