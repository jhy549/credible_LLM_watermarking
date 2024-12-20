import json
import os.path
import random
import warnings
from dataclasses import dataclass

from datasets import load_from_disk
from tqdm import tqdm

from .arg_classes.wm_arg_class import WmBaseArgs
from .attacker.substition_attack import SubstitutionAttacker
import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import uniform_filter1d
import torch
import torch.nn.functional as F

@dataclass
class CopyPasteArgs:
    cp_window_size = 190
    cp_smooth_window_size = 1
    cp_threshold = 0.5
    cp_order = 0
    cp_gap = 190
    cp_avg_pool_size = 10

    @property
    def cp_result_name(self):
        s = 'cp-analysis'
        return s


def check_cp_in(results, cb_args: CopyPasteArgs):
    if cb_args.cp_result_name in results:
        return True
    return False



def insert_watermark(x_watermarked, x_human_written):
    # Split the text into a list of words
    words = x_human_written.split()

    # Generate a random index
    index = random.randint(0, len(words))

    # Insert the watermark at the random index
    words.insert(index, x_watermarked)

    # Join the list of words back into a single string
    x_human_written = ' '.join(words)

    return x_human_written


def insert_watermark_for_sentences(x_watermarked, x_human_written, seed=42):
    randomer = random.Random()
    randomer.seed(seed)
    x_human_written = randomer.sample(x_human_written, k=len(x_watermarked))
    x_cped = [insert_watermark(x_wm_single, x_human_single) for x_wm_single, x_human_single in
              zip(x_watermarked, x_human_written)]
    return x_cped


class MessageFilter:
    def __init__(self, device: str):
        self.device = device

    def get_messages_and_confidences(self, log_probs, window_size, avg_pool_size):
        if not isinstance(log_probs, torch.Tensor):
            log_probs = torch.tensor(log_probs)
        log_probs = (log_probs > 0).float()
        log_probs = log_probs.transpose(0, 1)
        log_probs = log_probs.unsqueeze(1)
        log_probs = log_probs.to(self.device)
        # print(window_size)
        kernel = torch.ones((1, 1, window_size), device=log_probs.device)
        avg_kernel = torch.ones((1, 1, avg_pool_size), device=log_probs.device)
        sum_result = F.conv1d(log_probs, avg_kernel, stride=avg_pool_size)
        # print(sum_result.shape)
        sum_result = F.conv1d(sum_result, kernel)
        confidences, messages = sum_result.softmax(0).max(0)
        # confidences, messages = sum_result.max(0)
        log_probs = log_probs.cpu()
        return messages, confidences

    def print_messages_and_confidences(self, log_probs, window_size):
        messages, confidences = self.get_messages_and_confidences(log_probs, window_size)
        for message, confidence in zip(messages, confidences):
            print(f'message: {message}, confidence: {confidence}')

    def filter_message(self, log_probs, window_size, smooth_window_size, threshold, order=5,
                       gap=20, avg_pool_size=10):
        gap = gap // avg_pool_size
        window_size = window_size // avg_pool_size
        # threshold = threshold / avg_pool_size
        filtered_messages = []
        filtered_confidences = []
        messages, confidences = self.get_messages_and_confidences(log_probs, window_size,
                                                                  avg_pool_size)
        messages = messages[0].cpu().numpy()
        confidences = confidences[0].cpu().numpy()
        if order > 0:
            zeros = np.zeros(order)
            smoothed_data = np.hstack((zeros, confidences, zeros))
        else:
            smoothed_data = confidences
        if smooth_window_size > 1:
            smoothed_data = uniform_filter1d(smoothed_data, size=smooth_window_size)
        if order == 0:
            local_maximas = np.arange(len(smoothed_data))
        else:
            local_maximas = argrelextrema(smoothed_data, np.greater, order=order)[0]
        local_maximas = local_maximas[
            (local_maximas < len(confidences) + order) & (local_maximas >= order)]
        local_maximas = local_maximas - order
        before_local = -100
        before_max = -100
        for i, local_maxima in enumerate(local_maximas):
            if confidences[local_maxima] > threshold:
                if local_maxima - before_local < gap and messages[local_maxima] == messages[
                    before_local]:
                    if confidences[local_maxima] > before_max:
                        before_max = confidences[local_maxima]
                    continue
                filtered_confidences.append(float(before_max))
                # print(f'confidence updated: {before_max}')
                before_local = local_maxima
                before_max = confidences[local_maxima]
                confidence = confidences[local_maxima]
                message = messages[local_maxima]
                # print(f'message: {message}, confidence: {confidence}')
                filtered_messages.append(int(message))
        # print(f'confidence updated: {before_max}')
        filtered_confidences.append(float(before_max))
        return filtered_messages, filtered_confidences[1:]


def main(args: WmBaseArgs, cp_args: CopyPasteArgs):
    if not os.path.exists(args.complete_save_file_path):
        raise ValueError(f'analysis_file_name: {args.complete_save_file_path} does not exist!')

    with open(args.complete_save_file_path, 'r') as f:
        results = json.load(f)

    if check_cp_in(results, cp_args):
        warnings.warn('copy-paste already in results. Skipping analysis.')
        return

    cp_analysis_results = results.get(cp_args.cp_result_name, {})
    if not check_cp_in(results, cp_args):
        random.seed(args.sample_seed)
        decoder = args.decoder
        message_filter = MessageFilter(device=args.device)
        human_written_data_for_cp = load_from_disk('./c4-train.00102-of-00512_sliced_for_cp/')
        x_wms = results['output_text']
        x_humans = human_written_data_for_cp['train']['truncated_text']
        x_cpeds = insert_watermark_for_sentences(x_wms, x_humans, seed=args.sample_seed)

        decoded_results = []
        corrected = []

        for i in tqdm(range(len(x_cpeds))):
            y = decoder.decode(x_cpeds[i], disable_tqdm=True)
            zz = message_filter.filter_message(y[1][0], window_size=190, smooth_window_size=1,
                                               threshold=0.5, order=0, gap=190, avg_pool_size=10)
            corrected.append(zz[0] == decoder.message.tolist()[:len(zz)])
            decoded_results.append(zz)
        successful_rate = sum(corrected) / len(corrected)

        cp_analysis_results['successful_rate'] = successful_rate
        cp_analysis_results['decoded_results'] = decoded_results
        cp_analysis_results['cp_args'] = cp_args.__dict__
        results[cp_args.cp_result_name] = cp_analysis_results

    with open(args.complete_save_file_path, 'w') as f:
        json.dump(results, f, indent=4)
