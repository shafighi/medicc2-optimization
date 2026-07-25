import pickle
import sys

import fstlib


def shorten_cn_strings(string_1, string_2):
    if len(string_1) != len(string_2):
        raise ValueError("Copy-number profiles must have identical lengths.")
    if not string_1:
        return "", ""

    string_1_short = [string_1[0]]
    string_2_short = [string_2[0]]
    previous_1 = string_1[0]
    previous_2 = string_2[0]
    for index in range(1, len(string_1)):
        current_1 = string_1[index]
        current_2 = string_2[index]
        if current_1 != previous_1 or current_2 != previous_2:
            string_1_short.append(current_1)
            string_2_short.append(current_2)
        previous_1 = current_1
        previous_2 = current_2
    return "".join(string_1_short), "".join(string_2_short)


def calc_MED_distance(model_fst, profile_1, profile_2):
    profile_1, profile_2 = shorten_cn_strings(profile_1, profile_2)
    symbol_table = model_fst.input_symbols()
    profile_1_fsa = fstlib.factory.from_string(
        profile_1, isymbols=symbol_table, osymbols=symbol_table)
    profile_2_fsa = fstlib.factory.from_string(
        profile_2, isymbols=symbol_table, osymbols=symbol_table)
    return float(fstlib.kernel_score(model_fst, profile_1_fsa, profile_2_fsa))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m medicc.pairwise_worker MODEL_FST")

    model_fst = fstlib.read(sys.argv[1])
    requests = pickle.load(sys.stdin.buffer)
    try:
        results = []
        for sample_a_idx, sample_b_idx, profile_1, profile_2 in requests:
            distance = calc_MED_distance(model_fst, profile_1, profile_2)
            results.append((sample_a_idx, sample_b_idx, distance))
        response = (True, results)
    except Exception as exc:
        response = (False, repr(exc))
    pickle.dump(response, sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
