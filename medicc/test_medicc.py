import os
import pathlib
import subprocess
import time
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import fstlib
import medicc


def _reference_shorten_cn_strings(string_1, string_2):
    keep_indices = [i for i in range(len(string_1)) if
                    i == 0 or (string_1[i] != string_1[i - 1]) or (string_2[i] != string_2[i - 1])]
    string_1_short = ''.join(string_1[i] for i in keep_indices)
    string_2_short = ''.join(string_2[i] for i in keep_indices)
    return string_1_short, string_2_short


def _reference_pairwise_distance_matrix(model_fst, cn_str_dict):
    samples = list(cn_str_dict.keys())
    pdm = pd.DataFrame(0, index=samples, columns=samples, dtype=float)

    for sample_a, sample_b in combinations(samples, 2):
        cur_dist = medicc.calc_MED_distance(model_fst, cn_str_dict[sample_a], cn_str_dict[sample_b])
        pdm.loc[sample_a, sample_b] = cur_dist
        pdm.loc[sample_b, sample_a] = cur_dist

    return pdm


def test_medicc_help_box():
    "Just testing that medicc can be started"
    process = subprocess.Popen(['python', "medicc2", "--help"],
                               stdout=subprocess.PIPE,
                               cwd=pathlib.Path(__file__).parent.parent.absolute())

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    assert process.returncode == 0


def test_shorten_cn_strings_matches_previous_reference_simulation():
    rng = np.random.default_rng(2026)
    alphabet = np.array(list('012345678X'))

    for length in rng.integers(1, 2000, size=100):
        profile_1 = ''.join(rng.choice(alphabet, size=length))
        profile_2 = ''.join(rng.choice(alphabet, size=length))
        assert medicc.shorten_cn_strings(profile_1, profile_2) == _reference_shorten_cn_strings(
            profile_1, profile_2)


def test_calc_MED_distance_is_not_unbounded_lru_cache():
    assert not hasattr(medicc.calc_MED_distance, 'cache_info')


def test_pairwise_distance_matrix_matches_previous_reference_simulation():
    rng = np.random.default_rng(2026)
    medicc_fst = medicc.io.read_fst()
    simulated_profiles = {}

    for sample_idx in range(14):
        chromosomes = []
        for _ in range(6):
            copy_numbers = rng.integers(0, 6, size=18)
            chromosomes.append(''.join(copy_numbers.astype(str)))
        simulated_profiles[f'cell_{sample_idx}'] = 'X'.join(chromosomes)

    expected = _reference_pairwise_distance_matrix(medicc_fst, simulated_profiles)
    observed = medicc.calc_pairwise_distance_matrix(medicc_fst, simulated_profiles, parallel_run=False)
    pd.testing.assert_frame_equal(observed, expected)


def test_medicc_distance_speed_up():
    def generate_random_profiles(dataset_size=50, chr_num=22):

        def _generate_random_copy_number_profile(n):
            '''Generate random copy number profiles of length n.
            Every copy number is chosen uniformly at random from 0 to 8.'''
            copy_number_profile = [str(np.random.randint(0, 8)) for _ in range(n)]
            return "".join(copy_number_profile)

        profile_1_list = []
        profile_2_list = []
        np.random.seed(5)

        for i in range(dataset_size):
            chr_bin_size = []
            for j in range(chr_num):
                chr_bin_size.append(np.random.randint(1, 11)) # every chromosome has 1 to 10 bin sizes
            profile_1 = []
            profile_2 = []
            for bin_size in chr_bin_size:
                profile_1.append(_generate_random_copy_number_profile(bin_size))
                profile_2.append(_generate_random_copy_number_profile(bin_size))

            profile_1 = "X".join(profile_1)
            profile_2 = "X".join(profile_2)

            profile_1_list.append(profile_1)
            profile_2_list.append(profile_2)
        
        return profile_1_list, profile_2_list

    medicc_fst = medicc.io.read_fst()
    symbol_table = medicc_fst.input_symbols()
    profile_1_list, profile_2_list = generate_random_profiles()

    for i in range(len(profile_1_list)):
        profile_1_str = profile_1_list[i]
        profile_2_str = profile_2_list[i]

        profile_1_fsa = fstlib.factory.from_string(profile_1_str, isymbols=symbol_table, osymbols=symbol_table)
        profile_2_fsa = fstlib.factory.from_string(profile_2_str, isymbols=symbol_table, osymbols=symbol_table)

        distance_true = float(fstlib.kernel_score(medicc_fst, profile_1_fsa, profile_2_fsa))
        distance_short = medicc.calc_MED_distance(medicc_fst, profile_1_str, profile_2_str)

        assert distance_true == distance_short, f"Distance calculated using `shorten_cn_strings` is not correct. " \
                                                f"Expected {distance_true}, got {distance_short}"


def test_medicc_with_simple_example():
    "Testing small example"
    output_dir = 'examples/test_output'
    process = subprocess.Popen(['python', "medicc2", "examples/simple_example/simple_example.tsv", 
                                output_dir, "--plot", "both", "--events", "--chromosomes-bed",
                                "default", "--regions-bed", "default"],
                               stdout=subprocess.PIPE,
                               cwd=pathlib.Path(__file__).parent.parent.absolute())

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    expected_files = ['simple_example_cn_profiles.pdf', 'simple_example_final_cn_profiles.tsv',
                      'simple_example_final_tree.new', 'simple_example_final_tree.png',
                      'simple_example_final_tree.xml', 'simple_example_pairwise_distances.tsv',
                      'simple_example_summary.tsv', 'simple_example_copynumber_events_df.tsv',
                      'simple_example_events_overlap.tsv', 'simple_example_branch_lengths.tsv',
                      'simple_example_cn_profiles_heatmap.pdf']
    all_files_exist = [os.path.isfile(os.path.join('examples/test_output/', f)) for f in expected_files]
    nr_events, tree_size = get_number_of_events(output_dir, 'simple_example')
    output_df = pd.read_csv(os.path.join(output_dir, "simple_example_final_cn_profiles.tsv"), sep='\t')
    events_df = pd.read_csv(os.path.join(output_dir, "simple_example_copynumber_events_df.tsv"), sep='\t')
    subprocess.Popen(["rm", output_dir, "-rf"])

    assert process.returncode == 0, 'Error while running MEDICC'
    assert np.all(all_files_exist), "Some files were not created! Missing files are: {}".format(
        np.array(expected_files)[~np.array(all_files_exist)])
    assert nr_events == tree_size, f"Number of events is {nr_events}, but tree size is {tree_size}"

    assert output_df['is_gain'].sum() == 7, f"Number of gained segments in _final_cn_profiles.tsv is not 7 but {output_df['is_gain'].sum()}"
    assert output_df['is_loss'].sum() == 5, f"Number of lost segments in _final_cn_profiles.tsv is not 5 but {output_df['is_loss'].sum()}"

    assert (events_df['type'] == 'gain').sum() == 4, f"Number of gains in events_df is not 4 but {(events_df['type'] == 'gain').sum()}"
    assert (events_df['type'] == 'loss').sum() == 3, f"Number of losses in events_df is not 3 but {(events_df['type'] == 'loss').sum()}"


def test_medicc_with_testing_example():
    "Testing testing example"
    output_dir = 'examples/test_output'
    process = subprocess.Popen(['python', "medicc2", "examples/testing_example/testing_example.tsv", 
                                output_dir, "--events", "--chromosomes-bed", "default", "--regions-bed", "default"],
                               stdout=subprocess.PIPE,
                               cwd=pathlib.Path(__file__).parent.parent.absolute())

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    expected_files = ['testing_example_cn_profiles.pdf', 'testing_example_final_cn_profiles.tsv',
                      'testing_example_final_tree.new', 'testing_example_final_tree.png',
                      'testing_example_final_tree.xml', 'testing_example_pairwise_distances.tsv',
                      'testing_example_summary.tsv', 'testing_example_copynumber_events_df.tsv',
                      'testing_example_events_overlap.tsv', 'testing_example_branch_lengths.tsv']
    all_files_exist = [os.path.isfile(os.path.join('examples/test_output/', f)) for f in expected_files]
    nr_events, tree_size = get_number_of_events(output_dir, 'testing_example')
    output_df = pd.read_csv(os.path.join(output_dir, "testing_example_final_cn_profiles.tsv"), sep='\t')
    subprocess.Popen(["rm", output_dir, "-rf"])

    assert process.returncode == 0, 'Error while running MEDICC'
    assert np.all(all_files_exist), "Some files were not created! Missing files are: {}".format(
        np.array(expected_files)[~np.array(all_files_exist)])
    assert nr_events == tree_size, f"Number of events is {nr_events}, but tree size is {tree_size}"

    assert output_df['is_gain'].sum() == 187, f"Number of gains in _final_cn_profiles.tsv is not 187 but {output_df['is_gain'].sum()}"
    assert output_df['is_loss'].sum() == 170, f"Number of losses in _final_cn_profiles.tsv is not 170 but {output_df['is_loss'].sum()}"


def test_medicc_with_testing_example_total_copy_numbers():
    "Testing small example"
    output_dir = 'examples/test_output_total_cn'
    process = subprocess.Popen(['python', "medicc2", "examples/testing_example/testing_example.tsv", 
                                output_dir, "--total-copy-numbers", 
                                "--input-allele-columns", "cn_a", "--events", "--chromosomes-bed", "default", "--regions-bed", "default"],
                               stdout=subprocess.PIPE,
                               cwd=pathlib.Path(__file__).parent.parent.absolute())

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    expected_files = ['testing_example_cn_profiles.pdf', 'testing_example_final_cn_profiles.tsv',
                      'testing_example_final_tree.new', 'testing_example_final_tree.png',
                      'testing_example_final_tree.xml', 'testing_example_pairwise_distances.tsv',
                      'testing_example_summary.tsv', 'testing_example_copynumber_events_df.tsv',
                      'testing_example_events_overlap.tsv', 'testing_example_branch_lengths.tsv']
    all_files_exist = [os.path.isfile(os.path.join(output_dir, f))
                       for f in expected_files]
    nr_events, tree_size = get_number_of_events(output_dir, 'testing_example')
    subprocess.Popen(["rm", output_dir, "-rf"])

    assert process.returncode == 0, 'Error while running MEDICC'
    assert np.all(all_files_exist), "Some files were not created! Missing files are: {}".format(
        np.array(expected_files)[~np.array(all_files_exist)])
    assert nr_events == tree_size, f"Number of events is {nr_events}, but tree size is {tree_size}"


def test_medicc_with_testing_example_parallelization():
    "Testing small example"
    output_dir = 'examples/test_output_parallelization'
    process = subprocess.Popen(['python', "medicc2", "examples/testing_example/testing_example.tsv", 
                                output_dir, "--n-cores", "4", "--events", "--chromosomes-bed", "default", "--regions-bed", "default"],
                               stdout=subprocess.PIPE,
                               cwd=pathlib.Path(__file__).parent.parent.absolute())

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    expected_files = ['testing_example_cn_profiles.pdf', 'testing_example_final_cn_profiles.tsv',
                      'testing_example_final_tree.new', 'testing_example_final_tree.png',
                      'testing_example_final_tree.xml', 'testing_example_pairwise_distances.tsv',
                      'testing_example_summary.tsv', 'testing_example_copynumber_events_df.tsv',
                      'testing_example_events_overlap.tsv', 'testing_example_branch_lengths.tsv']
    all_files_exist = [os.path.isfile(os.path.join(output_dir, f))
                       for f in expected_files]
    nr_events, tree_size = get_number_of_events(output_dir, 'testing_example')
    subprocess.Popen(["rm", output_dir, "-rf"])

    assert process.returncode == 0, 'Error while running MEDICC'
    assert np.all(all_files_exist), "Some files were not created! Missing files are: {}".format(
        np.array(expected_files)[~np.array(all_files_exist)])
    assert nr_events == tree_size, f"Number of events is {nr_events}, but tree size is {tree_size}"


def test_medicc_with_testing_example_parallelization():
    "Testing small example"
    output_dir = 'examples/test_output_parallelization'
    process = subprocess.Popen(['python', "medicc2", "examples/testing_example/testing_example.tsv", 
                                output_dir, "--n-cores", "4", "--events", "--chromosomes-bed", "default", "--regions-bed", "default"],
                               stdout=subprocess.PIPE,
                               cwd=pathlib.Path(__file__).parent.parent.absolute())

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    expected_files = ['testing_example_cn_profiles.pdf', 'testing_example_final_cn_profiles.tsv',
                      'testing_example_final_tree.new', 'testing_example_final_tree.png',
                      'testing_example_final_tree.xml', 'testing_example_pairwise_distances.tsv',
                      'testing_example_summary.tsv', 'testing_example_copynumber_events_df.tsv',
                      'testing_example_events_overlap.tsv', 'testing_example_branch_lengths.tsv']
    all_files_exist = [os.path.isfile(os.path.join(output_dir, f))
                       for f in expected_files]
    nr_events, tree_size = get_number_of_events(output_dir, 'testing_example')
    subprocess.Popen(["rm", output_dir, "-rf"])

    assert process.returncode == 0, 'Error while running MEDICC'
    assert np.all(all_files_exist), "Some files were not created! Missing files are: {}".format(
        np.array(expected_files)[~np.array(all_files_exist)])
    assert nr_events == tree_size, f"Number of events is {nr_events}, but tree size is {tree_size}"


def test_medicc_with_testing_example_nowgd():
    "Testing small example"
    output_dir = 'examples/test_output_nowgd'
    process = subprocess.Popen(['python', "medicc2", "examples/testing_example/testing_example.tsv", 
                                output_dir, "--no-wgd", "--events", "--chromosomes-bed", "default", "--regions-bed", "default"],
                               stdout=subprocess.PIPE,
                               cwd=pathlib.Path(__file__).parent.parent.absolute())

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    expected_files = ['testing_example_cn_profiles.pdf', 'testing_example_final_cn_profiles.tsv',
                      'testing_example_final_tree.new', 'testing_example_final_tree.png',
                      'testing_example_final_tree.xml', 'testing_example_pairwise_distances.tsv',
                      'testing_example_summary.tsv', 'testing_example_copynumber_events_df.tsv',
                      'testing_example_events_overlap.tsv', 'testing_example_branch_lengths.tsv']
    all_files_exist = [os.path.isfile(os.path.join(output_dir, f))
                        for f in expected_files]

    nr_events, tree_size = get_number_of_events(output_dir, 'testing_example')
    subprocess.Popen(["rm", output_dir, "-rf"])

    assert process.returncode == 0, 'Error while running MEDICC'
    assert np.all(all_files_exist), "Some files were not created! Missing files are: {}".format(
        np.array(expected_files)[~np.array(all_files_exist)])
    assert nr_events == tree_size, f"Number of events is {nr_events}, but tree size is {tree_size}"


def test_medicc_with_testing_example_WGD_x2():
    "Testing small example"
    output_dir = 'examples/test_output_wgd_x2'
    process = subprocess.Popen(['python', "medicc2", "examples/testing_example/testing_example.tsv", 
                                output_dir, "--wgd-x2", "--events", "--chromosomes-bed", "default", "--regions-bed", "default"],
                               stdout=subprocess.PIPE,
                               cwd=pathlib.Path(__file__).parent.parent.absolute())

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    expected_files = ['testing_example_cn_profiles.pdf', 'testing_example_final_cn_profiles.tsv',
                      'testing_example_final_tree.new', 'testing_example_final_tree.png',
                      'testing_example_final_tree.xml', 'testing_example_pairwise_distances.tsv',
                      'testing_example_summary.tsv', 'testing_example_copynumber_events_df.tsv',
                      'testing_example_events_overlap.tsv', 'testing_example_branch_lengths.tsv']
    all_files_exist = [os.path.isfile(os.path.join(output_dir, f))
                       for f in expected_files]
    nr_events, tree_size = get_number_of_events(output_dir, 'testing_example')
    subprocess.Popen(["rm", output_dir, "-rf"])

    assert process.returncode == 0, 'Error while running MEDICC'
    assert np.all(all_files_exist), "Some files were not created! Missing files are: {}".format(
        np.array(expected_files)[~np.array(all_files_exist)])
    assert nr_events == tree_size, f"Number of events is {nr_events}, but tree size is {tree_size}"


def test_medicc_with_multiple_cores():
    "Testing small example"
    output_dir = 'examples/test_output_multiple_cores'
    process = subprocess.Popen(['python', "medicc2", "examples/simple_example/simple_example.tsv", 
                                output_dir, "--n-cores", "4", "--events", "--chromosomes-bed", "default", "--regions-bed", "default"],
                               stdout=subprocess.PIPE,
                               cwd=pathlib.Path(__file__).parent.parent.absolute())

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    expected_files = ['simple_example_cn_profiles.pdf', 'simple_example_final_cn_profiles.tsv',
                      'simple_example_final_tree.new', 'simple_example_final_tree.png',
                      'simple_example_final_tree.xml', 'simple_example_pairwise_distances.tsv',
                      'simple_example_summary.tsv', 'simple_example_copynumber_events_df.tsv',
                      'simple_example_events_overlap.tsv', 'simple_example_branch_lengths.tsv']
    all_files_exist = [os.path.isfile(os.path.join('examples/test_output_multiple_cores/', f))
                       for f in expected_files]
    nr_events, tree_size = get_number_of_events(output_dir, 'simple_example')
    subprocess.Popen(["rm", output_dir, "-rf"])

    assert process.returncode == 0, 'Error while running MEDICC'
    assert np.all(all_files_exist), "Some files were not created! Missing files are: {}".format(
        np.array(expected_files)[~np.array(all_files_exist)])
    assert nr_events == tree_size, f"Number of events is {nr_events}, but tree size is {tree_size}"


def test_medicc_with_OV03_04():
    "Testing testing example"
    output_dir = 'examples/test_output_OV03_04'
    process = subprocess.Popen(['python', "medicc2", "examples/OV03-04/OV03-04_descr.txt", 
                                output_dir, "-i", "fasta", "--normal-name", "OV03-04_diploid",
                                "--plot", "both", "--events", "--chromosomes-bed", "default", "--regions-bed", "default"],
                               stdout=subprocess.PIPE,
                               cwd=pathlib.Path(__file__).parent.parent.absolute())

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    expected_files = ['OV03-04_descr_cn_profiles.pdf', 'OV03-04_descr_final_cn_profiles.tsv',
                      'OV03-04_descr_final_tree.new', 'OV03-04_descr_final_tree.png',
                      'OV03-04_descr_final_tree.xml', 'OV03-04_descr_pairwise_distances.tsv',
                      'OV03-04_descr_summary.tsv', 'OV03-04_descr_copynumber_events_df.tsv',
                      'OV03-04_descr_events_overlap.tsv', 'OV03-04_descr_branch_lengths.tsv',
                      'OV03-04_descr_cn_profiles_heatmap.pdf']
    all_files_exist = [os.path.isfile(os.path.join(output_dir, f)) for f in expected_files]
    subprocess.Popen(["rm", output_dir, "-rf"])

    assert process.returncode == 0, 'Error while running MEDICC'
    assert np.all(all_files_exist), "Some files were not created! Missing files are: {}".format(
        np.array(expected_files)[~np.array(all_files_exist)])


def test_medicc_with_bootstrap():
    "Testing bootstrap workflow"
    output_dir = 'examples/test_output_bootstrap'
    process = subprocess.Popen(['python', "medicc2",
                                "examples/simple_example/simple_example.tsv",
                                output_dir,
                                "--bootstrap-nr", "5"],
                               stdout=subprocess.PIPE,
                               cwd=pathlib.Path(__file__).parent.parent.absolute())

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    support_tree_exists = os.path.isfile('examples/test_output_bootstrap/simple_example_support_tree.new')
    subprocess.Popen(["rm", output_dir, "-rf"])

    assert process.returncode == 0, 'Error while running MEDICC'
    assert support_tree_exists, "Support tree file was not created"


gundem_et_al_2015_patients = ['PTX004', 'PTX005', 'PTX006', 'PTX007', 'PTX008', 
                              'PTX009', 'PTX010', 'PTX011', 'PTX012', 'PTX013']
extra_condition = ['normal', 'no_wgd', 'total_cn', 'wgd_x2']
@pytest.mark.parametrize("patient", gundem_et_al_2015_patients)
@pytest.mark.parametrize("extra_condition", extra_condition)
def test_gundem_et_al_2015(patient, extra_condition):
    "Testing if running of all Gundem data works"

    output_dir = f"examples/test_output_{patient}"
    command = ['python', "medicc2", f"examples/gundem_et_al_2015/{patient}_input_df.tsv", output_dir,
               "--events", "--chromosomes-bed", "default", "--regions-bed", "default"]
    if extra_condition == 'normal':
        pass
    elif extra_condition == 'no_wgd':
        command.append('--no-wgd')
    elif extra_condition == 'total_cn':
        command += ['--total-copy-numbers', '--input-allele-columns', 'cn_a']
    elif extra_condition == 'wgd_x2':
        command.append('--wgd-x2')

    command += ["--events", "--chromosomes-bed", "default", "--regions-bed", "default"]
        
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               cwd=pathlib.Path(__file__).parent.parent.absolute())

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    expected_files = [f'{patient}_input_df_cn_profiles.pdf', f'{patient}_input_df_final_cn_profiles.tsv',
                      f'{patient}_input_df_final_tree.new', f'{patient}_input_df_final_tree.png',
                      f'{patient}_input_df_final_tree.xml', f'{patient}_input_df_pairwise_distances.tsv',
                      f'{patient}_input_df_summary.tsv', f'{patient}_input_df_copynumber_events_df.tsv',
                      f'{patient}_input_df_events_overlap.tsv', f'{patient}_input_df_branch_lengths.tsv']

    all_files_exist = [os.path.isfile(os.path.join(output_dir, f)) for f in expected_files]
    nr_events, tree_size = get_number_of_events(output_dir, f'{patient}_input_df')
    subprocess.Popen(["rm", output_dir, "-rf"])

    assert process.returncode == 0, f'Error while running MEDICC for Gundem et al patient {patient}'
    assert np.all(all_files_exist), "Some files were not created! Missing files are: {}".format(
        np.array(expected_files)[~np.array(all_files_exist)])
    assert (extra_condition == 'total_cn') or (nr_events == tree_size), f"Number of events is {nr_events}, but tree size is {tree_size}"


all_ipynb_notebooks = [x for x in os.listdir('notebooks') if '.ipynb' in x]
@pytest.mark.parametrize("notebook", all_ipynb_notebooks)
def test_all_ipynb_notebooks(notebook):
    "Testing if all notebooks (with ending .ipynb) work"

    process = subprocess.Popen([f'ipython -c "%run {notebook}"'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               shell=True,
                               cwd=os.path.join(pathlib.Path(__file__).parent.parent.absolute(), 'notebooks'))

    while process.poll() is None:
        # Process hasn't exited yet
        time.sleep(0.5)

    assert process.returncode == 0, f'Error while running notebook {notebook}: {process.stderr.read()}'


def get_number_of_events(output_dir, file_prefix):
    with open(os.path.join(output_dir, f"{file_prefix}_copynumber_events_df.tsv"), 'r') as f:
        events = f.readlines()
    nr_events = len(events) - 1

    with open(os.path.join(output_dir, f"{file_prefix}_summary.tsv"), 'r') as f:
        summary = f.readlines()
    tree_size = float(summary[2].split('\t')[1].rstrip())

    return nr_events, tree_size
