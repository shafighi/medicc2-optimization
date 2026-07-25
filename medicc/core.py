import copy
import hashlib
import logging
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import Bio
import fstlib
import numpy as np
import pandas as pd

import medicc
from medicc import io, nj, tools, event_reconstruction


# prepare logger 
logger = logging.getLogger(__name__)

_PAIRWISE_WORKER_MODEL_FST = None
_PAIRWISE_WORKER_CN_STR_DICT = None
_PAIRWISE_THREAD_LOCAL = threading.local()
_PAIRWISE_THREAD_MODEL_STATE = None
_PAIRWISE_THREAD_CN_STR_DICT = None


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %s", name, value, default)
        return default
    return parsed


def main(input_df,
         asymm_fst,
         normal_name='diploid',
         input_tree=None,
         ancestral_reconstruction=True,
         chr_separator='X',
         prune_weight=0,
         allele_columns=['cn_a', 'cn_b'],
         wgd_x2=False,
         no_wgd=False,
         total_cn=False,
         n_cores=None,
         reconstruct_events=False):
    """ MEDICC Main Method """

    symbol_table = asymm_fst.input_symbols()

    ## Validate input
    logger.info("Validating input.")
    io.validate_input(input_df, symbol_table, normal_name=normal_name)

    ## Compile compact strings for pairwise distance calculation. Full FSAs are
    ## constructed only if ancestral reconstruction needs them.
    logger.info("Compiling input copy-number strings.")
    CN_str_dict = create_cn_string_dict_from_data(input_df, chr_separator)
    sample_labels = input_df.index.get_level_values('sample_id').unique()

    ## Reconstruct a tree
    if input_tree is None:
        ## Calculate pairwise distances
        logger.info("Calculating pairwise distance matrices")
        if n_cores is not None and n_cores > 1:
            pairwise_distances = parallelization_calc_pairwise_distance(sample_labels, asymm_fst, CN_str_dict,
                                                                                    n_cores)
        else:
            pairwise_distances = calc_pairwise_distance_matrix(asymm_fst, CN_str_dict)

        if (pairwise_distances == np.inf).any().any():
            affected_pairs = [(pairwise_distances.index[s1], pairwise_distances.index[s2])
                              for s1, s2 in zip(*np.where((pairwise_distances == np.inf)))]
            raise MEDICCError("Evolutionary distances could not be calculated for some sample "
                              "pairings. Please check the input data.\n\nThe affected pairs are: "
                              f"{affected_pairs}")

        logger.info("Inferring tree topology.")
        nj_tree = infer_tree_topology(
            pairwise_distances.values, pairwise_distances.index, normal_name=normal_name)
    else:
        logger.info("Tree provided, using it. No pairwise distance matrix is calculated!")

        pairwise_distances = pd.DataFrame(
            0, columns=CN_str_dict.keys(), index=CN_str_dict.keys())

        assert len([x for x in list(input_tree.find_clades()) if x.name is not None and 'internal' not in x.name]) == \
            len(np.unique(input_df.index.get_level_values('sample_id'))), \
                "Number of samples differs in input tree and input dataframe"
        assert np.all(
            np.sort([x.name for x in list(input_tree.find_clades()) if x.name is not None and 'internal' not in x.name]) ==
            np.sort(np.unique(input_df.index.get_level_values('sample_id')))), (
                "Input tree does not match input dataframe: "
                f"{np.sort([x.name for x in list(input_tree.find_clades()) if x.name is not None and 'internal' not in x.name])}\n"
                f"{np.sort(np.unique(input_df.index.get_level_values('sample_id')))}")
        
        # necessary for the way that reconstruct_ancestors is performed
        if ancestral_reconstruction:
            input_tree.root_with_outgroup([x for x in input_tree.root.clades if x.name != normal_name][0].name)

        nj_tree = input_tree

    final_tree = copy.deepcopy(nj_tree)

    if ancestral_reconstruction:
        logger.info("Compiling input sequences into FSAs for ancestor reconstruction.")
        FSA_dict = create_fsa_dict_from_strings(CN_str_dict, symbol_table)
        logger.info("Reconstructing ancestors.")
        ancestors = medicc.reconstruct_ancestors(tree=final_tree,
                                                 samples_dict=FSA_dict,
                                                 fst=asymm_fst,
                                                 normal_name=normal_name,
                                                 prune_weight=prune_weight)

        ## Create and write output data frame with ancestors
        logger.info("Creating output copynumbers.")
        output_df = create_df_from_fsa(input_df, ancestors)

        ## Update branch lengths with ancestors
        logger.info("Updating branch lengths of final tree using ancestors.")
        update_branch_lengths(final_tree, asymm_fst, ancestors, normal_name)
    else:
        output_df = input_df.copy()

    nj_tree.root_with_outgroup(normal_name)
    final_tree.root_with_outgroup(normal_name)

    if ancestral_reconstruction and reconstruct_events:
        logger.info("Reconstructing events.")
        output_df, events_df = event_reconstruction.calculate_all_cn_events(
            final_tree, output_df, allele_columns, normal_name,
            wgd_x2=wgd_x2, no_wgd=no_wgd, total_cn=total_cn)
        if len(events_df) != final_tree.total_branch_length():
            faulty_nodes = []
            for node in final_tree.find_clades():
                if node.name is not None and node.name != normal_name and node.branch_length != 0 and node.branch_length != len(events_df.loc[node.name]):
                    faulty_nodes.append(node.name)
            logger.warning("Event recreation was faulty. Events in '_cn_events_df.tsv' will be "
                        f"incorrect for the following nodes: {faulty_nodes}. "
                        f"total_branch_length: {final_tree.total_branch_length()}, "
                        f"nr of inferred events: {len(events_df)}")

    else:
        events_df = None

    return sample_labels, pairwise_distances, nj_tree, final_tree, output_df, events_df


def create_standard_fsa_dict_from_data(input_data,
                                       symbol_table: fstlib.SymbolTable,
                                       separator: str = "X") -> dict:
    """ Creates a dictionary of FSAs from input DataFrame or Series.
    The keys of the dictionary are the sample/taxon names. 
    If the input is a DataFrame, the FSA will be the concatenated copy number profiles of all allele columns"""

    cn_str_dict = create_cn_string_dict_from_data(input_data, separator)
    fsa_dict = create_fsa_dict_from_strings(cn_str_dict, symbol_table)

    return fsa_dict, cn_str_dict


def create_cn_string_dict_from_data(input_data,
                                    separator: str = "X") -> dict:
    """Create compact copy-number strings without constructing native FSAs."""
    cn_str_dict = {}
    if isinstance(input_data, pd.DataFrame):
        logger.info('Creating FSA for pd.DataFrame with the following data columns: {}'.format(
            input_data.columns.values))
        def aggregate_copy_number_profile(cnp):
            return separator.join([separator.join(["".join(x.astype('str'))
                                                   for _, x in cnp[allele].groupby('chrom', observed=False)]) for allele in cnp.columns])

    elif isinstance(input_data, pd.Series):
        logger.info('Creating FSA for pd.Series with the name {}'.format(input_data.name))
        def aggregate_copy_number_profile(cnp):
            return separator.join(["".join(x.astype('str')) for _, x in cnp.groupby('chrom', observed=False)])

    else:
        raise MEDICCError("Input to function create_standard_fsa_dict_from_data has to be either"
                          "pd.DataFrame or pd.Series. \n input provided was {}".format(type(input_data)))

    for taxon, cnp in input_data.groupby('sample_id'):
        cn_str_dict[taxon] = aggregate_copy_number_profile(cnp)

    return cn_str_dict


def create_fsa_dict_from_strings(cn_str_dict,
                                 symbol_table: fstlib.SymbolTable) -> dict:
    """Construct native FSAs from precompiled copy-number strings."""
    return {
        taxon: fstlib.factory.from_string(
            cn_str,
            arc_type="standard",
            isymbols=symbol_table,
            osymbols=symbol_table)
        for taxon, cn_str in cn_str_dict.items()
    }


def create_phasing_fsa_dict_from_df(input_df: pd.DataFrame, symbol_table: fstlib.SymbolTable, separator: str = "X") -> dict:
    """ Creates a dictionary of FSAs from two allele columns (Pandas DataFrame).
    The keys of the dictionary are the sample/taxon names. """
    allele_columns = input_df.columns
    if len(allele_columns) != 2:
        raise MEDICCError("Need exactly two alleles for phasing.")

    fsa_dict = {}
    for taxon, cnp in input_df.groupby('sample_id'):
        allele_a = cnp[allele_columns[0]]
        allele_b = cnp[allele_columns[1]]
        cn_str_a = separator.join(["".join(x) for _,x in allele_a.groupby(level='chrom', sort=False)])
        cn_str_b = separator.join(["".join(x) for _,x in allele_b.groupby(level='chrom', sort=False)])
        encoded = np.array([list(zip(cn_str_a, cn_str_b)), list(zip(cn_str_b, cn_str_a))])
        fsa_dict[taxon] = fstlib.factory.from_array(encoded, symbols=symbol_table, arc_type='standard')
        fsa_dict[taxon] = fstlib.determinize(fsa_dict[taxon]).minimize()

    return fsa_dict

def phase(input_df: pd.DataFrame, model_fst: fstlib.Fst, reference_sample='diploid', separator: str = 'X') -> pd.DataFrame:
    """ Phases every FST against the reference sample. 
    Returns two standard FSA dicts, one for each allele. """
    
    diploid_fsa = medicc.tools.create_diploid_fsa(model_fst)
    phasing_dict = medicc.create_phasing_fsa_dict_from_df(input_df, model_fst.input_symbols(), separator)
    fsa_dict_a, fsa_dict_b, _ = phase_dict(phasing_dict, model_fst, diploid_fsa)
    output_df = medicc.create_df_from_phasing_fsa(input_df, [fsa_dict_a, fsa_dict_b], separator)

    # Phasing across chromosomes is random, so we need to swap haplotype assignment per chromosome
    # so that the higher ploidy haplotype is always cn_a
    output_df['width'] = output_df.eval('end+1-start')
    output_df['cn_a_width'] = output_df['cn_a'].astype(float) * output_df['width']
    output_df['cn_b_width'] = output_df['cn_b'].astype(float) * output_df['width']

    swap_haplotypes_ind = output_df.groupby(['sample_id', 'chrom'])[
        ['cn_a_width', 'cn_b_width']].mean().diff(axis=1).iloc[:, 1] > 0

    output_df = output_df.join(swap_haplotypes_ind.rename('swap_haplotypes_ind'), on=['sample_id', 'chrom'])
    output_df.loc[output_df['swap_haplotypes_ind'], ['cn_a', 'cn_b']] = output_df.loc[output_df['swap_haplotypes_ind'], ['cn_b', 'cn_a']].values
    output_df = output_df.drop(['width', 'cn_a_width', 'cn_b_width', 'swap_haplotypes_ind'], axis=1)

    return output_df

def phase_dict(phasing_dict, model_fst, reference_fst):
    """ Phases every FST against the reference sample. 
    Returns two standard FSA dicts, one for each allele. """
    fsa_dict_a = {}    
    fsa_dict_b = {}
    scores = {}
    left = (reference_fst * model_fst).project('output')
    right = (~model_fst * reference_fst).project('input')
    for sample_id, sample_fst in phasing_dict.items():
        phased_fst = fstlib.align(sample_fst, left, right).topsort()
        score = fstlib.shortestdistance(phased_fst, reverse=True)[phased_fst.start()]
        scores[sample_id] = float(score)
        fsa_dict_a[sample_id] = fstlib.arcmap(phased_fst.copy().project('input'), map_type='rmweight')
        fsa_dict_b[sample_id] = fstlib.arcmap(phased_fst.project('output'), map_type='rmweight')
    
    return fsa_dict_a, fsa_dict_b, scores


def create_df_from_fsa(input_df: pd.DataFrame, fsa, separator: str = 'X'):
    """ 
    Takes a single FSA dict or a list of FSA dicts and extracts the copy number profiles.
    The allele names are taken from the input_df columns and the returned data frame has the same 
    number of rows and row index as the input_df. """

    alleles = input_df.columns
    if not isinstance(fsa, dict):
        raise MEDICCError("fsa input to create_df_from_fsa has to be a dict"
                          "Input type is {}".format(type(fsa)))

    nr_alleles = len(alleles)
    samples = input_df.index.get_level_values('sample_id').unique()
    segment_index = input_df.xs(samples[0], level='sample_id').index
    internal_frames = []
    for node in fsa:
        if node in samples:
            continue
        cns = tools.fsa_to_string(fsa[node]).split(separator)
        if len(cns) % nr_alleles != 0:
            raise MEDICCError('For sample {} we have {} haplotype-specific chromosomes for {} alleles'
                              '\nnumber of chromosomes has to be divisible by nr of alleles'.format(node,
                                                                                                    len(cns),
                                                                                                    nr_alleles))
        nr_chroms = int(len(cns) // nr_alleles)
        node_cn = {}
        for i, allele in enumerate(alleles):
            node_cn[allele] = list(
                ''.join(cns[(i*nr_chroms):((i+1)*nr_chroms)]))
        internal_frames.append(pd.DataFrame(node_cn, index=segment_index))

    if not internal_frames:
        return input_df.sort_index()

    internal_nodes = [
        node for node in fsa
        if node not in samples
    ]
    internal_df = pd.concat(
        internal_frames, keys=internal_nodes, names=['sample_id'])
    internal_df.index = internal_df.index.set_names(
        ['sample_id', 'chrom', 'start', 'end'])

    return pd.concat([input_df, internal_df], axis=0).sort_index()


def create_df_from_phasing_fsa(input_df: pd.DataFrame, fsas, separator: str = 'X'):
    """ 
    Takes a two FSAs dicts from phasing and extracts the copy number profiles.
    The allele names are taken from the input_df columns and the returned data frame has the same 
    number of rows and row index as the input_df. """

    alleles = input_df.columns
    if len(fsas) != 2:
        raise MEDICCError("fsas has to be of length 2")
    if not all([isinstance(fsa, dict) for fsa in fsas]):
        raise MEDICCError("all fsas entries have to be dicts")
    if fsas[0].keys() != fsas[1].keys():
        raise MEDICCError("fsas keys have to be the same")


    output_df = input_df.copy()[[]]
    output_df[alleles] = ''

    for sample in fsas[0].keys():
        cns_a = tools.fsa_to_string(fsas[0][sample]).split(separator)
        cns_b = tools.fsa_to_string(fsas[1][sample]).split(separator)
        if len(cns_a) != len(cns_b):
            raise MEDICCError(f"length of alleles is not the same for sample {sample}")

        output_df.loc[sample, alleles[0]] = list(''.join(cns_a))
        output_df.loc[sample, alleles[1]] = list(''.join(cns_b))

    # output_df = output_df.stack('sample_id')
    # output_df = output_df.reorder_levels(['sample_id', 'chrom', 'start', 'end']).sort_index()
    
    return output_df


def shorten_cn_strings(string_1, string_2):
    '''
    Takes two strings string_1 and string_2 and removes entires that are consecutive duplicates in both strings.

    Example:
        Input:
            string_1 = "abccd"
            string_2 = "1233d"
        Output:
            string_1_short = "abcd"
            string_2_short = "123d"
    '''
    assert len(string_1) == len(string_2)
    if len(string_1) == 0:
        return '', ''

    string_1_short = [string_1[0]]
    string_2_short = [string_2[0]]
    prev_1 = string_1[0]
    prev_2 = string_2[0]
    for idx in range(1, len(string_1)):
        char_1 = string_1[idx]
        char_2 = string_2[idx]
        if char_1 != prev_1 or char_2 != prev_2:
            string_1_short.append(char_1)
            string_2_short.append(char_2)
        prev_1 = char_1
        prev_2 = char_2

    return ''.join(string_1_short), ''.join(string_2_short)


def parallelization_calc_pairwise_distance(sample_labels, asymm_fst, CN_str_dict, n_cores):
    workers_default = max(1, min(int(n_cores), 2)) if n_cores is not None else 1
    slurm_memory_mb = _env_int("SLURM_MEM_PER_NODE", 0)
    if slurm_memory_mb and slurm_memory_mb < 196608:
        workers_default = 1
    os.environ.setdefault("MEDICC2_PAIRWISE_WORKERS", str(workers_default))
    logger.info("Using memory-bounded pairwise MEDICC implementation; "
                "set MEDICC2_PAIRWISE_WORKERS and MEDICC2_PAIRWISE_BATCH_SIZE to tune.")
    return calc_pairwise_distance_matrix(
        asymm_fst,
        {key: CN_str_dict[key] for key in sample_labels},
        parallel_run=False)


def calc_MED_distance(model_fst, profile_1, profile_2):
    '''
    Calculate the MED distance between two profiles represented as strings.
    '''

    profile_1_short, profile_2_short = shorten_cn_strings(profile_1, profile_2)

    # Convert shrunken string to fsa
    symbol_table = model_fst.input_symbols()
    profile_1_short_fsa = fstlib.factory.from_string(profile_1_short, isymbols=symbol_table, osymbols=symbol_table)
    profile_2_short_fsa = fstlib.factory.from_string(profile_2_short, isymbols=symbol_table, osymbols=symbol_table)

    # Calculate the MED distance
    distance = float(fstlib.kernel_score(model_fst, profile_1_short_fsa, profile_2_short_fsa))

    return distance


def _pairwise_worker_init(model_fst, cn_str_dict):
    global _PAIRWISE_WORKER_MODEL_FST
    global _PAIRWISE_WORKER_CN_STR_DICT
    _PAIRWISE_WORKER_MODEL_FST = model_fst
    _PAIRWISE_WORKER_CN_STR_DICT = cn_str_dict


def _pairwise_chunk_worker(chunk):
    results = []
    for sample_a_idx, sample_b_idx, sample_a, sample_b in chunk:
        cur_dist = calc_MED_distance(
            _PAIRWISE_WORKER_MODEL_FST,
            _PAIRWISE_WORKER_CN_STR_DICT[sample_a],
            _PAIRWISE_WORKER_CN_STR_DICT[sample_b])
        results.append((sample_a_idx, sample_b_idx, cur_dist))
    return results


def _pairwise_chunks(samples, batch_size, completed_mask=None):
    chunk = []
    for sample_a_idx, sample_b_idx in combinations(range(len(samples)), 2):
        if completed_mask is not None and completed_mask[sample_a_idx, sample_b_idx]:
            continue
        chunk.append((sample_a_idx, sample_b_idx,
                      samples[sample_a_idx], samples[sample_b_idx]))
        if len(chunk) >= batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _fill_pairwise_distances(pdm, chunk_results):
    for sample_a_idx, sample_b_idx, cur_dist in chunk_results:
        pdm[sample_a_idx, sample_b_idx] = cur_dist
        pdm[sample_b_idx, sample_a_idx] = cur_dist


def _chunk_waves(chunks, workers):
    wave = []
    for chunk in chunks:
        wave.append(chunk)
        if len(wave) == workers:
            yield wave
            wave = []
    if wave:
        yield wave


def _pairwise_child_worker(connection, model_fst, cn_str_dict, chunk):
    try:
        _pairwise_worker_init(model_fst, cn_str_dict)
        connection.send((True, _pairwise_chunk_worker(chunk)))
    except Exception as exc:
        connection.send((False, repr(exc)))
    finally:
        connection.close()


def _run_pairwise_chunk_in_forked_child(ctx, model_fst, cn_str_dict, chunk):
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_pairwise_child_worker,
                          args=(child_conn, model_fst, cn_str_dict, chunk))
    process.start()
    child_conn.close()
    ok, payload = parent_conn.recv()
    process.join()
    parent_conn.close()
    if not ok:
        raise RuntimeError(payload)
    if process.exitcode != 0:
        raise RuntimeError(f"Pairwise worker exited with status {process.exitcode}")
    return payload


def _external_chunk_payload(cn_str_dict, chunk):
    return [
        (sample_a_idx, sample_b_idx,
         cn_str_dict[sample_a], cn_str_dict[sample_b])
        for sample_a_idx, sample_b_idx, sample_a, sample_b in chunk
    ]


def _run_pairwise_chunks_in_external_workers(model_path, cn_str_dict, chunks):
    workers = []
    for chunk in chunks:
        command = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "pairwise_worker.py"),
            model_path,
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        payload = pickle.dumps(
            _external_chunk_payload(cn_str_dict, chunk),
            protocol=pickle.HIGHEST_PROTOCOL)
        process.stdin.write(payload)
        process.stdin.close()
        workers.append(process)

    all_results = []
    errors = []
    for process in workers:
        stdout = process.stdout.read()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait()
        process.stdout.close()
        process.stderr.close()
        if returncode != 0:
            errors.append(
                f"external pairwise worker exited with status {returncode}: {stderr}")
            continue
        try:
            ok, payload = pickle.loads(stdout)
        except Exception as exc:
            errors.append(
                f"external pairwise worker returned invalid data: {exc!r}; {stderr}")
            continue
        if not ok:
            errors.append(str(payload))
        else:
            all_results.extend(payload)

    if errors:
        raise RuntimeError("; ".join(errors))
    return all_results


def _pairwise_thread_init(model_fst, cn_str_dict):
    global _PAIRWISE_THREAD_MODEL_STATE
    global _PAIRWISE_THREAD_CN_STR_DICT
    _PAIRWISE_THREAD_MODEL_STATE = model_fst.write_to_string()
    _PAIRWISE_THREAD_CN_STR_DICT = cn_str_dict


def _pairwise_thread_chunk_worker(chunk):
    model_fst = getattr(_PAIRWISE_THREAD_LOCAL, "model_fst", None)
    if model_fst is None:
        model_fst = fstlib.Fst.read_from_string(_PAIRWISE_THREAD_MODEL_STATE)
        _PAIRWISE_THREAD_LOCAL.model_fst = model_fst

    results = []
    for sample_a_idx, sample_b_idx, sample_a, sample_b in chunk:
        cur_dist = calc_MED_distance(
            model_fst,
            _PAIRWISE_THREAD_CN_STR_DICT[sample_a],
            _PAIRWISE_THREAD_CN_STR_DICT[sample_b])
        results.append((sample_a_idx, sample_b_idx, cur_dist))
    return results


def _profile_fingerprints(samples, cn_str_dict):
    return [
        hashlib.sha256(cn_str_dict[sample].encode("utf-8")).hexdigest()
        for sample in samples
    ]


def _load_pairwise_checkpoint(path, samples, profile_fingerprints, shape):
    completed = np.eye(shape[0], dtype=bool)
    pdm = np.zeros(shape, dtype=float)
    if not path or not os.path.exists(path):
        return pdm, completed

    with np.load(path, allow_pickle=False) as checkpoint:
        checkpoint_samples = checkpoint["samples"].tolist()
        checkpoint_fingerprints = checkpoint.get(
            "profile_fingerprints", np.asarray([])).tolist()
        if (checkpoint_samples != samples or
                checkpoint_fingerprints != profile_fingerprints):
            logger.warning(
                "Ignoring pairwise checkpoint because its input profiles "
                "do not match the current run.")
            return pdm, completed
        if (checkpoint["pdm"].shape != shape or
                checkpoint["completed"].shape != shape):
            logger.warning(
                "Ignoring pairwise checkpoint because its dimensions "
                "do not match the current run.")
            return pdm, completed
        return checkpoint["pdm"].copy(), checkpoint["completed"].copy()


def _save_pairwise_checkpoint(
        path, samples, profile_fingerprints, pdm, completed):
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".medicc2-pairwise-", suffix=".npz", dir=directory)
    os.close(fd)
    try:
        np.savez_compressed(
            temporary_path,
            samples=np.asarray(samples),
            profile_fingerprints=np.asarray(profile_fingerprints),
            pdm=pdm,
            completed=completed)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _record_completed(completed, chunk_results):
    for sample_a_idx, sample_b_idx, _ in chunk_results:
        completed[sample_a_idx, sample_b_idx] = True
        completed[sample_b_idx, sample_a_idx] = True


def calc_pairwise_distance_matrix(model_fst, cn_str_dict, parallel_run=True):
    samples = list(cn_str_dict.keys())
    checkpoint_path = os.environ.get("MEDICC2_PAIRWISE_CHECKPOINT")
    shape = (len(samples), len(samples))
    profile_fingerprints = _profile_fingerprints(samples, cn_str_dict)
    pdm, completed_mask = _load_pairwise_checkpoint(
        checkpoint_path, samples, profile_fingerprints, shape)
    ncombs = len(samples) * (len(samples) - 1) // 2
    completed = int(np.count_nonzero(np.triu(completed_mask, k=1)))
    next_log_percentage = 10
    batch_size = max(1, _env_int("MEDICC2_PAIRWISE_BATCH_SIZE", 64))
    workers = max(1, _env_int("MEDICC2_PAIRWISE_WORKERS", 1))
    mode = os.environ.get("MEDICC2_PAIRWISE_MODE", "external").strip().lower()
    chunks = _pairwise_chunks(
        samples, batch_size, completed_mask=completed_mask)

    if mode == "external" and ncombs > completed:
        logger.info("Calculating pairwise MEDICC distances in fresh external workers "
                    "(pairs=%d, batch_size=%d, workers=%d).",
                    ncombs, batch_size, workers)
        with tempfile.TemporaryDirectory(prefix="medicc2-pairwise-") as temp_dir:
            model_path = os.path.join(temp_dir, "model.fst")
            model_fst.write(model_path)
            if not os.path.isfile(model_path):
                raise RuntimeError("Could not serialize MEDICC model for external workers.")
            for wave in _chunk_waves(chunks, workers):
                chunk_results = _run_pairwise_chunks_in_external_workers(
                    model_path, cn_str_dict, wave)
                _fill_pairwise_distances(pdm, chunk_results)
                _record_completed(completed_mask, chunk_results)
                completed += len(chunk_results)
                _save_pairwise_checkpoint(
                    checkpoint_path, samples, profile_fingerprints,
                    pdm, completed_mask)
                if ncombs > 0:
                    percentage_done = 100 * completed / ncombs
                    if percentage_done >= next_log_percentage:
                        logger.info(f'{percentage_done:.2f}')
                        next_log_percentage += 10
        return pd.DataFrame(pdm, index=samples, columns=samples)

    if mode == "threads" and ncombs > completed:
        logger.info("Calculating pairwise MEDICC distances with native threads "
                    "(pairs=%d, batch_size=%d, workers=%d).",
                    ncombs, batch_size, workers)
        _pairwise_thread_init(model_fst, cn_str_dict)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for wave in _chunk_waves(chunks, workers):
                for chunk_results in executor.map(
                        _pairwise_thread_chunk_worker, wave):
                    _fill_pairwise_distances(pdm, chunk_results)
                    _record_completed(completed_mask, chunk_results)
                    completed += len(chunk_results)
                _save_pairwise_checkpoint(
                    checkpoint_path, samples, profile_fingerprints,
                    pdm, completed_mask)
                if ncombs > 0:
                    percentage_done = 100 * completed / ncombs
                    if percentage_done >= next_log_percentage:
                        logger.info(f'{percentage_done:.2f}')
                        next_log_percentage += 10
        return pd.DataFrame(pdm, index=samples, columns=samples)

    if mode == "forked" and ncombs > completed and "fork" in mp.get_all_start_methods():
        logger.info("Calculating pairwise MEDICC distances in forked batches "
                    "(pairs=%d, batch_size=%d, workers=%d).",
                    ncombs, batch_size, workers)
        ctx = mp.get_context("fork")
        if workers != 1:
            logger.warning("MEDICC2_PAIRWISE_WORKERS=%d requested, but forked mode "
                           "currently runs one batch process at a time.", workers)
        for chunk in chunks:
            chunk_results = _run_pairwise_chunk_in_forked_child(
                ctx, model_fst, cn_str_dict, chunk)
            _fill_pairwise_distances(pdm, chunk_results)
            _record_completed(completed_mask, chunk_results)
            completed += len(chunk_results)
            _save_pairwise_checkpoint(
                checkpoint_path, samples, profile_fingerprints,
                pdm, completed_mask)
            if ncombs > 0:
                percentage_done = 100 * completed / ncombs
                if percentage_done >= next_log_percentage:
                    logger.info(f'{percentage_done:.2f}')
                    next_log_percentage += 10
        return pd.DataFrame(pdm, index=samples, columns=samples)

    logger.info("Calculating pairwise MEDICC distances in-process "
                "(pairs=%d, mode=%s).", ncombs, mode)
    for chunk in chunks:
        _pairwise_worker_init(model_fst, cn_str_dict)
        chunk_results = _pairwise_chunk_worker(chunk)
        _fill_pairwise_distances(pdm, chunk_results)
        _record_completed(completed_mask, chunk_results)
        completed += len(chunk_results)
        _save_pairwise_checkpoint(
            checkpoint_path, samples, profile_fingerprints,
            pdm, completed_mask)

        if ncombs > 0:
            percentage_done = 100 * completed / ncombs
            if percentage_done >= next_log_percentage:
                logger.info(f'{percentage_done:.2f}')
                next_log_percentage += 10

    return pd.DataFrame(pdm, index=samples, columns=samples)


def infer_tree_topology(pairwise_distances, labels, normal_name):
    if len(labels) > 2:
        tree = nj.NeighbourJoining(pairwise_distances, labels).tree

        tmpsearch = [c for c in tree.find_clades(name = normal_name)]
        normal_node = tmpsearch[0]
        root_path = tree.get_path(normal_node)[::-1]

        if len(root_path)>1:
            new_root = root_path[1]
            tree.root_with_outgroup(new_root)
    else:
        clade_ancestor = Bio.Phylo.PhyloXML.Clade(branch_length=0, name='internal_1')
        clade_ancestor.clades = [Bio.Phylo.PhyloXML.Clade(
            name=label, branch_length=0 if label == normal_name else 1) for label in labels]

        tree = Bio.Phylo.PhyloXML.Phylogeny(root=clade_ancestor)
        tree.root_with_outgroup(normal_name)

    return tree


def update_branch_lengths(tree, fst, ancestor_fsa, normal_name='diploid'):
    """ Updates the branch lengths in the tree using the internal nodes supplied in the FSA dict 
    """
    if len(ancestor_fsa) == 2:
        child_clade = [x for x in tree.find_clades() if x.name is not None and x.name != normal_name][0]
        child_clade.branch_length = float(fstlib.score(
            fst, ancestor_fsa[normal_name], ancestor_fsa[child_clade.name]))

    if not isinstance(ancestor_fsa, dict):
        raise MEDICCError("input ancestor_fsa to function update_branch_lengths has to be either a dict"
                          "provided type is {}".format(type(ancestor_fsa)))

    def _distance_to_child(fst, ancestor_fsa, sample_1, sample_2):
        return float(fstlib.score(fst, ancestor_fsa[sample_1], ancestor_fsa[sample_2]))

    for clade in tree.find_clades():
        if clade.name is None:
            continue
        children = clade.clades
        if len(children) != 0:
            for child in children:
                if child.name == normal_name:  # exception: evolution goes from diploid to internal node
                    logger.debug(f'Updating MRCA branch length from {child.name} to {clade.name}')
                    brs = _distance_to_child(fst, ancestor_fsa, child.name, clade.name)
                else:
                    logger.debug(f'Updating branch length from {clade.name} to {child.name}')
                    brs = _distance_to_child(fst, ancestor_fsa, clade.name, child.name)
                logger.debug(f'branch length: {brs}')
                child.branch_length = brs


def summarize_patient(tree, pdm, sample_labels, normal_name='diploid', events_df=None):
    """Calculate several summary values for the provided samples

    Args:
        tree (Bio.Phylo.Tree): Phylogenetic tree
        pdm (pandas.DataFrame): Pairwise distance matrix between the samples
        sample_labels (list): List of all samples
        normal_name (str, optional): Name of normal sample. Defaults to 'diploid'.
        events_df (pandas.DataFrame, optional): DataFrame containg all copy-number events. Defaults to None.

    Returns:
        pandas.DataFrame: Summary DataFrame
    """    
    branch_lengths = []
    for parent in tree.find_clades(terminal=False, order="level"):
        for child in parent.clades:
            if child.branch_length:
                branch_lengths.append(child.branch_length)

    nsamples = len(sample_labels)
    tree_length = np.sum(branch_lengths) if len(branch_lengths) > 0 else None
    avg_branch_length = np.mean(branch_lengths) if len(branch_lengths) > 0 else None
    min_branch_length = np.min(branch_lengths) if len(branch_lengths) > 0 else None
    max_branch_length = np.max(branch_lengths) if len(branch_lengths) > 0 else None
    median_branch_length = np.median(branch_lengths) if len(branch_lengths) > 0 else None
    # p_star = stats.star_topology_test(pdm)
    # p_clock = stats.molecular_clock_test(pdm,
    #                                      np.flatnonzero(np.array(sample_labels) == normal_name)[0])
    if events_df is None:
        wgd_status = "unknown (run with --events flag to detect WGDs)"
    else:
        if "wgd" in events_df['type'].values:
            wgd_status = "WGD on branch " + \
                "and ".join(events_df.loc[events_df['type'] ==
                                          'wgd'].index.get_level_values('sample_id'))
        else:
            wgd_status = "no WGD"

    result = pd.Series({
        'nsamples': nsamples,
        'normal_name': normal_name,
        'tree_length': tree_length,
        'mean_branch_length': avg_branch_length,
        'median_branch_length': median_branch_length,
        'min_branch_length': min_branch_length,
        'max_branch_length': max_branch_length,
        # 'p_star': p_star,
        # 'p_clock': p_clock,
        'wgd_status': wgd_status,
    })
    
    return result


def detect_wgd(input_df, sample, total_cn=False, wgd_x2=False, n_wgd=None):
    if n_wgd is not None and n_wgd > 2:
        raise NotImplementedError("MEDICC can only detect WGDs with n_wgd <= 2")

    if n_wgd is None:
        wgd_fst = io.read_fst(total_copy_numbers=total_cn, wgd_x2=wgd_x2, n_wgd=n_wgd)
        no_wgd_fst = io.read_fst(no_wgd=True)
    elif n_wgd == 1:
        wgd_fst = io.read_fst(total_copy_numbers=total_cn, wgd_x2=wgd_x2, n_wgd=2)
        no_wgd_fst = io.read_fst(total_copy_numbers=total_cn, wgd_x2=wgd_x2, n_wgd=1)
    elif n_wgd == 2:
        wgd_fst = io.read_fst(total_copy_numbers=total_cn, wgd_x2=wgd_x2, n_wgd=None)
        no_wgd_fst = io.read_fst(total_copy_numbers=total_cn, wgd_x2=wgd_x2, n_wgd=2)

    diploid_fsa = medicc.tools.create_diploid_fsa(no_wgd_fst)
    symbol_table = no_wgd_fst.input_symbols()
    fsa_dict, _ = medicc.create_standard_fsa_dict_from_data(input_df.loc[[sample]],
                                                         symbol_table, 'X')

    distance_wgd = float(fstlib.score(wgd_fst, diploid_fsa, fsa_dict[sample]))
    distance_no_wgd = float(fstlib.score(no_wgd_fst, diploid_fsa, fsa_dict[sample]))

    return distance_wgd < distance_no_wgd


class MEDICCError(Exception):
    pass
