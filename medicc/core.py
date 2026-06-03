import copy
import logging
import multiprocessing as mp
import os
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

    ## Compile input data into FSAs stored in dictionaries
    logger.info("Compiling input sequences into FSAs.")
    FSA_dict, CN_str_dict = create_standard_fsa_dict_from_data(input_df, symbol_table, chr_separator)
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

        pairwise_distances = pd.DataFrame(0, columns=FSA_dict.keys(), index=FSA_dict.keys())

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

    fsa_dict = {}
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
        cn_str = aggregate_copy_number_profile(cnp)
        fsa_dict[taxon] = fstlib.factory.from_string(cn_str,
                                                     arc_type="standard",
                                                     isymbols=symbol_table,
                                                     osymbols=symbol_table)
        cn_str_dict[taxon] = cn_str

    return fsa_dict, cn_str_dict


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
    output_df = input_df.unstack('sample_id')

    # Create dict and concat later to prevent pandas PerformanceWarning
    internal_cns = dict()
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
        for i, allele in enumerate(alleles):
            cn = list(''.join(cns[(i*nr_chroms):((i+1)*nr_chroms)]))
            internal_cns[(allele, node)] = cn

    internal_cns_df = pd.DataFrame(internal_cns, index=output_df.index)
    internal_cns_df.columns.names = ['allele', 'sample_id']
    output_df = (pd.concat([output_df, internal_cns_df], axis=1)
                 .stack('sample_id')
                 .reorder_levels(['sample_id', 'chrom', 'start', 'end'])
                 .sort_index())

    return output_df


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


def _pairwise_chunks(samples, batch_size):
    chunk = []
    for sample_a_idx, sample_b_idx in combinations(range(len(samples)), 2):
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


def calc_pairwise_distance_matrix(model_fst, cn_str_dict, parallel_run=True):
    samples = list(cn_str_dict.keys())
    pdm = np.zeros((len(samples), len(samples)), dtype=float)
    ncombs = len(samples) * (len(samples) - 1) // 2
    next_log_percentage = 10
    batch_size = max(1, _env_int("MEDICC2_PAIRWISE_BATCH_SIZE", 64))
    workers = max(1, _env_int("MEDICC2_PAIRWISE_WORKERS", 1))
    mode = os.environ.get("MEDICC2_PAIRWISE_MODE", "forked").strip().lower()

    if mode == "forked" and ncombs > batch_size and "fork" in mp.get_all_start_methods():
        logger.info("Calculating pairwise MEDICC distances in forked batches "
                    "(pairs=%d, batch_size=%d, workers=%d).",
                    ncombs, batch_size, workers)
        completed = 0
        ctx = mp.get_context("fork")
        chunks = list(_pairwise_chunks(samples, batch_size))
        with ctx.Pool(processes=workers,
                      initializer=_pairwise_worker_init,
                      initargs=(model_fst, cn_str_dict),
                      maxtasksperchild=1) as pool:
            for chunk_results in pool.imap_unordered(_pairwise_chunk_worker, chunks):
                _fill_pairwise_distances(pdm, chunk_results)
                completed += len(chunk_results)
                if ncombs > 0:
                    percentage_done = 100 * completed / ncombs
                    if percentage_done >= next_log_percentage:
                        logger.info(f'{percentage_done:.2f}')
                        next_log_percentage += 10
        return pd.DataFrame(pdm, index=samples, columns=samples)

    logger.info("Calculating pairwise MEDICC distances in-process "
                "(pairs=%d, mode=%s).", ncombs, mode)
    for i, chunk in enumerate(_pairwise_chunks(samples, batch_size), start=1):
        _pairwise_worker_init(model_fst, cn_str_dict)
        chunk_results = _pairwise_chunk_worker(chunk)
        _fill_pairwise_distances(pdm, chunk_results)

        completed = min(i * batch_size, ncombs)
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
