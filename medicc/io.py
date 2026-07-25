import logging
import os

import Bio
import fstlib
import matplotlib
import numpy as np
import pandas as pd

from medicc import plot, tools

logger = logging.getLogger(__name__)


def read_and_parse_input_data(filename, normal_name='diploid', input_type='tsv', separator='X',
                              chrom_column='chrom', allele_columns=['cn_a', 'cn_b'], maxcn=8,
                              total_copy_numbers=False):
    
    if maxcn > 8:
        raise MEDICCIOError("Maximum copy number must be <= 8.")

    if len(allele_columns) == 1 and not total_copy_numbers:
        logger.warning('You have provided only one allele column but the --total-copy-numbers flag was not set')
    if total_copy_numbers and not len(allele_columns) == 1:
        raise MEDICCIOError("You have set the --total-copy-numbers flag but provided more than one allele column. "
                            "Set allele columns with the flag --input-allele-columns")

    ## Read in input data
    if input_type.lower() == "fasta" or input_type.lower() == 'f':
        logger.info("Reading FASTA input.")
        input_df = _read_fasta_as_dataframe(filename, separator=separator, allele_columns=allele_columns, maxcn=maxcn)
    elif input_type.lower() == "tsv" or input_type.lower() == 't':
        logger.info("Reading TSV input.")
        input_df = _read_tsv_as_dataframe(
            filename, allele_columns=allele_columns, maxcn=maxcn, chrom_column=chrom_column)
    else:
        raise MEDICCIOError("Unknown input type, possible options are 'fasta' or 'tsv'.")

    input_df.columns.name = 'allele'
    input_df_stacked = input_df.stack('allele').unstack('sample_id').T

    duplicated_entries = input_df_stacked.duplicated(keep=False)
    if duplicated_entries.any():
        logger.warning("Duplicated entries found in input data: "
                    f"{input_df_stacked.index[duplicated_entries]}")

    normal_value = '2' if total_copy_numbers else '1'
    normal_samples = np.setdiff1d(input_df_stacked.index[
        (input_df_stacked == normal_value).all(axis=1)], normal_name)
    if len(normal_samples) > 0:
        logger.warning(f"Diploid samples found in input data: {normal_samples}")
    if len(normal_samples) == len(np.unique(input_df.index.get_level_values('sample_id'))):
        logger.warning("All samples are diploid! MEDICC2 results will be meaningless!")
    
    ## Add normal sample if needed
    input_df = add_normal_sample(input_df, normal_name, allele_columns=allele_columns, 
                                    total_copy_numbers=total_copy_numbers, chrom_column=chrom_column)
    nsamples = input_df.index.get_level_values('sample_id').unique().shape[0]
    nchr = input_df.index.get_level_values(chrom_column).unique().shape[0]
    nsegs = input_df.loc[normal_name,:].shape[0]
    logger.info(f"Read {nsamples} samples, {nchr} chromosomes, {nsegs} segments per sample")

    gaps = (input_df.loc[normal_name].eval('start') - 
            np.roll(input_df.loc[normal_name].eval('end'), 1)).values
    total_gaps = gaps[gaps>0].sum()
    if total_gaps > 1e8:
        logger.warning(f"Total of {total_gaps:.1e} bp gaps in the segmentation. Large gaps might "
                    "affect the performance of MEDICC2.")

    return input_df


def read_fst(user_fst=None, no_wgd=False, n_wgd=None, total_copy_numbers=False, wgd_x2=False,
             force_wgd=False):
    """Simple wrapper for loading the FST using the fstlib read function. """

    objects_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "objects")

    fst_paths = {
        (True, None, False, False, False): os.path.join(objects_dir, 'no_wgd_asymm.fst'),
        (True, None, True, False, False): os.path.join(objects_dir, 'no_wgd_asymm.fst'),
        (False, None, False, False, False): os.path.join(objects_dir, 'wgd_asymm.fst'),
        (False, None, False, True, False): os.path.join(objects_dir, 'wgd_x2_asymm.fst'),
        (False, None, True, False, False): os.path.join(objects_dir, 'wgd_total_cn_asymm.fst'),
        (False, 1, False, False, False): os.path.join(objects_dir, 'wgd_1_asymm.fst'),
        (False, 2, False, False, False): os.path.join(objects_dir, 'wgd_2_asymm.fst'),
        (False, 3, False, False, False): os.path.join(objects_dir, 'wgd_3_asymm.fst'),
        (False, 1, False, True, False): os.path.join(objects_dir, 'wgd_x2_1_asymm.fst'),
        (False, 1, True, False, False): os.path.join(objects_dir, 'wgd_total_cn_1_asymm.fst'),
        (False, None, False, False, True): os.path.join(objects_dir, 'forced_1_wgd_asymm.fst'),
        (False, None, True, False, True): os.path.join(objects_dir, 'forced_1_wgd_total_cn_asymm.fst'),
        (False, None, False, True, True): os.path.join(objects_dir, 'forced_1_wgd_x2_asymm.fst'),
    }

    if user_fst is not None:
        fst_path = user_fst
    else:
        fst_key = (no_wgd, n_wgd, total_copy_numbers, wgd_x2, force_wgd)
        if fst_key not in fst_paths:
            raise MEDICCIOError("Invalid combination of the following parameters for loading the FST: "
                                "no_wgd, n_wgd, total_copy_numbers, wgd_x2, force_wgd")
        fst_path = fst_paths[fst_key]
    
    # elif no_wgd:
    #     fst_path = os.path.join(objects_dir, 'no_wgd_asymm.fst')
    # elif wgd_x2:
    #     fst_path = os.path.join(objects_dir, 'wgd_x2_asymm.fst')
    #     if n_wgd == 1:
    #         fst_path = os.path.join(objects_dir, 'wgd_x2_1_asymm.fst')
    # elif total_copy_numbers:
    #     fst_path = os.path.join(objects_dir, 'wgd_total_cn_asymm.fst')
    #     if n_wgd == 1:
    #         fst_path = os.path.join(objects_dir, 'wgd_total_cn_1_asymm.fst')
    # elif n_wgd is not None and int(n_wgd) <= 3:
    #     fst_path = os.path.join(objects_dir, 'wgd_{}_asymm.fst'.format(int(n_wgd)))
    # else:
    #     fst_path = os.path.join(objects_dir, 'wgd_asymm.fst')

    return fstlib.read(fst_path)


def load_main_fsts(return_symbol_table=False):
    asymm_fst = read_fst()
    asymm_fst_nowgd = read_fst(no_wgd=True)
    asymm_fst_1_wgd = read_fst(n_wgd=1)
    asymm_fst_2_wgd = read_fst(n_wgd=2)

    if return_symbol_table:
        symbol_table = asymm_fst.input_symbols()
        return asymm_fst, asymm_fst_nowgd, asymm_fst_1_wgd, asymm_fst_2_wgd, symbol_table
    else:
        return asymm_fst, asymm_fst_nowgd, asymm_fst_1_wgd, asymm_fst_2_wgd


def validate_input(input_df, symbol_table=None, normal_name='diploid'):
    """Validate the input DataFrame."""

    # Check that normal sample is present
    if normal_name not in input_df.index.get_level_values('sample_id').unique():
        raise MEDICCIOError(f"Normal sample '{normal_name}' not found in input data. Specify a "
                            "different name using the --normal-name flag.")
    
    if input_df.index.get_level_values('sample_id').nunique() <= 2:
        raise MEDICCIOError("MEDICC2 requires at least 2 non-diploid samples to run.")

    # Check if the index names are correct
    if input_df.index.names != ['sample_id', 'chrom', 'start', 'end']:
        raise MEDICCIOError("DataFrame must be indexed by ['sample_id', 'chrom', 'start', 'end'].")

    # Check if the chromosome is categorical
    if not isinstance(input_df.index.dtypes['chrom'], pd.CategoricalDtype):
        raise MEDICCIOError("""
            Chromosome index 'chrom' must be of type pd.CategoricalDtype. 
            You can use medicc.tools.format_chromosomes() or create it yourself.""")

    # Check if the index is sorted
    if not input_df.index.is_monotonic_increasing:
        raise MEDICCIOError("DataFrame index must be sorted.")

    # Check the number of alleles
    if len(input_df.columns)>2:
        raise MEDICCIOError("More than 2 alleles are currently not supported.")

    # Check if there are any alleles
    if len(input_df.columns)==0:
        raise MEDICCIOError("No alleles found.")
    
    # Check data type allele columns - these should all be of type str (object)
    if not np.all([pd.api.types.is_string_dtype(x) for x in input_df.dtypes]):
        raise MEDICCIOError("Allele columns must be of type: string.")

    # Check data type start and end columns
    if (input_df.index.get_level_values('start').dtype != int or 
        input_df.index.get_level_values('end').dtype != int):
        raise MEDICCIOError("Start and end columns must be of type: integer.")

    # Compare each sample's segment index without materializing a wide
    # sample-by-segment table.
    sample_ids = input_df.index.get_level_values('sample_id').unique()
    reference_segments = input_df.xs(sample_ids[0], level='sample_id').index
    segments_match = all(
        input_df.xs(sample_id, level='sample_id').index.equals(reference_segments)
        for sample_id in sample_ids[1:])
    has_missing_values = any(input_df[column].hasnans for column in input_df)
    if has_missing_values or not segments_match:
        unique_segments = input_df.index.droplevel('sample_id').unique()
        raise MEDICCIOError("The samples have different segments!\n"
                            "Total number of unique segments: {}\n".format(len(unique_segments)))

    if symbol_table is not None:
        # Check if symbols are in symbol table
        alphabet = {x[1] for x in symbol_table}
        data_chars = set(input_df.values.flatten())
        if not data_chars.issubset(alphabet):
            not_in_set = data_chars.difference(alphabet)
            raise MEDICCIOError(f"Not all input symbols are contained in symbol table. Offending symbols: {str(not_in_set)}")


    logger.info('Input data is valid!')


def filter_by_segment_length(input_df, filter_size):
    segment_length = input_df.eval('end+1-start')
    return input_df.loc[segment_length > float(filter_size)]


def _read_tsv_as_dataframe(path, allele_columns=['cn_a','cn_b'], maxcn=8, chrom_column='chrom'):
    logger.info(f"Reading TSV file {path}")
    input_file = pd.read_csv(path, sep = "\t")
    input_file["sample_id"] = input_file["sample_id"].astype(str)
    columnn_names = ['sample_id', chrom_column, 'start', 'end'] + allele_columns
    if len(np.setdiff1d(columnn_names, input_file.columns)) > 0:
        raise MEDICCIOError(f"TSV file needs the following columns: sample_id, chrom, start, end and the allele columns ({allele_columns})"
                            "\nMissing columns are: {}".format(
                                np.setdiff1d(columnn_names, input_file.columns)))

    logger.info(f"Successfully read input file. Using columns: {', '.join(columnn_names)}")
    input_file = input_file[columnn_names]
    for c in allele_columns:
        if input_file[c].dtype in [np.dtype('float64'), np.dtype('float32')]:
            logger.warning("Floating point payload! I will round, but this might not be intended.")
            input_file[c] = input_file[c].round().astype('int')
        if input_file[c].dtype in [np.dtype('int64'), np.dtype('int32')]:
            if np.any(input_file[c]>maxcn):
                logger.warning(f"Integer CN > maxcn {maxcn}, capping.")
                input_file[c] = np.fmin(input_file[c], maxcn)
    input_file[chrom_column] = tools.format_chromosomes(input_file[chrom_column])
    if any(["Y" in str(x) for x in input_file[chrom_column].unique()]):
        logger.warning("Y chromosome detected in input. This might cause errors down the line!")
    input_file.set_index(['sample_id', chrom_column, 'start', 'end'], inplace=True)
    input_file.sort_index(inplace=True)
    input_file[allele_columns] = input_file[allele_columns].astype(str)

    return input_file


def _read_fasta_as_dataframe(infile: str, separator: str = 'X', allele_columns = ['cn_a','cn_b'], maxcn: int = 8):
    """Reads FASTA decriptor file (old MEDICC input format) and reads the corresponding FASTA files to generate
    a data frame with the same format as the input TSV format. """
    logger.info(f"Reading FASTA dataset from description file {infile}.")
    description_file = pd.read_csv(infile,
                                    delim_whitespace = True,
                                    header = None,
                                    names = ["chrom",] + allele_columns,
                                    usecols = ["chrom",] + allele_columns)
    description_file.set_index('chrom', inplace=True)
    description_file.columns.name='allele'
    description_file = description_file.stack()
    inpath = os.path.dirname(infile)

    payload = []
    for filename in description_file:
        with open(os.path.join(inpath, filename), 'r') as fd:
            text = fd.read()
            df = pd.DataFrame([s.strip().split('\n') for s in text.split('>') if s.strip() !=''], columns=['sample_id', 'cnp'])
            df.set_index('sample_id', inplace=True)
            payload.append(df)

    payload = pd.concat(payload, keys=description_file.index, names=description_file.index.names)

    ## deal with the case where multiple chromosomes are given in one string
    payload=payload.cnp.str.split(separator, expand=True)
    if payload.shape[1]>1: ## multiple columns
        payload.columns.name='id'
        payload=payload.unstack('chrom').reorder_levels(['chrom','id'],axis=1)
        payload.columns = [f"{c}{i+1}" for c,i in payload.columns]
        payload.columns.name = 'chrom'
        payload = payload.stack()
        payload = payload.reorder_levels([1,2,0]).sort_index()
    else:
        payload = payload.iloc[:,0]
        payload = payload.reorder_levels([2,0,1]).sort_index()

    payexpand = [pd.DataFrame(list(s), columns=['cn']) for s in payload]
    paylong = pd.concat(payexpand, keys=payload.index, names=payload.index.names + ['segment'])
    paylong['cn'] = paylong['cn'].apply(tools.hex2int)
    if np.any(paylong['cn'] > maxcn):
        logger.warning(f"Integer CN > maxcn {maxcn}, capping.")
        paylong['cn'] = np.fmin(paylong['cn'], maxcn)
    paylong['cn'] = paylong['cn'].astype(str)
    result = paylong.unstack(['allele'])
    result = result.droplevel(0, axis=1).reset_index()
    result.columns.name = None
    result['start'] = result['segment']
    result['end'] = result['start']+1
    result = result[['sample_id','chrom','start','end'] + allele_columns]
    result['chrom'] = tools.format_chromosomes(result['chrom'])
    result.sort_values(['sample_id', 'chrom', 'start', 'end'], inplace=True)
    result.set_index(['sample_id', 'chrom', 'start', 'end'], inplace=True)
    result.sort_index(inplace=True)

    return result

def add_normal_sample(df, normal_name, allele_columns=['cn_a','cn_b'], total_copy_numbers=False,
                      chrom_column='chrom'):
    """Adds an artificial normal samples with the supplied name to the data frame.
    The normal sample has CN=1 on all supplied alleles. """
    samples = df.index.get_level_values('sample_id').unique()

    if total_copy_numbers:
        normal_value = '2'
    else:
        normal_value = '1'

    if normal_name is not None and normal_name not in samples:
        logger.info(f"Normal sample '{normal_name}' not found, adding artifical normal by the name: '{normal_name}'.")
        tmp = df.unstack('sample_id')
        for col in allele_columns:
            tmp.loc[:, (col, normal_name)] = normal_value
        tmp = tmp.stack('sample_id')
        tmp = tmp.reorder_levels(['sample_id', chrom_column, 'start', 'end']).sort_index()
    else:
        logger.info(f"Sample '{normal_name}' was found in data is is used as normal")
        if np.any(df.loc[normal_name] == '0'):
            logger.warning("The provided normal sample contains segments with copy number 0. "
                        "If any other sample has non-zero values in these segments, MEDICC will crash")
        if np.any(df.loc[normal_name] != normal_value):
            logger.warning("The provided normal sample contains segments with copy number != {}.".format(normal_value))

        tmp = df

    return tmp


def write_tree_files(tree, out_name: str, plot_tree=True, draw_ascii=False, normal_name='diploid'):
    """Writes a Newick, PhyloXML, Ascii graphic and PNG grahic file of the tree. """
    Bio.Phylo.write(tree, out_name + ".new", "newick")
    Bio.Phylo.write(tree, out_name + ".xml", "phyloxml")

    if draw_ascii:
        with open(out_name + ".txt", "w") as f:
            Bio.Phylo.draw_ascii(tree, file = f)

    if plot_tree:
        plot.plot_tree(tree,
                       output_name=out_name,
                       normal_name=normal_name,
                       label_func=lambda x: x if 'internal' not in x else '',
                       show_branch_lengths=True)


def write_branch_lengths(tree, out_name: str):
    """Writes a file with the branch lengths of the tree."""
    with open(out_name, "w") as f:
        for node in tree.find_clades():
            if node.name is not None and node.name != 'diploid':
                f.write("{}\t{}\n".format(node.name, node.branch_length))


def write_pairwise_distances(sample_labels, pairwise_distances, filename_prefix):
    """Write the pairwise distance matrix as a tsv."""

    if not isinstance(pairwise_distances, pd.DataFrame):
        pairwise_distances = pd.DataFrame(pairwise_distances, columns=sample_labels, index=sample_labels)
    pairwise_distances.to_csv(f"{filename_prefix}.tsv", sep='\t')


def import_tree(tree_file, normal_name='diploid', file_format='newick', quality_checks=True):
    """Loads a phylogenetic tree in the given format and roots it at the normal sample. """
    tree = Bio.Phylo.read(tree_file, file_format)
    input_tree = Bio.Phylo.BaseTree.copy.deepcopy(tree)
    tmpsearch = [c for c in input_tree.find_clades(name = normal_name)]
    if len(tmpsearch) == 0:
        raise ValueError(f"normal name '{normal_name}' not found in tree")
    normal_name = tmpsearch[0]
    root_path = input_tree.get_path(normal_name)[::-1]

    if len(root_path) > 1:
        new_root = root_path[1]
        input_tree.root_with_outgroup(new_root)
    else:
        pass

    if quality_checks:
        node_names = [c.name for c in input_tree.find_clades() if c.name is not None]
        if len(node_names) != len(np.unique(node_names)):
            raise ValueError("Internal node names of provided tree are not unique. Please provide a tree with unique node names.")
        if any([clade.name is None for clade in input_tree.get_terminals()]):
            raise ValueError("Some samples in the tree do not have names. Please provide a tree with names for all samples. Names must contain at least one letter.")
        if sum([clade.name is None for clade in input_tree.get_nonterminals()]) > 1:
            raise ValueError("Some internal nodes in the tree do not have names. Please provide a tree with names for all samples. Names must contain at least one letter.")
        if any([len(clade.clades) != 2 for clade in input_tree.get_nonterminals()]):
            raise ValueError("Some internal nodes in the tree do not have exactly 2 children. Please provide a binary tree.")
    
    return input_tree


def read_bed_file(filename):

    data = pd.read_csv(filename, header=None, comment='#', sep='\t')

    if len(data.columns) != 4:
        print('WARNING: expected 4 columns')
        return None

    data.columns = ['Chromosome', 'Start', 'End', 'name']
    return data


class MEDICCIOError(Exception):
    pass
