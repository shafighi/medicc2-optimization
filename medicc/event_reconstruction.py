import copy
import logging
import os

import fstlib
import numpy as np
import pandas as pd

from medicc import io, tools

# prepare logger 
logger = logging.getLogger(__name__)


def _load_event_fsts(total_cn=False, wgd_x2=False, no_wgd=False):
    event_fsts = list(io.load_main_fsts(return_symbol_table=True))
    if total_cn:
        event_fsts[0] = io.read_fst(total_copy_numbers=True)
        event_fsts[2] = io.read_fst(total_copy_numbers=True, n_wgd=1)
        event_fsts[3] = None
    elif wgd_x2:
        event_fsts[0] = io.read_fst(wgd_x2=True)
        event_fsts[2] = io.read_fst(wgd_x2=True, n_wgd=1)
        event_fsts[3] = None
    elif no_wgd:
        event_fsts[0] = io.read_fst(no_wgd=True)
        event_fsts[2] = None
        event_fsts[3] = None
    return tuple(event_fsts)


def calculate_all_cn_events(tree, cur_df, alleles=['cn_a', 'cn_b'], normal_name='diploid',
                            wgd_x2=False, no_wgd=False, total_cn=False, max_wgd=1):
    """Create a DataFrame containing all copy-number events in the current data

    Args:
        tree (Bio.Phylo.Tree): Phylogenetic tree created by MEDICC2's tree reconstruction
        cur_df (pandas.DataFrame): DataFrame containing the copy-numbers of the samples and internal nodes
        alleles (list, optional): List of alleles. Defaults to ['cn_a', 'cn_b'].
        normal_name (str, optional): Name of the normal sample. Defaults to 'diploid'.

    Returns:
        pandas.DataFrame: Updated copy-number DataFrame
        pandas.DataFrame: DataFrame of copy-number events
    """
    
    cur_df[['is_gain', 'is_loss', 'is_wgd']] = False
    cur_df[alleles] = cur_df[alleles].astype(int)
    if tree == None:
        cur_df[['is_normal', 'is_clonal']] = False
        events = None
    else:
        event_columns = [
            'sample_id', 'chrom', 'start', 'end', 'allele', 'type', 'cn_child']
        event_frames = []
        event_fsts = _load_event_fsts(
            total_cn=total_cn, wgd_x2=wgd_x2, no_wgd=no_wgd)

        clades = [x for x in tree.find_clades()]

        for clade in clades:
            if not len(clade.clades):
                continue
            if clade.name is None:
                clade = copy.deepcopy(clade)
                clade.name = normal_name
            for child in clade.clades:
                if child.branch_length == 0:
                    continue

                cur_df, cur_events = calculate_cn_events_per_branch(
                    cur_df, clade.name, child.name, alleles=alleles, wgd_x2=wgd_x2,
                    total_cn=total_cn, no_wgd=no_wgd, normal_name=normal_name,
                    max_wgd=max_wgd, event_fsts=event_fsts, copy_input=False)

                event_frames.append(cur_events)

        events = pd.concat(
            [pd.DataFrame(columns=event_columns)] + event_frames,
            ignore_index=True)

        event_flags = ['is_loss', 'is_gain', 'is_wgd']
        segment_levels = ['chrom', 'start', 'end']
        is_normal = ~(
            cur_df[event_flags]
            .groupby(level=segment_levels, sort=True)
            .any()
            .any(axis=1))
        is_normal.name = 'is_normal'
        mrca = [x for x in tree.root.clades if x.name != normal_name][0].name
        non_mrca = cur_df.index.get_level_values('sample_id') != mrca
        is_clonal = ~(
            cur_df.loc[non_mrca, event_flags]
            .groupby(level=segment_levels, sort=True)
            .any()
            .any(axis=1))
        is_clonal.name = 'is_clonal'

        cur_df = cur_df.drop(['is_normal', 'is_clonal'], axis=1, errors='ignore')
        cur_df = (cur_df
                .join(is_normal, how='inner')
                .reorder_levels(['sample_id', 'chrom', 'start', 'end'])
                .sort_index()
                .join(is_clonal, how='inner')
                .reset_index())
        cur_df['chrom'] = tools.format_chromosomes(cur_df['chrom'])
        cur_df = (cur_df
                .set_index(['sample_id', 'chrom', 'start', 'end'])
                .sort_index())

        events = events.set_index(['sample_id', 'chrom', 'start', 'end'])

    return cur_df, events


def calculate_cn_events_per_branch(cur_df, parent_name, child_name, alleles=['cn_a', 'cn_b'],
                                   wgd_x2=False, total_cn=False, no_wgd=False, normal_name='diploid',
                                   max_wgd=1, event_fsts=None, copy_input=True):
    """Calculate copy-number events for a single branch. Used in calculate_all_cn_events

    Args:
        cur_df (pandas.DataFrame): DataFrame containing the copy-numbers of the samples and internal nodes
        parent_name (str): Name of the parent sample
        child_name (str): Name of the child sample
        alleles (list, optional): List of alleles. Defaults to ['cn_a', 'cn_b'].

    Returns:
        pandas.DataFrame: Updated copy-number DataFrame
        pandas.DataFrame: DataFrame of copy-number events
    """

    if copy_input:
        cur_df = cur_df.copy()
    if len(np.setdiff1d(['is_gain', 'is_loss', 'is_wgd'], cur_df.columns)) > 0:
        cur_df[['is_gain', 'is_loss', 'is_wgd']] = False
    if not all(pd.api.types.is_integer_dtype(cur_df[allele]) for allele in alleles):
        cur_df[alleles] = cur_df[alleles].astype(int)

    if event_fsts is None:
        event_fsts = _load_event_fsts(
            total_cn=total_cn, wgd_x2=wgd_x2, no_wgd=no_wgd)
    (asymm_fst, asymm_fst_nowgd, asymm_fst_1_wgd,
     asymm_fst_2_wgd, symbol_table) = event_fsts


    events_df = pd.DataFrame(columns=['sample_id', 'chrom', 'start', 'end', 'allele', 'type', 'cn_child'])

    cur_parent_cn = cur_df.loc[parent_name, alleles].astype(int)
    cur_child_cn = cur_df.loc[child_name, alleles].astype(int)

    def get_int_chrom(x):
        if x == 'chrX':
            return 23
        elif x == 'chrY':
            return 24
        else:
            return int(x.split('chr')[-1])

    cur_chroms = cur_df.loc[normal_name].index.get_level_values(
        'chrom').map(get_int_chrom).values.astype(int)

    # 1. find total loss (loh)
    for allele in alleles:
        parent_loh = cur_parent_cn[allele].values == 0
        cur_loh = cur_child_cn.loc[~parent_loh, allele] == 0
        if cur_loh.sum() == 0:
            continue
        max_previous_cn = np.max(
            np.unique(cur_parent_cn.loc[~parent_loh, allele].loc[cur_loh]))
        for loh_height in np.arange(max_previous_cn):
            parent_loh = cur_parent_cn[allele].values == 0
            cur_loh = cur_child_cn.loc[~parent_loh, allele] == 0
            cn_changes = (cur_child_cn.loc[~parent_loh, allele] - 
                        cur_parent_cn.loc[~parent_loh, allele])

            # LOH detection
            cur_change_location = (cn_changes < 0)

            # + cur_chroms enables detection of chromosome boundaries
            cur_loss_labels_ = ((np.cumsum(np.concatenate([[0], np.diff(
                (cur_change_location.values + cur_chroms[~parent_loh]))])
                * cur_change_location.values) + 1)
                * cur_change_location.values)

            cur_loss_labels = np.zeros_like(cur_loss_labels_)
            for i, j in enumerate(np.setdiff1d(np.unique(cur_loss_labels_), [0])):
                cur_loss_labels[cur_loss_labels_ == j] = i + 1

            loh_regions = np.unique(cur_loss_labels[cur_loh])

            cur_events = (cur_child_cn
                            .loc[~parent_loh]
                            .reset_index()
                            .loc[np.array([np.argmax(cur_loss_labels == val) for val in np.setdiff1d(np.unique(loh_regions), [0])])]
                            [['chrom', 'start', 'end', allele]].values)

            # adjust ends
            cur_events[:, 2] = (cur_child_cn
                                .loc[~parent_loh]
                                .reset_index()
                                .loc[np.array([len(cur_loss_labels) - np.argmax(cur_loss_labels[::-1] == val) - 1 for val in np.setdiff1d(np.unique(loh_regions), [0])])]
                                ['end'].values)
            
            cur_ind = np.arange(len(events_df), len(events_df)+len(cur_events))
            events_df = pd.concat([events_df, pd.DataFrame(index=cur_ind)])
            events_df.loc[cur_ind, 'sample_id'] = child_name
            events_df.loc[cur_ind, 'allele'] = allele
            events_df.loc[cur_ind, 'type'] = 'loss'
            events_df.loc[cur_ind, 'cn_child'] = max_previous_cn - loh_height - 1
            events_df.loc[cur_ind, ['chrom', 'start', 'end']] = cur_events[:, :3]

            cur_parent_cn.loc[cn_changes.loc[np.stack([cur_loss_labels == val for val in loh_regions]).any(axis=0)].index, allele] -= 1

            cur_df.loc[cur_df.loc[[child_name]].loc[~parent_loh].loc[cur_change_location.values].index, 'is_loss'] = True

    loh_pos = (cur_parent_cn == 0)

    # 2. WGDs
    # only check if >30% of is gained
    wgd_candidate_threshold = 0.3

    widths = cur_df.loc[[child_name]].eval('end+1-start')
    fraction_gain = ((cur_df.loc[child_name, alleles] > 1).astype(int).sum(axis=1) * widths.loc[child_name]
                        ).sum() / (2 * widths.loc[child_name].sum())
    parent_fsa = fstlib.factory.from_string('X'.join(['X'.join(["".join(x.astype('str')) for _, x in cur_df.loc[parent_name, alleles][allele].groupby('chrom')]) for allele in alleles]),
                                            arc_type="standard",
                                            isymbols=symbol_table,
                                            osymbols=symbol_table)
    child_fsa = fstlib.factory.from_string('X'.join(['X'.join(["".join(x.astype('str')) for _, x in cur_df.loc[child_name, alleles][allele].groupby('chrom')]) for allele in alleles]),
                                            arc_type="standard",
                                            isymbols=symbol_table,
                                            osymbols=symbol_table)

    score_wgd = float(fstlib.score(asymm_fst, parent_fsa, child_fsa))
    fraction_double_gain = (((cur_df.loc[child_name, alleles] > 2)
                            .astype(int)
                            .sum(axis=1)
                            * widths.loc[child_name]
                            ).sum() / widths.loc[child_name].sum())
    if not no_wgd and fraction_gain > wgd_candidate_threshold:
        if wgd_x2:
            # double wgd
            if (fraction_double_gain > wgd_candidate_threshold) and (float(fstlib.score(asymm_fst_1_wgd, parent_fsa, child_fsa)) != score_wgd):
                cur_parent_cn = 4 * cur_parent_cn
                events_df.loc[len(events_df.index)] = [child_name, 'chr0', cur_df.index.get_level_values('start').min(),
                                                cur_df.index.get_level_values('end').max(), 'both', 'wgd', 0]
                events_df.loc[len(events_df.index)] = [child_name, 'chr0', cur_df.index.get_level_values('start').min(),
                                                    cur_df.index.get_level_values('end').max(), 'both', 'wgd', 0]
                cur_df.loc[child_name, 'is_wgd'] = True
            # single wgd
            elif float(fstlib.score(asymm_fst_nowgd, parent_fsa, child_fsa)) != score_wgd:
                cur_parent_cn = 2 * cur_parent_cn
                events_df.loc[len(events_df.index)] = [child_name, 'chr0', cur_df.index.get_level_values('start').min(),
                                                cur_df.index.get_level_values('end').max(), 'both', 'wgd', 0]
                cur_df.loc[child_name, 'is_wgd'] = True

        elif total_cn:
            # single wgd
            if float(fstlib.score(asymm_fst_nowgd, parent_fsa, child_fsa)) != score_wgd:
                cur_parent_cn[~loh_pos] = cur_parent_cn[~loh_pos] + 2
                events_df.loc[len(events_df.index)] = [child_name, 'chr0', cur_df.index.get_level_values('start').min(),
                                                cur_df.index.get_level_values('end').max(), 'both', 'wgd', 0]
                cur_df.loc[child_name, 'is_wgd'] = True

        else:
            # triple wgd
            if max_wgd >= 3 and (fraction_double_gain > wgd_candidate_threshold) and (float(fstlib.score(asymm_fst_2_wgd, parent_fsa, child_fsa)) != score_wgd):
                cur_parent_cn[~loh_pos] = cur_parent_cn[~loh_pos] + 3
                events_df.loc[len(events_df.index)] = [child_name, 'chr0', cur_df.index.get_level_values('start').min(),
                                                cur_df.index.get_level_values('end').max(), 'both', 'wgd', 0]
                events_df.loc[len(events_df.index)] = [child_name, 'chr0', cur_df.index.get_level_values('start').min(),
                                                    cur_df.index.get_level_values('end').max(), 'both', 'wgd', 0]
                events_df.loc[len(events_df.index)] = [child_name, 'chr0', cur_df.index.get_level_values('start').min(),
                                                    cur_df.index.get_level_values('end').max(), 'both', 'wgd', 0]
                cur_df.loc[child_name, 'is_wgd'] = True
            # double wgd
            elif max_wgd >= 2 and (fraction_double_gain > wgd_candidate_threshold) and (float(fstlib.score(asymm_fst_1_wgd, parent_fsa, child_fsa)) != score_wgd):
                cur_parent_cn[~loh_pos] = cur_parent_cn[~loh_pos] + 2
                events_df.loc[len(events_df.index)] = [child_name, 'chr0', cur_df.index.get_level_values('start').min(),
                                                cur_df.index.get_level_values('end').max(), 'both', 'wgd', 0]
                events_df.loc[len(events_df.index)] = [child_name, 'chr0', cur_df.index.get_level_values('start').min(),
                                                    cur_df.index.get_level_values('end').max(), 'both', 'wgd', 0]
                cur_df.loc[child_name, 'is_wgd'] = True
            # single wgd
            elif max_wgd >= 1 and float(fstlib.score(asymm_fst_nowgd, parent_fsa, child_fsa)) != score_wgd:
                cur_parent_cn[~loh_pos] = cur_parent_cn[~loh_pos] + 1
                events_df.loc[len(events_df.index)] = [child_name, 'chr0', cur_df.index.get_level_values('start').min(),
                                                cur_df.index.get_level_values('end').max(), 'both', 'wgd', 0]
                cur_df.loc[child_name, 'is_wgd'] = True

    # 3. losses and gains
    for allele in alleles:

        cn_changes = (cur_child_cn[allele] - cur_parent_cn[allele]).values
        all_cn_change_vals = np.unique(cn_changes)

        cur_df.loc[child_name, 'is_loss'] = np.logical_or(cur_df.loc[child_name, 'is_loss'].values,
                                                          cn_changes < 0)
        cur_df.loc[child_name, 'is_gain'] = np.logical_or(cur_df.loc[child_name, 'is_gain'].values,
                                                          cn_changes > 0)

        # enumerate over all possible change values
        all_cn_change_vals = np.setdiff1d(np.arange(np.min([np.min(all_cn_change_vals), 0]), np.max(all_cn_change_vals)+1), [0])
        for cur_cn_change in all_cn_change_vals[np.argsort(np.abs(all_cn_change_vals))[::-1]]:
            cur_event = 'gain' if cur_cn_change > 0 else 'loss'

            cur_change_location = ((cur_child_cn.loc[~loh_pos[allele], allele] -
                        cur_parent_cn.loc[~loh_pos[allele], allele]) == cur_cn_change)

            # + cur_chroms enables detection of chromosome boundaries
            event_labels_ = ((np.cumsum(np.concatenate([[0], np.diff(
                (cur_change_location.values + cur_chroms[~loh_pos[allele]]))])
                * cur_change_location.values) + 1)
                * cur_change_location.values)

            # Label events starting at 1
            event_labels = np.zeros_like(event_labels_)
            for i, j in enumerate(np.setdiff1d(np.unique(event_labels_), [0])):
                event_labels[event_labels_ == j] = i + 1

            cur_events = (cur_child_cn
                          .loc[~loh_pos[allele]]
                          .reset_index()
                          .loc[np.array([np.argmax(event_labels == val) for val in np.setdiff1d(np.unique(event_labels), [0])])]
                          [['chrom', 'start', 'end', allele]].values)

            # adjust ends
            cur_events[:, 2] = (cur_child_cn
                                .loc[~loh_pos[allele]]
                                .reset_index()
                                .loc[np.array([len(event_labels) - np.argmax(event_labels[::-1] == val) - 1 for val in np.setdiff1d(np.unique(event_labels), [0])])]
                                ['end'].values)

            cur_ind = np.arange(len(events_df), len(events_df)+len(cur_events))
            events_df = pd.concat([events_df, pd.DataFrame(index=cur_ind)])
            events_df.loc[cur_ind, 'sample_id'] = child_name
            events_df.loc[cur_ind, 'allele'] = allele
            events_df.loc[cur_ind, 'type'] = cur_event
            events_df.loc[cur_ind, 'cn_child'] = cur_events[:, 3]
            events_df.loc[cur_ind, ['chrom', 'start', 'end']] = cur_events[:, :3]

            cur_child_cn.loc[np.intersect1d(loh_pos.loc[~loh_pos[allele]].index,
                                            cur_change_location.loc[cur_change_location].index), allele] += (1 if (cur_cn_change < 0) else -1)

    events_df['chrom'] = tools.format_chromosomes(events_df['chrom'])
    events_df = (events_df[['sample_id', 'allele', 'chrom', 'start', 'end', 'type', 'cn_child']]
                 .reset_index(drop=True)
                 .sort_values(['sample_id', 'allele', 'chrom', 'start', 'end', 'type', 'cn_child']))

    return cur_df, events_df


def overlap_events(events_df=None, output_df=None, tree=None, overlap_threshold=0.9,
                   chromosome_bed='default', regions_bed='default',
                   replace_loh_with_loss=True, alleles=['cn_a', 'cn_b'],
                   replace_both_arms_with_chrom=True, normal_name='diploid'):
    """Overlap copy-number events with regions of interest

    Args:
        events_df (pandas.DataFrame, optional): All copy-number events. Defaults to None.
        output_df (pandas.DataFrame, optional): DataFrame containing all copy-numbers. Defaults to None.
        tree (Bio.Phylo.Tree, optional): Phylogenetic tree. Defaults to None.
        overlap_threshold (float, optional): Threshold above which an overlap is considered. Defaults to 0.9.
        chromosome_bed (str, optional): Name of BED file containing chromosome arm information. Defaults to 'none'.
        regions_bed (str, optional): Name of BED file containing regions of interest. Defaults to 'none'.
        replace_loh_with_loss (bool, optional): If True, loh is considered like a normal loss. Defaults to True.
        alleles (list, optional): List of alleles. Defaults to ['cn_a', 'cn_b'].
        replace_both_arms_with_chrom (bool, optional): If True, an event in the p- and q-arm of a chromosome will be displayed as a single event. Defaults to True.
        normal_name (str, optional): Name of normal sample. Defaults to 'diploid'.

    Returns:
        pandas.DataFrame: DataFrame with events concerning the regions of interest
    """
    try:
        import pyranges as pr
    except ImportError:
        raise ImportError("You have to install pyranges to overlap events with regions of interest") from None       

    if chromosome_bed == 'none':
        chromosome_bed = None
    elif chromosome_bed == 'default':
        chromosome_bed = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      "objects", "hg38_chromosome_arms.bed")
    if regions_bed == 'none':
        regions_bed = None
    elif regions_bed == 'default':
        regions_bed = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "objects", "Davoli_2013_TSG_OG_genes.bed")

    all_events = pd.DataFrame(columns=['Chromosome', 'Start', 'End', 'name', 'NumberOverlaps',
                                       'FractionOverlaps', 'event', 'branch']).set_index(['Chromosome', 'Start', 'End'])

    if events_df is None:
        if output_df is None or tree is None:
            raise MEDICCEventReconstructionError("Either events_df or df and tree has to be specified")
        _, events_df = calculate_all_cn_events(tree, output_df, alleles=alleles, normal_name=normal_name)
    if replace_loh_with_loss:
        events_df.loc[events_df['type'] == 'loh', 'type'] = 'loss'

    # Read chromosome regions and other regions
    if chromosome_bed is None and regions_bed is None:
        raise MEDICCEventReconstructionError("Either chromosome_bed or regions_bed has to be specified")

    chr_arm_regions = None
    if chromosome_bed is not None:
        logger.debug(f'Overlap with chromosomes bed file {chromosome_bed}')
        chr_arm_regions = io.read_bed_file(chromosome_bed)
        whole_chromosome = chr_arm_regions.groupby('Chromosome').min()
        whole_chromosome['End'] = chr_arm_regions.groupby('Chromosome')['End'].max()
        whole_chromosome['name'] = whole_chromosome.index
        chr_arm_regions = pd.concat([chr_arm_regions, whole_chromosome.reset_index()]).sort_values('Chromosome')
        chr_arm_regions = pr.PyRanges(chr_arm_regions)

    regions = None
    if regions_bed is not None:
        logger.debug(f'Overlap with regions bed file {regions_bed}')
        regions = []
        if isinstance(regions_bed, list) or isinstance(regions_bed, tuple):
            for f in regions_bed:
                regions.append(pr.PyRanges(io.read_bed_file(f)))
        else:
            regions.append(pr.PyRanges(io.read_bed_file(regions_bed)))

    # Add WGD
    for ind, _ in events_df.loc[events_df['type'] == 'wgd'].iterrows():
        all_events.loc[('all', '0', '0')] = ['WGD', 1, 1., 'WGD', ind[0]]

    for cur_branch in events_df.index.get_level_values('sample_id').unique():
        for allele in alleles:
            cur_events_df = events_df.loc[cur_branch]
            cur_events_df = cur_events_df.loc[cur_events_df['allele']==allele]
            for event_type in ['gain', 'loss'] if replace_loh_with_loss else ['gain', 'loh', 'loss']:
                cur_events_ranges = pr.PyRanges(cur_events_df.loc[cur_events_df['type'] == event_type].reset_index(
                ).rename({'chrom': 'Chromosome', 'start': 'Start', 'end': 'End'}, axis=1))

                # Calculate chromosomal events
                if chr_arm_regions is not None:
                    chr_events = overlap_regions(
                        chr_arm_regions, cur_events_ranges, event_type, cur_branch, overlap_threshold)
                    # remove arms if the whole chromosome is in there
                    if replace_both_arms_with_chrom and len(chr_events) > 0:
                        chr_events = chr_events[~chr_events['name'].isin(np.concatenate(
                            [[name + 'p', name + 'q'] if ('q' not in name and 'p' not in name) else [] for name in chr_events['name']]))]
                    all_events = pd.concat([all_events, chr_events])

                # Calculate other events
                if regions is not None:
                    for region in regions:
                        chr_events = overlap_regions(
                            region, cur_events_ranges, event_type, cur_branch, overlap_threshold)
                        all_events = pd.concat([all_events, chr_events])

    all_events['final_name'] = all_events['name'].apply(lambda x: x.split(
        'chr')[-1]) + all_events['event'].apply(lambda x: ' +' if x == 'gain' else (' -' if x == 'loss' else (' 0' if x == 'loh' else '')))

    all_events = (all_events
                  .reset_index()
                  .sort_values(['branch', 'Chromosome', 'Start', 'End'], ascending=True)
                  .set_index(['branch'])
                  .drop('NumberOverlaps', axis=1)
                  .rename({'Start': 'Event-Start', 'End': 'Event-End'}, axis=1)
                  )

    return all_events


def overlap_regions(region, cur_events_ranges, event, branch, overlap_threshold):

    # Workaround for bug in PyRanges for numpy version 1.20 and higher
    np.long = np.int_

    cur_events_overlaps = region.coverage(cur_events_ranges).as_df()
    cur_events_overlaps = cur_events_overlaps.loc[cur_events_overlaps['FractionOverlaps']
                                                > overlap_threshold]
    cur_events_overlaps = cur_events_overlaps.set_index(['Chromosome', 'Start', 'End'])
    cur_events_overlaps['event'] = event
    cur_events_overlaps['branch'] = branch

    return cur_events_overlaps


class MEDICCEventReconstructionError(Exception):
    pass
