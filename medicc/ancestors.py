import logging
import os
import tempfile

import Bio
import Bio.Phylo
import fstlib
import numpy as np

import medicc

logger = logging.getLogger(__name__)


class _CandidateStore:
    def __init__(self, spill):
        self.spill = spill
        self.values = {}
        spill_parent = (
            os.environ.get("MEDICC2_ANCESTOR_SPILL_DIR") or
            os.environ.get("SLURM_TMPDIR"))
        if spill_parent:
            os.makedirs(spill_parent, exist_ok=True)
        self.temp_dir = (
            tempfile.TemporaryDirectory(
                prefix="medicc2-ancestors-", dir=spill_parent)
            if spill else None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

    def put(self, name, candidate):
        if not self.spill:
            self.values[name] = candidate
            return
        path = os.path.join(
            self.temp_dir.name, f"candidate-{len(self.values):06d}.fst")
        candidate.write(path)
        if not os.path.isfile(path):
            raise MEDICCAncestorReconstructionError(
                f"Could not spill ancestral candidate {name!r} to disk")
        self.values[name] = path

    def get(self, name):
        value = self.values[name]
        return fstlib.read(value) if self.spill else value

    def __contains__(self, name):
        return name in self.values


def _spill_ancestral_candidates(sample_count):
    mode = os.environ.get("MEDICC2_ANCESTOR_SPILL_MODE", "auto").lower()
    if mode not in {"auto", "always", "never"}:
        logger.warning(
            "Unknown MEDICC2_ANCESTOR_SPILL_MODE=%r; using auto.", mode)
        mode = "auto"
    try:
        threshold = int(
            os.environ.get("MEDICC2_ANCESTOR_SPILL_THRESHOLD", "64"))
    except ValueError:
        logger.warning(
            "Invalid MEDICC2_ANCESTOR_SPILL_THRESHOLD; using 64.")
        threshold = 64
    return mode == "always" or (mode == "auto" and sample_count >= threshold)


def reconstruct_ancestors(tree, samples_dict, fst, normal_name, prune_weight=0):

    if len(samples_dict) == 2:
        return samples_dict

    fsa_dict = samples_dict.copy()
    tree = Bio.Phylo.BaseTree.copy.deepcopy(tree)

    clade_list = [clade for clade in tree.find_clades(order="preorder") if clade.name != normal_name]
    spill = _spill_ancestral_candidates(len(samples_dict))
    if spill:
        logger.info(
            "Spilling intermediate ancestral FSAs to disk to bound memory.")

    with _CandidateStore(spill) as candidates:
        logger.info("Ancestor reconstruction: Up the tree")
        # up the tree (leaf to root)
        for node in reversed(clade_list):
            if len(node.clades) != 0:
                children = [
                    item for item in node.clades if item.name != normal_name]
                left_name = children[0].name
                right_name = children[1].name
                logger.debug(
                    f"Clade: {node.name}, left: {left_name}, right: {right_name}")

                left = (
                    candidates.get(left_name)
                    if left_name in candidates else fsa_dict[left_name])
                right = (
                    candidates.get(right_name)
                    if right_name in candidates else fsa_dict[right_name])
                intersection = intersect_clades_detmin(
                    left, right, fst, prune_weight=prune_weight,
                    detmin_before_intersect=False,
                    detmin_after_intersect=True)
                candidates.put(node.name, intersection)

        logger.debug("Ancestor reconstruction for root")
        # root node is calculated separately w.r.t. normal node
        root_name = clade_list[0].name
        sp = fstlib.align(
            fst, fsa_dict[normal_name], candidates.get(root_name))
        fsa_dict[root_name] = fstlib.arcmap(
            sp.project('output'), map_type='rmweight')

        logger.info("Ancestor reconstruction: Down the tree")
        # down the tree (root to leaf)
        for node in clade_list:
            if len(node.clades) != 0:
                children = [q for q in node.clades if len(q.clades) != 0]
                logger.debug(
                    f"Clade: {node.name}, internal children: {children}")
                for child in children:
                    sp = fstlib.align(
                        fst, fsa_dict[node.name], candidates.get(child.name))
                    fsa_dict[child.name] = fstlib.arcmap(
                        sp.project('output'), map_type='rmweight')

    # check if ancestors were correctly reconstructed
    sample_lengths = {sample: len(medicc.tools.fsa_to_string(fsa_dict[sample])) for sample, fsa in fsa_dict.items()}
    normal_length = sample_lengths[normal_name]

    if np.any([x != normal_length for x in sample_lengths.values()]):
        raise MEDICCAncestorReconstructionError("Some ancestors could not be reconstructed. These are:\n"
                                                "{}".format('\n'.join([sample for sample, length in sample_lengths.items() if length != normal_length])) + \
                                                "\nCheck whether your normal sample contains segments with copy number zero")

    return fsa_dict


def intersect_clades_detmin(left, right, fst, prune_weight=None, detmin_before_intersect=True, detmin_after_intersect=True):
    L = fstlib.compose(fst, left.arcsort('ilabel')).project('input')
    R = fstlib.compose(fst, right.arcsort('ilabel')).project('input')
    if detmin_before_intersect:
        L = fstlib.determinize(L).minimize()
        R = fstlib.determinize(R).minimize()
    intersection = fstlib.intersect(L.arcsort('olabel'), R)
    # For prune_weight=0, deletes all paths but the shortest one
    if prune_weight is not None:
        pruned = fstlib.prune(intersection, weight=prune_weight)
    else:
        pruned = intersection
    if detmin_after_intersect:
        pruned = fstlib.determinize(pruned).minimize()
    return pruned


class MEDICCAncestorReconstructionError(Exception):
    pass
