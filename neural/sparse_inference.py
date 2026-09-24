"""
Batched inference for variable-topology NEAT genomes (Phase 7, plan
step 7 - the open batching question).

The problem: Phase 1.5 batched inference packs a population into
stacked (N, in, out) tensors because every genome shares one topology.
NEAT populations do not - each genome has its own node and connection
sets - so there is nothing to stack.

The strategy: depth-layered padded batching.

  1. Every genome is placed in level order: nodes sorted by
     longest-path depth, so activations only flow forward one layer at
     a time.
  2. Across the batch, each depth layer is padded to the widest layer.
     Each genome's incoming connections become a dense
     (max_prev, max_cur) weight block with zero padding.
  3. One bmm per depth layer processes the whole population. Padding
     columns receive no signal and are never read back.

Compilation model: a population's topologies are fixed for a whole
evaluation episode, so the padded blocks are compiled once
(compile_population) and executed once per step (run_compiled) with
row indexing for the shrinking live set. Recompiling per step would
redo the entire Python-side block construction 150 times per episode
for nothing.

Activation contract matches the fixed-topology controller exactly:
hidden nodes tanh, outputs raw and then decoded as tanh (steering),
sigmoid (acceleration), sigmoid (eat). Keeping the engine contract
means apply_action and EAT_SIGNAL_THRESHOLD work unchanged.

Two batching strategies are provided so the plan's open question gets
a measured answer: population-wide (batched_forward) and per-species
(batched_forward_by_species, which pads only to a species' maximum).
forward_single is the unbatched reference both are verified against.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from evolution.neat.genome import HIDDEN, NeatGenome

# The engine's decoded output split (same as neural.batched_inference)
N_OUTPUTS = 3


def _level_order(genome: NeatGenome) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Level-order node ids and the (start, end) span of each depth layer.

    Level order means every enabled connection runs from an earlier
    depth to a strictly later one, so activations propagate one depth
    layer at a time. Ties within a layer are ordered by node id for
    determinism.
    """
    depths = genome.node_depths()
    by_depth: Dict[int, List[int]] = {}
    for node_id, depth in depths.items():
        by_depth.setdefault(depth, []).append(node_id)
    for node_ids in by_depth.values():
        node_ids.sort()

    order: List[int] = []
    spans: List[Tuple[int, int]] = []
    start = 0
    for depth in sorted(by_depth):
        order.extend(by_depth[depth])
        spans.append((start, len(order)))
        start = len(order)
    return order, spans


@dataclass
class CompiledPopulation:
    """
    Padded depth-layer tensors for a fixed population of genomes.

    Built once per evaluation episode by compile_population and
    executed per step by run_compiled.
    """
    orders: List[List[int]]
    spans: List[List[Tuple[int, int]]]
    weights: List[torch.Tensor]      # per layer: (N, max_prev, max_cur)
    hidden_masks: List[torch.Tensor] # per layer: (N, max_cur) bool
    output_rows: List[List[int]]     # per genome: level-order row of each output
    n_inputs: List[int]              # per genome: depth-0 node count
    max_nodes: int


def compile_population(genomes: Sequence[NeatGenome],
                       device: torch.device = torch.device("cpu")
                       ) -> CompiledPopulation:
    """
    Build the padded depth-layer blocks for a population once.

    Args:
        genomes: The fixed population.
        device: Torch device.

    Returns:
        The CompiledPopulation handle.
    """
    population_size = len(genomes)
    orders: List[List[int]] = []
    spans: List[List[Tuple[int, int]]] = []
    for genome in genomes:
        order, genome_spans = _level_order(genome)
        orders.append(order)
        spans.append(genome_spans)
    n_layers = max(len(genome_spans) for genome_spans in spans)
    max_nodes = max(len(order) for order in orders)

    positions = [{node_id: pos for pos, node_id in enumerate(order)}
                 for order in orders]
    depths = [genome.node_depths() for genome in genomes]

    weights: List[torch.Tensor] = []
    hidden_masks: List[torch.Tensor] = []
    for depth in range(1, n_layers):
        max_cur = max((spans[genome_index][depth][1]
                       - spans[genome_index][depth][0])
                      for genome_index in range(population_size)
                      if len(spans[genome_index]) > depth)
        max_prev = max(spans[genome_index][depth][0]
                       for genome_index in range(population_size)
                       if len(spans[genome_index]) > depth)
        block = torch.zeros(population_size, max_prev, max_cur,
                            dtype=torch.float32, device=device)
        mask = torch.zeros(population_size, max_cur, dtype=torch.bool,
                           device=device)
        for genome_index, genome in enumerate(genomes):
            genome_spans = spans[genome_index]
            if len(genome_spans) <= depth:
                continue
            col_offset = genome_spans[depth][0]
            genome_order = orders[genome_index]
            for node_id in genome_order[genome_spans[depth][0]:
                                          genome_spans[depth][1]]:
                if genome.nodes[node_id].node_type == HIDDEN:
                    mask[genome_index,
                         positions[genome_index][node_id] - col_offset] = True
            for gene in genome.enabled_connections():
                if depths[genome_index][gene.out_node] != depth:
                    continue
                row = positions[genome_index][gene.in_node]
                col = positions[genome_index][gene.out_node] - col_offset
                block[genome_index, row, col] = gene.weight
        weights.append(block)
        hidden_masks.append(mask)

    output_rows = []
    for genome_index, genome in enumerate(genomes):
        position = positions[genome_index]
        output_rows.append([position[node_id]
                            for node_id in genome.output_node_ids()[:N_OUTPUTS]])

    n_inputs = [genome.n_inputs for genome in genomes]
    return CompiledPopulation(orders=orders, spans=spans, weights=weights,
                              hidden_masks=hidden_masks,
                              output_rows=output_rows, n_inputs=n_inputs,
                              max_nodes=max_nodes)


def run_compiled(
    compiled: CompiledPopulation,
    observations: torch.Tensor,
    indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Run one batched forward pass over a compiled population.

    Args:
        compiled: Handle from compile_population.
        observations: (rows, n_inputs) observation rows. When indices
            is given, observations are the rows for those population
            members (e.g. the live agents).
        indices: Population rows being evaluated; None means all.

    Returns:
        (len(rows), 3) decoded actions: steering tanh in [-1, 1],
        acceleration and eat_signal sigmoid in [0, 1].
    """
    device = observations.device
    if indices is None:
        indices = list(range(len(compiled.orders)))
    else:
        indices = list(indices)
    row_index = torch.as_tensor(indices, dtype=torch.long)
    row_count = len(indices)
    n_layers = len(compiled.weights) + 1

    activations = torch.zeros(row_count, compiled.max_nodes,
                              dtype=torch.float32, device=device)
    for row, genome_index in enumerate(indices):
        # Only the real input columns take observations; any depth-0
        # source-less node (crossover can orphan a hidden node) stays
        # zero, matching the per-agent reference's tanh(0) = 0
        n_inputs = compiled.n_inputs[genome_index]
        activations[row, :n_inputs] = observations[row, :n_inputs]

    for depth in range(1, n_layers):
        block = compiled.weights[depth - 1].index_select(0, row_index)
        hidden_mask = compiled.hidden_masks[depth - 1].index_select(0, row_index)
        n_prev, n_cur = block.shape[1], block.shape[2]
        previous = activations[:, :n_prev].unsqueeze(1)
        current = torch.bmm(previous, block).squeeze(1)
        current = torch.where(hidden_mask, torch.tanh(current), current)
        for row, genome_index in enumerate(indices):
            genome_spans = compiled.spans[genome_index]
            if len(genome_spans) <= depth:
                continue
            start, end = genome_spans[depth]
            activations[row, start:end] = current[row, :end - start]

    output_values = torch.zeros(row_count, N_OUTPUTS,
                                dtype=torch.float32, device=device)
    for row, genome_index in enumerate(indices):
        for offset, position in enumerate(compiled.output_rows[genome_index]):
            output_values[row, offset] = activations[row, position]

    steering = torch.tanh(output_values[:, 0:1])
    acceleration = torch.sigmoid(output_values[:, 1:2])
    eat_signal = torch.sigmoid(output_values[:, 2:3])
    return torch.cat([steering, acceleration, eat_signal], dim=1)


def batched_forward(
    observations: torch.Tensor,
    genomes: Sequence[NeatGenome],
) -> torch.Tensor:
    """
    One batched forward pass across a population of variable-topology
    genomes via depth-layered padded batching (compile + run).

    Args:
        observations: (N, n_inputs) observation rows, one per genome.
        genomes: The population, row-aligned with observations.

    Returns:
        (N, 3) decoded actions in the engine's contract.
    """
    return run_compiled(compile_population(genomes), observations)


def batched_forward_by_species(
    observations: torch.Tensor,
    genomes: Sequence[NeatGenome],
    species_ids: Sequence[int],
) -> torch.Tensor:
    """
    Per-species batched forward: the depth-layered batch runs once per
    species, so padding is only to the species maximum rather than the
    population's. The 'per-species first' strategy of plan step 7;
    batched_forward is the population-wide alternative. The report
    benchmarks both.
    """
    outputs = torch.zeros(len(genomes), N_OUTPUTS,
                          dtype=torch.float32, device=observations.device)
    groups: Dict[int, List[int]] = {}
    for index, species_id in enumerate(species_ids):
        groups.setdefault(species_id, []).append(index)

    for indices in groups.values():
        compiled = compile_population([genomes[i] for i in indices])
        group_out = run_compiled(compiled, observations[indices])
        for row, index in enumerate(indices):
            outputs[index] = group_out[row]
    return outputs


def forward_single(observation: torch.Tensor,
                   genome: NeatGenome) -> torch.Tensor:
    """
    Reference per-agent forward: topological evaluation, no padding, no
    batching. Both batched strategies are verified against this and
    benchmarked against it.

    Returns:
        (1, 3) decoded action row.
    """
    depths = genome.node_depths()
    input_ids = sorted(node_id for node_id, gene in genome.nodes.items()
                       if gene.node_type == 'input')
    activations: Dict[int, float] = {
        node_id: float(observation[0, index])
        for index, node_id in enumerate(input_ids)
    }

    incoming: Dict[int, List[Tuple[int, float]]] = {}
    for gene in genome.enabled_connections():
        incoming.setdefault(gene.out_node, []).append(
            (gene.in_node, gene.weight))

    for node_id, depth in sorted(depths.items(), key=lambda item: item[1]):
        if node_id in activations:
            continue
        total = 0.0
        for source, weight in incoming.get(node_id, []):
            total += weight * activations[source]
        if genome.nodes[node_id].node_type == HIDDEN:
            total = float(torch.tanh(torch.tensor(total)))
        # Output nodes stay raw; the engine's decode applies the split
        activations[node_id] = total

    raw = torch.zeros(1, N_OUTPUTS, dtype=torch.float32)
    for offset, node_id in enumerate(genome.output_node_ids()[:N_OUTPUTS]):
        raw[0, offset] = activations[node_id]

    return torch.cat([
        torch.tanh(raw[:, 0:1]),
        torch.sigmoid(raw[:, 1:2]),
        torch.sigmoid(raw[:, 2:3]),
    ], dim=1)
