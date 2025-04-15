from typing import Dict, List, Tuple, Union, Optional
from pydantic import BaseModel, ConfigDict
import networkx as nx
import matplotlib.pyplot as plt
from glotaran.model.model import Model
from glotaran.parameter.parameters import Parameters
from glotaran.model.item import fill_item
from collections import deque
from enum import Enum


class TransitionData(Enum):
    VALUE = "value"
    LABEL = "label"


class VisualizationOptions(BaseModel):
    alternative_node_names: Dict[str, str] = {}
    colour_node_mapping: Dict[str, List[str]] = {}
    omitted_rate_constants: List[str] = []
    plot_size: Tuple[int, int] = (10, 10)
    plot_graph_node_size: int = 4000
    transition_data: TransitionData = TransitionData.VALUE

    model_config = ConfigDict(extra="allow")


def round_and_convert(value_in_ps_inverse):
    value_in_ns_inverse = value_in_ps_inverse * 1e3
    return (
        round(value_in_ns_inverse) if value_in_ns_inverse >= 1 else round(value_in_ns_inverse, 2)
    )


def build_all_transitions(megacomplex_k_matrices, omitted_rate_constants, transition_data):
    transitions = []
    idx = 1
    total_decay_rates = set()

    for system in megacomplex_k_matrices.values():
        for (state_from, state_to), param in system.matrix.items():
            if param.label not in omitted_rate_constants:
                rate_constant_value = round_and_convert(param.value)
                extra_edge_attribute = (
                    {"weight": rate_constant_value}
                    if transition_data == TransitionData.VALUE
                    else {"label": param.label}
                )
                if state_from != state_to:
                    transitions.append((state_to, state_from, extra_edge_attribute))
                elif (state_to, rate_constant_value) not in total_decay_rates:
                    transitions.append((state_to, f"GS{idx}", extra_edge_attribute))
                    total_decay_rates.add((state_to, rate_constant_value))
                    idx += 1
    return transitions


def get_filled_megacomplex_k_matrices(
    megacomplexes: List[str], model: Model, parameters: Parameters
):
    k_matrices = {}
    for mc in megacomplexes:
        if mc not in model.megacomplex:
            raise ValueError(f"Megacomplex {mc} not found.")
        if model.megacomplex[mc].type != "decay":
            continue
        filled_megacomplex = fill_item(model.megacomplex[mc], model, parameters)
        k_matrices[mc] = filled_megacomplex.get_k_matrix()
    return k_matrices


def visualize_megacomplex(
    megacomplex: Union[str, List[str]],
    model: Model,
    parameter: Parameters,
    visualization_options: VisualizationOptions = VisualizationOptions(),
):
    if isinstance(megacomplex, str):
        megacomplexes = [megacomplex]
    else:
        megacomplexes = megacomplex

    k_matrices = get_filled_megacomplex_k_matrices(megacomplexes, model, parameter)

    transitions = build_all_transitions(
        k_matrices,
        visualization_options.omitted_rate_constants,
        visualization_options.transition_data,
    )

    visualize(transitions, visualization_options)


def visualize_dataset_model(
    dataset_model: str,
    model: Model,
    parameter: Parameters,
    exclude_megacomplexes: Optional[List[str]] = None,
    visualization_options: VisualizationOptions = VisualizationOptions(),
):
    if dataset_model not in model.dataset:
        raise ValueError(f"Dataset model {dataset_model} not found in the model.")

    associated_megacomplexes = model.dataset[dataset_model].megacomplex
    if exclude_megacomplexes:
        megacomplexes = [mc for mc in associated_megacomplexes if mc not in exclude_megacomplexes]
    else:
        megacomplexes = associated_megacomplexes

    k_matrices = get_filled_megacomplex_k_matrices(megacomplexes, model, parameter)

    transitions = build_all_transitions(
        k_matrices,
        visualization_options.omitted_rate_constants,
        visualization_options.transition_data,
    )

    visualize(transitions, visualization_options)


def is_directed_acyclic(graph):
    return nx.is_directed_acyclic_graph(graph)


def layout_directed_acyclic_graph(graph, visualization_options):
    topological_order = list(nx.topological_sort(graph))

    x_pos = 0
    y_pos = 0
    layer_width = {}
    node_positions = {}

    # Start positioning from the first node in topological order
    root_node = topological_order[0]
    update_position_in_directed_acyclic_graph(
        graph, root_node, x_pos, y_pos, node_positions, layer_width
    )

    # Adjust the width of nodes with multiple predecessors
    for node in topological_order:
        predecessors = list(graph.predecessors(node))
        if len(predecessors) > 1:
            pred_y_levels = [node_positions[p][1] for p in predecessors]
            pred_x_levels = [node_positions[p][0] for p in predecessors]
            if len(set(pred_y_levels)) == 1:  # All predecessors at the same horizontal level
                max_predecessor_x = max(pred_x_levels)
                new_pos = (max_predecessor_x, node_positions[predecessors[0]][1] - 1)
            elif len(set(pred_x_levels)) == 1:  # All predecessors at the same vertical level
                min_predecessor_y = min(pred_y_levels)
                new_pos = (node_positions[predecessors[0]][0] + 1, min_predecessor_y)
            else:
                max_predecessor_x = max(pred_x_levels)
                new_pos = (max_predecessor_x + 1, node_positions[node][1])
            shift_x = new_pos[0] - node_positions[node][0]
            shift_y = new_pos[1] - node_positions[node][1]
            # Shift the node and all its successors
            nodes_to_shift = [node]
            while nodes_to_shift:
                current_node = nodes_to_shift.pop()
                current_x, current_y = node_positions[current_node]
                node_positions[current_node] = (current_x + shift_x, current_y + shift_y)
                nodes_to_shift.extend(graph.successors(current_node))
    visualization_options.plot_graph_edge_connection_style = "arc3"
    return graph, node_positions, visualization_options


# Function to update position recursively
def update_position_in_directed_acyclic_graph(graph, node, x, y, pos, layer_width):
    pos[node] = (x, y)
    successors = list(graph.successors(node))
    num_successors = len(successors)
    if num_successors == 1:
        update_position_in_directed_acyclic_graph(graph, successors[0], x, y - 1, pos, layer_width)
    elif num_successors > 1:
        for i, successor in enumerate(successors):
            if i == 0:
                update_position_in_directed_acyclic_graph(
                    graph, successor, x + 1, y, pos, layer_width
                )
            else:
                update_position_in_directed_acyclic_graph(
                    graph, successor, x, y - 1, pos, layer_width
                )
    layer_width[x] = max(layer_width.get(x, 0), y)


def layout_directed_cyclic_graph(graph, visualization_options):
    node_positions = {}
    degree_dict = dict(graph.degree())
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)

    # Center position for the starting node
    center = (0, 0)
    node_positions[sorted_nodes[0]] = center

    directions = [(1, 0), (0, 1), (0, -1), (-1, 0)]

    used_positions = set()
    used_positions.add(center)

    queue = deque([sorted_nodes[0]])

    corner_position = (10, 10)

    while queue:
        current_node = queue.popleft()
        current_pos = node_positions[current_node]
        neighbors = list(graph.neighbors(current_node)) + list(graph.predecessors(current_node))
        neighbor_pos_index = 0

        for neighbor in neighbors:
            if neighbor not in node_positions:
                attempts = 0
                while True:
                    direction = directions[neighbor_pos_index % 4]
                    new_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                    neighbor_pos_index += 1
                    attempts += 1
                    if new_pos not in used_positions:
                        node_positions[neighbor] = new_pos
                        used_positions.add(new_pos)
                        queue.append(neighbor)
                        break
                    if attempts >= 4:  # If can't place in four attempts, place as isolated
                        node_positions[neighbor] = corner_position
                        used_positions.add(corner_position)
                        corner_position = (corner_position[0] + 1, corner_position[1] + 1)
                        break
    visualization_options.plot_graph_edge_connection_style = "arc3,rad=0.1"
    visualization_options.plot_graph_node_size = 5000
    return graph, node_positions, visualization_options


def apply_some_adjustments(graph):
    for node in graph:
        ground_state_neighbors = [neighbor for neighbor in graph[node] if "GS" in neighbor]

        if len(ground_state_neighbors) > 1:
            total_rate_constant = 0
            first_neighbor = ground_state_neighbors[0]

            for neighbor in ground_state_neighbors[1:]:
                total_rate_constant += graph[node][neighbor]["weight"]
                graph.remove_edge(node, neighbor)

            graph[node][first_neighbor]["weight"] += total_rate_constant

    return graph


def style_and_draw(graph, node_positions, visualization_options):
    plt.figure(figsize=visualization_options.plot_size)

    non_ground_nodes = [node for node in graph.nodes() if "GS" not in node]

    node_labels = {n: "" if "GS" in n else n for n in graph}
    for (
        original_node_name,
        alternate_node_name,
    ) in visualization_options.alternative_node_names.items():
        node_labels[original_node_name] = alternate_node_name

    node_colour_dict = {
        node: colour
        for colour, nodes in visualization_options.colour_node_mapping.items()
        for node in nodes
    }
    colour_order = [
        node_colour_dict[node] if node in node_colour_dict else "skyblue"
        for node in non_ground_nodes
    ]

    nx.draw_networkx_nodes(
        graph,
        node_positions,
        nodelist=non_ground_nodes,
        node_size=visualization_options.plot_graph_node_size,
        node_color=colour_order,
        edgecolors="black",
    )
    nx.draw_networkx_labels(
        graph, node_positions, labels=node_labels, font_size=10, font_weight="bold"
    )
    nx.draw_networkx_edges(
        graph,
        node_positions,
        arrows=True,
        node_size=visualization_options.plot_graph_node_size,
        connectionstyle=visualization_options.plot_graph_edge_connection_style,
    )
    edge_labels = nx.get_edge_attributes(graph, "weight")

    if nx.is_directed_acyclic_graph(graph):
        nx.draw_networkx_edge_labels(
            graph,
            node_positions,
            edge_labels=edge_labels,
            font_size="6",
            rotate=False,
            font_color="red",
        )
    else:
        for (n1, n2), label in edge_labels.items():
            x1, y1 = node_positions[n1]
            x2, y2 = node_positions[n2]
            label_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
            dx, dy = x2 - x1, y2 - y1
            offset_x = 0.1 * dy if graph.has_edge(n2, n1) else 0
            offset_y = -0.1 * dx if graph.has_edge(n2, n1) else 0
            plt.text(
                label_pos[0] + offset_x,
                label_pos[1] + offset_y,
                s=label,
                bbox=dict(
                    facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.5
                ),
            )
    plt.show()


def visualize(transitions, visualization_options):
    graph = nx.DiGraph()
    graph.add_edges_from(transitions)
    if nx.is_directed_acyclic_graph(graph):
        graph, node_positions, visualization_options = layout_directed_acyclic_graph(
            graph, visualization_options
        )
    else:
        graph, node_positions, visualization_options = layout_directed_cyclic_graph(
            graph, visualization_options
        )
    graph = apply_some_adjustments(graph)
    style_and_draw(graph, node_positions, visualization_options)
