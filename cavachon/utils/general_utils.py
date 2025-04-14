import copy
import itertools
import re
from typing import Any, Dict, List, Set

import networkx as nx

from cavachon.environment.constants import Constants


class GeneralUtils:
    @staticmethod
    def order_components(
        component_configs: Dict[str, Any],
    ) -> List[Any]:
        """Reorder the components based on the dependency. The
        components are ordered based on the number of predecessors
        components.

        Parameters
        ----------
        component_configs: Dict[str, Any]
            component configs (which can be loaded with
            Config.components), where the keys are the component name,
            values are the component configs.

        Raises
        ------
        AttributeError
            if the dependencies between components is not a directed
            acyclic graph.

        Returns
        -------
        List[Any]:
            reordered components.

        """
        # TODO: fix the return type hint when refactoring configs
        component_id_mapping: dict[str, int] = dict()
        id_component_mapping: dict[int, str] = dict()
        for i, (component_name, component_config) in enumerate(
            component_configs.items()
        ):
            component_id_mapping.setdefault(component_name, i)
            id_component_mapping.setdefault(i, component_config)

        n_components = len(component_configs)
        component_ids = list(range(n_components))

        G = nx.DiGraph()
        for i in component_ids:
            G.add_node(i)
        for component_id, component_config in id_component_mapping.items():
            conditioned_on_z = component_config.get(
                Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z, []
            )
            conditioned_on_z_hat = component_config.get(
                Constants.CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT, []
            )
            if len(conditioned_on_z) + len(conditioned_on_z_hat) != 0:
                for conditioned_on_name in itertools.chain(
                    conditioned_on_z, conditioned_on_z_hat
                ):
                    conditioned_on_component_id = component_id_mapping.get(
                        conditioned_on_name
                    )
                    G.add_edge(component_id, conditioned_on_component_id)

        if not nx.is_directed_acyclic_graph(G):
            message = "".join(
                (
                    "The conditioning relationships between components form a directed cyclic ",
                    "graph. Please check the conditioning relationships between components in ",
                    "the config file.\n",
                )
            )
            raise AttributeError(message)

        component_id_ordered_list: List[int] = list()
        added_component_id_set: Set[int] = set()
        while len(component_id_ordered_list) < n_components:
            for component_id in G.nodes:
                node_successors_not_added: Set[int] = set()
                for bfs_successors in nx.bfs_successors(G, component_id):
                    node, successors = bfs_successors
                    node_successors_not_added = node_successors_not_added.union(
                        set(successors)
                    )
                node_successors_not_added = node_successors_not_added.difference(
                    set(component_id_ordered_list)
                )
                if (
                    len(node_successors_not_added) == 0
                    and component_id not in added_component_id_set
                ):
                    component_id_ordered_list.append(component_id)
                    added_component_id_set.add(component_id)

        components = [
            id_component_mapping[component_id]
            for component_id in component_id_ordered_list
        ]
        return components

    @staticmethod
    def convert_to_tensorflow_compatible_string(string: str, lower: bool = True) -> str:
        """Generate string which is compatible with Tensorflow object
        names.

        Parameters
        ----------
        string: str
            string to be processed.

        lower: bool, optional
            whether or not to lowercase the string.

        Returns
        -------
        str:
            string which is compatible with Tensorflow object names.
        """
        valid_full_regex = re.compile(Constants.TENSORFLOW_NAME_REGEX)
        valid_start_regex = re.compile(Constants.TENSORFLOW_NAME_START_REGEX)
        converted_string = ""
        if not valid_full_regex.match(string):
            if valid_start_regex.match(string):
                converted_string = re.sub(r"[^A-Za-z0-9_.\\/>-]", "_", string)
            else:
                converted_string = "t_" + re.sub(r"[^A-Za-z0-9_.\\/>-]", "_", string)
        else:
            converted_string = string

        if lower:
            return converted_string.lower()
        else:
            return converted_string

    @staticmethod
    def duplicate_obj_to_list(obj: Any, n_objs: int) -> List[Any]:
        """Duplicate any object into a list of the same duplicated
        objects.

        Parameters
        ----------
        obj: Any
            object to be duplicated.

        n_objs: int
            number of duplication.

        Returns
        -------
        List[Any]:
            list of duplicated objects.

        """
        res = list()
        for i in range(0, n_objs):
            res.append(copy.deepcopy(obj))
        return res
