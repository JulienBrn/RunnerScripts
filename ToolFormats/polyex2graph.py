from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Annotated, Literal, ClassVar
import pandas as pd
import abc, re
from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths

class Args(CLI):
    """
        to fill
    """
    input_path: Annotated[Path, Field(
        description="Path to the poly task file",
        default="/media/filer2/T4b/Temporary/....xls", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xls"]))
    )]
    output_path: Annotated[Path, Field(
        description="Path to the result graph image",
        default="/media/filer2/T4b/Temporary/....svg", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".svg"]))
    )]
    allow_output_overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(cpu=0.1, memory=0.1, gpu=0.0, disk=0.6)

args = Args()
from dafn.tool_converter import polyex2graph
import networkx as nx, graphviz
import tempfile

with check_output_paths(args.output_path, args.allow_output_overwrite) as output_path:
    graph = polyex2graph(args.input_path)
    display_graph = graph

    for node, attrs in display_graph.nodes(data=True):
        label = '\n'.join([f'{k}: {v}' for k,v in attrs.items()])
        for attr in list(attrs):
            del attrs[attr]
        display_graph.nodes[node]["label"] = label

    for n1, n2, attrs in display_graph.edges(data=True):
        label = '\n'.join([f'{k}: {v}' for k,v in attrs.items()])
        for attr in list(attrs):
            del attrs[attr]
        display_graph.edges[n1, n2]["label"] = label

    with tempfile.NamedTemporaryFile(delete=True) as temp:
        nx.nx_pydot.write_dot(display_graph, temp.name)
        graphviz.render("dot", filepath=temp.name, outfile=output_path, format="svg")