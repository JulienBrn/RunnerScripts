from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Annotated, Literal
import pandas as pd
import abc, re
import networkx as nx, graphviz

    

class PolyTaskInput(BaseModel):
    method: Literal["poly_task"] = "poly_task"
    file: Path

    def to_graph(self):
        task_path = self.file
        with Path(task_path).open("r") as f:
            i=0
            while(f):
                l = f.readline().split("\t")
                if len([x for x in l if "NEXT" in x]) >1:
                    break
                i+=1
        task_df = pd.read_csv(task_path, sep="\t", skiprows=i)
        task_df = task_df.rename(columns={task_df.columns[0]: "task_node" })
        df = task_df
        df = df.loc[~pd.isna(df["task_node"])]
        df = df.dropna(subset=df.columns[1:], how="all")
        df["task_node"] = df["task_node"].astype(int)
        graph = nx.DiGraph()
        for _, row in df.iterrows():
            row = row.dropna().to_dict()
            names = []
            graph.add_node(row["task_node"])
            node = graph.nodes[row["task_node"]]
            for col in row:
                if col.startswith("NEXT"):
                    pattern = r'\(.+\)$'
                    ns = re.findall(pattern, row[col])
                    if len(ns) == 0:
                        next_line = row["task_node"]+1
                        cond = row[col]
                    elif len(ns) ==1:
                        cond = row[col][:-len(ns[0])]
                        nlname = ns[0][1: -1]
                        if re.match(r'\d+', nlname):
                            next_line = int(nlname)
                        else:
                            next_line = df.loc[(df[["T1", "T2", "T3"]].apply(lambda s: s.str.lstrip("_")) == nlname).any(axis=1)]["task_node"]
                            if len(next_line) != 1:
                                raise Exception(f"problem {len(next_line)} {nlname}")
                            next_line = next_line.iat[0]
                    else:
                        raise Exception("Problem")
                    graph.add_edge(row["task_node"], next_line, cond=cond)
                elif re.match("T\d+", col):
                    m = re.match(r'(?P<time>\d*-?\d*)_(?P<name>\w+)$', str(row[col]))
                    if not m is None:
                        names.append(m["name"])
                        if m["time"]:
                            node[col] = m["time"]
                    else:
                        node[col] = row[col]
                else:
                    node[col] = row[col]
            node["poly_names"] = names
        return graph


class Computable(BaseModel):
    name: str
    data: Literal["duration", "count", "has"]
    aggregation: Literal["mean", "median", "sum"] = "mean"

class SplitPolyDatInput(BaseModel):
    method: Literal["split_poly_dat"] = "split_poly_dat"
    file: Path
    split: List[int]
    split_when: Literal["before", "after", "nosplit"] = "before"
    arrows: List[Computable]
    nodes: List[Computable]
    display_nsplits: bool = True

class NoSplitPolyDatInput(BaseModel):
    method: Literal["nosplit_poly_dat"] = "nosplit_poly_dat"
    file: Path
    arrows: List[Computable]
    nodes: List[Computable]


class Args(CLI):
    output_file: Path = Field(examples=["/media/filer2/T4b/Temporary/graph.svg"], pattern=".*svg")
    # inputs: List[PolyTaskInput | SplitPolyDatInput |  NoSplitPolyDatInput] = Field(examples=[
    #     [PolyTaskInput(file="/media/filer2/T4b/Temporary/file.xls"),
    #      SplitPolyDatInput(file="/media/filer2/T4b/Temporary/file.dat", split=[2], arrows=[Computable(name="data1", data="duration")], nodes=[Computable(name="data2", data="count")]),
    #      NoSplitPolyDatInput(file="/media/filer2/T4b/Temporary/file.dat", arrows=[Computable(name="total_count", data="count")], nodes=[Computable(name="total_count", data="count")]),
    #      ]])
    inputs: List[PolyTaskInput] = Field(examples=[
        [PolyTaskInput(file="/media/filer2/T4b/Temporary/file.xls")]])

params = Args()


for i in params.inputs:
    g = i.to_graph()

display_graph = g

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

nx.nx_pydot.write_dot(display_graph, "/tmp/graph.dot")
graphviz.render("dot", filepath="/tmp/graph.dot", outfile=params.output_file, format="svg")