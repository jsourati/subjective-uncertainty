from __future__ import annotations

import os
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from spacy import Language
from spacy.tokens import Doc


def phrase_to_deptree(nlp: Language, document: str) -> tuple[Doc, nx.DiGraph]:
    """
    given nlp and a phrase (string) - yield spacy doc and a digraph representing syn parsing
    :param nlp:
    :param document:
    :return:
    """
    graph = nx.DiGraph()

    rdoc = nlp(document)

    keys_to_pick = [
        "i",
        "dep_",
        "tag_",
        "lower_",
        "lemma_",
        "text",
    ]

    map_keys = {"i": "s", "lower_": "lower", "lemma_": "lemma"}

    vs = [
        (
            token.i,
            {
                map_keys[k] if k in map_keys else k: token.__getattribute__(k)
                for k in keys_to_pick
            },
        )
        for token in rdoc
    ]

    # add label
    for i, v in vs:
        v["label"] = f"{v['s']}-{v['lower']}-{v['dep_']}-{v['tag_']}"

    es = []
    for token in rdoc:
        for child in token.children:
            es.append((token.i, child.i))

    graph.add_nodes_from(vs)
    graph.add_edges_from(es)

    return rdoc, graph


def plot_graph(graph, path, name, plot_png=False, plot_pdf=True, prog="dot"):
    """

    :param graph:
    :param path:
    :param name:
    :param plot_png:
    :param plot_pdf:
    :param prog: prog=[‘neato’|’dot’|’twopi’|’circo’|’fdp’|’nop’]
    :return:
    """

    dot = to_agraph(graph)
    dot.layout("dot")
    if plot_png:
        dot.draw(path=os.path.join(path, f"{name}.png"), format="png", prog=prog)
    if plot_pdf:
        dot.draw(path=os.path.join(path, f"{name}.pdf"), format="pdf", prog=prog)
