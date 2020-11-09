import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np


def add_layer(g, current, type, layer, start_index=0, normalized=False):
    edges = list()
    
    for prev in g.nodes:       
        w = layer.linear.weight[:, g.nodes[prev]['start_index']:g.nodes[prev]['start_index'] + g.nodes[prev]['n_nodes']]
        w = float(w.abs().sum())
        
        
        if normalized:
            w_sum_in = float(layer.linear.weight[:, :start_index].abs().sum())
            w /= w_sum_in
        
        i = list(g.nodes).index(prev)
        j = len(g.nodes)
        up = (j-i) % 2
        
        edges.append((prev, current, dict(weight=w, up=up)))

    g.add_node(current, type=type, n_nodes=layer.linear.out_features, start_index=start_index)
    
    g.add_edges_from(edges)
    return current
    

def graph_from_model(model, normalized=False):
    with model.discrete():
        g = nx.DiGraph()

        g.add_node('input', type='in', n_nodes=model.in_features, start_index=0)

        start_index = model.in_features

        for i, layer in enumerate(model.hidden_layers):
            node = add_layer(g, f'hidden {i+1}', type='hidden', layer=layer, start_index=start_index, normalized=normalized)

            start_index += g.nodes[node]['n_nodes']

        add_layer(g, 'output', type='out', layer=model.output_layer, start_index=start_index, normalized=normalized)
    return g

N = 100
start=0.05

colors = np.zeros((N, 4))
colors[:, -1] = np.linspace(start,1,colors.shape[0])
cm_transparent_to_black = matplotlib.colors.ListedColormap(colors)

colors = np.empty((N, 4))
colors[:, 0] = colors[:, 1] = colors[:, 2] = np.linspace(1-start,0,colors.shape[0])
colors[:, -1] = 1
cm_white_to_black = matplotlib.colors.ListedColormap(colors)


def draw_model_graph(g, normalized=False):
    type_colors = {
        'in': '#fdb462',
        'hidden': '#80b1d3',
        'out': '#b3de69'
    }

    positions = np.zeros((len(g), 2))
    positions[:, 0] = np.arange(len(g))
    positions[:, 1] = 0

    positions = nx.drawing.layout.rescale_layout(positions)

    
    label_positions = {
        n: positions[i, :] + np.array([0,-.2]) for i, n in enumerate(g.nodes)
    }
    
    positions = {
        n: positions[i, :] for i, n in enumerate(g.nodes)
    }
    
    

    params = dict(
        pos=positions,

        node_size=[
            g.nodes[n]['n_nodes'] for n in g.nodes
        ],

        node_color=[type_colors[g.nodes[n]['type']] for n in g.nodes()],
        cmap=plt.get_cmap('tab20')
    )

    label_offset = np.zeros((len(g), 2))
    label_offset[:, 1] = .1
    
    nx.draw_networkx_nodes(g, **params)
    nx.draw_networkx_labels(
        g, pos=label_positions, font_size=10, labels={
            n: str(g.nodes[n]['n_nodes']) if g.nodes[n]['type']=='hidden' else g.nodes[n]['type'] for n in g.nodes
    })

    #nx.draw_networkx_labels(g, positions, font_size=8)
    
    edge_list_up = list()
    edge_list_down = list()
    for e in g.edges:
        if g.edges[e]['up']:
            edge_list_up.append(e)
        else:
            edge_list_down.append(e)    
    
    for up in [True, False]:
        edgelist = (edge_list_up if up else edge_list_down)
        edge_weights = np.array([g.edges[e]['weight'] for e in edgelist])

        nx.draw_networkx_edges(
            g, pos=positions, edgelist=edgelist,
            connectionstyle=f"arc3, rad={('-' if up else '')}0.5",
            edge_color=edge_weights,
            edge_cmap=cm_transparent_to_black,
            edge_vmin=0,
            edge_vmax=edge_weights.max(),
            width=2.0,
        )
    
    plt.gca().set_ylim([-1,1])
    
    
    mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, (1.0 if normalized else edge_weights.max())), cmap=cm_white_to_black)
    cbar = plt.colorbar(mappable)
    cbar.outline.set_visible(False)
    
    if normalized:
        cbar.ax.set_ylabel('Percentage of incoming edges')
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        xticks = mtick.FormatStrFormatter(fmt)
        cbar.ax.xaxis.set_major_formatter(xticks)
    else:
        cbar.ax.set_ylabel('Number of edges')
        
    cbar.ax.yaxis.set_label_coords(4.5,0.5)
    
    plt.gca().set_axis_off()

    