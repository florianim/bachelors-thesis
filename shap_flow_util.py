from shapflow.flow import GraphExplainer, node_dict2str_dict
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns

def calculate_edge_credit(causal_graph, bg_i, fg, nruns, silent=True):
    cf_c = GraphExplainer(causal_graph, bg_i, nruns=nruns, silent=silent).shap_values(fg)
    return node_dict2str_dict(cf_c.edge_credit)

def calculate_edge_credit_alt(causal_graph, bg_i, fg_file, nruns, silent=True):
    fg = read_csv_incl_timeindex(fg_file)
    cf_c = GraphExplainer(causal_graph, bg_i, nruns=nruns, silent=silent).shap_values(fg)
    return node_dict2str_dict(cf_c.edge_credit)

def read_csv_incl_timeindex(filepath):
    # expect column 'timestamp' to exist and contain valid timestamps
    df = pd.read_csv(filepath)
    df.index = pd.to_datetime(df['timestamp'])
    df.drop('timestamp', axis=1, inplace=True)
    return df

def read_csv_between(filepath, start_date, end_date):
    # includes start_date, includes end_date
    df = read_csv_incl_timeindex(filepath)
    return df[start_date:end_date]

def create_pytorch_lightning_f(X_scaler, y_scaler, model):
    # shapley flow is given the original data (untransformed)
    # data is transformed by this method when predicting
    # model prediction is transformed back to original scaling
    # -> shapley values represent impact on untransformed data and predictions
    def f_(*args):
        data_array = np.column_stack(args)
        data_array_transformed = X_scaler.transform(data_array)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor = torch.tensor(data_array_transformed, dtype=torch.float32).to(device)

        model.eval()
        model.to(device)

        with torch.no_grad():
            predictions = model(tensor)
        predictions_np = predictions.cpu().numpy()
        predictions_np_orig = y_scaler.inverse_transform(predictions_np)
        return predictions_np_orig
    return f_

def get_old_feature_name(new_name):
    new_to_old_name = {
        'Month': 'month',
        'Day of week': 'day_of_week',
        'Day of year cos': 'dayofyear_cos',
        'Day of year sin': 'dayofyear_sin',
        'Hour cos': 'hour_cos',
        'Hour sin': 'hour_sin',
        'Generation day-ahead': 'generation_da',
        'Load day-ahead': 'load_da',
        'Solar day-ahead': 'solar_da',
        'Wind day-ahead': 'wind_da',
        'Generation day-ahead ramp': 'generation_da_ramp',
        'Load day-ahead ramp': 'load_da_ramp',
        'Solar day-ahead ramp': 'solar_da_ramp',
        'Wind day-ahead ramp': 'wind_da_ramp',
        'Nuclear availability': 'nuclear_avail',
        'Temperature': 'temp_mean',
        'Temperature 7d avg.': 'temp_mean_7d_avg',
        'Gas price': 'gas_price',
        'Oil price': 'oil_price',
        'Carbon price': 'carbon_price',
        'Generation day-ahead noise': 'generation_da noise',
        'Load day-ahead noise': 'load_da noise',
        'Solar day-ahead noise': 'solar_da noise',
        'Wind day-ahead noise': 'wind_da noise',
        'Generation day-ahead ramp noise': 'generation_da_ramp noise',
        'Load day-ahead ramp noise': 'load_da_ramp noise',
        'Solar day-ahead ramp noise': 'solar_da_ramp noise',
        'Wind day-ahead ramp noise': 'wind_da_ramp noise',
        'Nuclear availability noise': 'nuclear_avail noise',
        'Temperature noise': 'temp_mean noise',
        'Temperature 7d avg. noise': 'temp_mean_7d_avg noise',
        'Gas price noise': 'gas_price noise',
        'Oil price noise': 'oil_price noise',
        'Carbon price noise': 'carbon_price noise',
        'Day-ahead price': 'price_da'
    }
    old_name = new_to_old_name[new_name]
    return old_name

def plot_dependency(cf, name1, name2, fg_values, color=True, save=False, file_name='', figsize=(8, 7), x_label='', y_label='', color_label='', scale_color=1, scale_x=1):
    # note: if name2 is target feature, do not color graph (as shapley value directy indicates prediction value)
    for node1, d in cf.edge_credit.items():
        for node2, val in d.items():
            if node1.name == name1 and node2.name == name2:
                name1_old = get_old_feature_name(name1)
                df = fg_values[name1_old].copy()
                df_shap = pd.DataFrame(val.flatten(), index=df.index, columns=['shapley-flow'])
                df = pd.concat([df, df_shap], axis=1)
                plt.figure(figsize=figsize)
                plt.title('Dependence plot: {} → {}'.format(name1, name2))
                name2_old = get_old_feature_name(name2)
                if name2_old in fg_values.columns and color:
                    sc = plt.scatter(x=df[name1_old]/scale_x, y=df['shapley-flow'], s=5, c=fg_values[name2_old]/scale_color, cmap='viridis')
                    colorbar = plt.colorbar(sc)
                    if color_label == '':
                        colorbar.set_label(name2)
                    else: 
                        colorbar.set_label(color_label)
                else:
                    plt.scatter(x=df[name1_old]/scale_x, y=df['shapley-flow'], s=5)

                if x_label == '':
                    plt.xlabel(name1)
                else: 
                    plt.xlabel(x_label)
                if y_label == '':
                    plt.ylabel('Shapley flow value')
                else:
                    plt.ylabel(y_label)
                
                plt.tight_layout()
                if save:
                    plt.savefig("./plots/dependency_plots/{}_dependency_{}_{}.pdf".format(file_name, name1, name2))
                plt.show()
                return
    raise Exception("Feature not found in graph!")

# returns a dataframe containing the mean direct credit attribution (same as the SHAP attribution, i.e. edges in causal graph from input features to target features).
def get_mean_shap_attr(cf, target):
    dict = {}
    for node1, d in cf.edge_credit.items():
        for node2, val in d.items():
            if node2.name == target:
                dict[node1.name] = abs(val.flatten()).mean()
    return pd.DataFrame.from_dict(dict, orient='index', columns=['credit'])
              
def plot_beeswarm(cf, name1, name2, fg_values, color=False, save=False, file_name='', figsize=(8, 7)):
    for node1, d in cf.edge_credit.items():
        for node2, val in d.items():
            if node1.name == name1 and node2.name == name2:
                if color:
                    df = fg_values[name1].copy()
                    df_shap = pd.DataFrame(val.flatten(), index=df.index, columns=['shapley-flow'])
                    df = pd.concat([df, df_shap], axis=1)
                else:
                    df = pd.DataFrame(val.flatten(), index=fg_values.index, columns=['shapley-flow'])
                plt.figure(figsize=figsize)
                plt.title('Beeswarm plot: {} -> {}'.format(name1, name2))
                if name2 in fg_values.columns and color:
                    sc = plt.scatter(x=df[name1], y=df['shapley-flow'], s=5, c=fg_values[name2], cmap='viridis')
                    colorbar = plt.colorbar(sc)
                    colorbar.set_label(name2, fontsize=12)
                else:
                    sns.swarmplot(x=df['shapley-flow'])
                plt.xlabel(name1, fontsize=14)
                plt.ylabel('shapley-flow', fontsize=14)
                if save:
                    plt.savefig("./img/dependency_plots/{}_dependency_{}_{}.pdf".format(file_name, name1, name2))
                plt.show()
                return   

# save graph using this method to avoid overlapping edges (happens when using the save_graph method from shapley-flow)        
def save_graph_thesis(graph, path_file_name, format='pdf', view=False):
    # file name needs to be provided without file extension
    string = graph.string()
    G = graphviz.Source(string)
    G.render(path_file_name, format=format, view=view)

# rename a node in a shapley flow graph
def rename_node(g, old_name, new_name):
    node_names = []
    for x in g.nodes: node_names.append(x.name)
    index = node_names.index(old_name)
    g.nodes[index].name = new_name

# rename all nodes in shapley flow graph according to dictionary
def rename_all_nodes(g, dict):
    for old_name, new_name in dict.items():
        rename_node(g, old_name, new_name)

# rename nodes in creditflow object according to dict
def rename_nodes_in_graph(cf, dict):
    rename_all_nodes(cf.graph, dict)

def rename_nodes_in_graph_FR(cf):
    rename_dict = {
        'month': 'Month',
        'day_of_week': 'Day of week',
        'dayofyear_cos': 'Day of year cos',
        'dayofyear_sin': 'Day of year sin',
        'hour_cos': 'Hour cos',
        'hour_sin': 'Hour sin',
        'generation_da': 'Generation day-ahead',
        'load_da': 'Load day-ahead',
        'solar_da': 'Solar day-ahead',
        'wind_da': 'Wind day-ahead',
        'generation_da_ramp': 'Generation day-ahead ramp',
        'load_da_ramp': 'Load day-ahead ramp',
        'solar_da_ramp': 'Solar day-ahead ramp',
        'wind_da_ramp': 'Wind day-ahead ramp',
        'nuclear_avail': 'Nuclear availability',
        'temp_mean': 'Temperature',
        'temp_mean_7d_avg': 'Temperature 7d avg.',
        'gas_price': 'Gas price',
        'oil_price': 'Oil price',
        'carbon_price': 'Carbon price',
        'generation_da noise': 'Generation day-ahead noise',
        'load_da noise': 'Load day-ahead noise',
        'solar_da noise': 'Solar day-ahead noise',
        'wind_da noise': 'Wind day-ahead noise',
        'generation_da_ramp noise': 'Generation day-ahead ramp noise',
        'load_da_ramp noise': 'Load day-ahead ramp noise',
        'solar_da_ramp noise': 'Solar day-ahead ramp noise',
        'wind_da_ramp noise': 'Wind day-ahead ramp noise',
        'nuclear_avail noise': 'Nuclear availability noise',
        'temp_mean noise': 'Temperature noise',
        'temp_mean_7d_avg noise': 'Temperature 7d avg. noise',
        'gas_price noise': 'Gas price noise',
        'oil_price noise': 'Oil price noise',
        'carbon_price noise': 'Carbon price noise',
        'price_da': 'Day-ahead price'
    }
    rename_nodes_in_graph(cf, rename_dict)

def plot_bar_mean_abs_shap(cf, target, figsize=(6, 3.4), save=False, name='', xlabel='mean(|SHAP value|)'):
    df = get_mean_shap_attr(cf, target)
    df = df.sort_values(by='credit', ascending=True)
    df.plot(kind='barh', legend=False, color='#008bfb', figsize=figsize, width=0.7)
    plt.xlabel(xlabel)
    plt.tight_layout()
    if save:
        plt.savefig('./plots/bar_plot_mean_abs_shap_{}.pdf'.format(name))
    plt.show()

def plot_bar_mean_abs_asv(cf, figsize=(6, 3.4), save=False, name='', xlabel='mean(|ASV attribution|)'):
    sample_ind = -1
    cf.fold_noise = False
    # also draws graph, but easier to implement this way
    g = cf.draw_asv(idx=sample_ind, show_fg_val=True) 

    def remove_noise_from_end(s):
        suffix = ' noise'
        if s.endswith(suffix):
            s = s[:-len(suffix)]
        return s

    dict = {}
    for edge in g.edges():
        node_name, b = edge
        credit = float(edge.attr['label'])
        feature_name = remove_noise_from_end(node_name)
        dict[feature_name] = credit
    df = pd.DataFrame.from_dict(dict, orient='index', columns=['credit'])
    df = df.sort_values(by='credit', ascending=True)
    df.plot(kind='barh', legend=False, color='#008bfb', figsize=figsize, width=0.7)
    plt.xlabel(xlabel)
    plt.tight_layout()
    if save:
        plt.savefig('./plots/bar_plot_mean_abs_ASV_{}.pdf'.format(name))
    plt.show()