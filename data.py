import re
import subprocess
import numpy as np
import hypernetx as hnx
from os import makedirs, getcwd
import torch_geometric.utils as tutils
from tqdm.notebook import tqdm

def create_subhgraphs(bottoms, modes, symbols, predicates, types, sat):
    dataset = []
    y = []
    for targets, pin in enumerate(["neg","pos"]):
        for bottom,s in tqdm(zip(bottoms[pin],sat[pin]), desc="feats"):
            subh = hnx.Hypergraph(createHfrombot(bottom,modes))
            nodes = list(subh.nodes)
            hedges = list(subh.edges)
            arestas = [(v, e) for e in subh.edges for v in subh.edges[e]]
            hyperedge_ = []
            
            mapping_he = dict(zip(list(set(hedges)), range(len(list(set(hedges))))))
            mapping_v = dict(zip(list(set(nodes)), range(len(list(set(nodes))))))
            edge_index = torch.empty((2, len(arestas)), dtype=torch.long)
            for i, (src, dst) in enumerate(arestas):
                aux = np.zeros((len(predicates)))
                aux[np.where(np.array(predicates) == re.findall(r'([a-z_\d]+)_\d+',dst))[0][0]] = 1
                hyperedge_.append(aux)
                edge_index[0, i] = mapping_v[src]
                edge_index[1, i] = mapping_he[dst]
            hyperedge_ = np.array(hyperedge_)
            num_nodes = len(mapping_v)
            num_edges = len(mapping_he)
            x = [0]*num_nodes
            for i in nodes:
                feats = np.array([])

                no_pred = False
                feat = np.zeros((len(predicates)+1))
                if i[1] == "No_predicado":
                    no_pred = True
                    feat[np.where(np.array(predicates) == re.findall(r'([a-z_\d]+)_\d+',i[0].strip())[0])[0][0]+1] = 1
                feats = np.concatenate((feats,feat))

                
                feat = np.zeros((len(types) if len(types) != 0 else 1))
                if not no_pred:
                    feat[np.where(np.array(types) == i[0].strip())[0][0]] = 1
                feats = np.concatenate((feats,feat))
    
                
                feat = np.zeros((len(symbols) if len(symbols) != 0 else 1))
                if not no_pred:
                    iiii = re.findall(r'(.*?)__\d+',i[1].strip())
                    if len(iiii):
                        iiii = iiii[0]
                        if iiii in symbols:
                            print(f"i = {i[1]}")
                            feat[np.where(np.array(symbols) == iiii)[0][0]] = 1
                feats = np.concatenate((feats,feat))
                
                try:
                    iiii = re.findall(r'(.*?)__\d+',i[1].strip())
                    feat = np.array([float(iiii[0])])
                except (ValueError, TypeError,IndexError):
                    feat = np.array([float(0)])
    
                feats = np.concatenate((feats,feat))
                x[mapping_v[i]] = feats.copy()
            if len(x) != num_nodes:
                print("Error")
            dataset.append(Data(x=torch.tensor(x, dtype=torch.float), 
                                num_nodes=num_nodes, 
                                edge_index=torch.tensor(edge_index, dtype=torch.long),
                                hyperedge_=torch.tensor(hyperedge_, dtype=torch.float),
                                y=torch.tensor(targets, dtype=torch.float)
                               ))
            y.append(targets)
    return dataset, y


def createHfrombot(bot,modes,aresta = 0):
    hypergrafo = {}
    aaaa = 0
    for b in bot:
        haresta = []
        predicado = re.findall(r'([a-z\d_]+)\(', b)[0]
        data = re.findall(r'\((.*?)\)', b)


        for e,j in enumerate(data[0].split(",")):
            if len(j.strip()) == 0: continue
            if not j.isupper():
                j = f"{j}__{aaaa}"
                aaaa+=1
            haresta.append((modes[predicado][e], j))
        haresta.append((f"{predicado}_{aresta}", "No_predicado"))
        hypergrafo[f"{predicado}_{aresta}"] = haresta
        aresta+=1
    return hypergrafo

def aleph_settings(mode_file, bk_file, depth, data_files={}):
    script_lines = []
    # script_lines += [f':- set(verbosity, 0).']
    script_lines += [f':- set(i,{depth+1}).']
    # script_lines += [f':- set(check_useless,true).']
    script_lines += [f':- set(minpos,2).']
    for set_name, data_file in data_files.items():
        script_lines += [f':- set({set_name}, "{data_file}").']
    script_lines += [f':- read_all("{mode_file}").']
    script_lines += [f':- read_all("{bk_file}").']
    return script_lines

def load_examples(file_name):
    with open(file_name, 'r') as f:
        lines = [l.strip('. \n') for l in f.readlines()]
    return np.array(lines)

def create_script(directory, script_lines):
    file_name = f"{directory}/script.pl"
    with open(file_name, 'w') as f:
        f.writelines([l + '\n' for l in script_lines + [':- halt.']])
    return file_name

def run_aleph(script_file):
    aleph_file = get_aleph()
    cmd = f'{getcwd()}/SWI-Prolog.app/Contents/MacOS/swipl -f {aleph_file} -l {script_file}'
    return execute(cmd, return_output=True)

def get_aleph():
    return f'{getcwd()}/CILP-master/cilp/aleph.pl'

def execute(cmd, return_output=False):
    popen = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             shell=True,
                             universal_newlines=True)

    if return_output:
        output, err = popen.communicate()
    else:
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line.rstrip())
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        print('Subproc error:')
        raise subprocess.CalledProcessError(return_code, cmd)

    if return_output:
        return output
