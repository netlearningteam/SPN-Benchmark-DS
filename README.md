# The Benchmark Datasets for Stochastic Petri Net Learning

The datasets and related codes in the paper "The Benchmark Datasets for Stochastic Petri Net Learning".

## 1. Datasets 

### 1.1 Download

We provide two download methods, including Baidu Cloud and Google Cloud.  And their links are as follows:

>Baidu cloud link:  https://pan.baidu.com/s/1LTj6ShnH5JSSRjLI3pAl2A   password :  5mg7
>
>Google cloud link : https://drive.google.com/file/d/1cBjJsT3dC3AOon3U5SfdEzYxo3ki33Ax/view?usp=sharing

### 1.2 File Structure

> │  
> ├─GridData
> │  ├─DS1
> │  │  ├─ori_data
> │  │  │      test_data.json
> │  │  │      train_data.json
> │  │  │      
> │  │  ├─package_data
> │  │  │      dataset.pkl
> │  │  │      
> │  │  └─preprocessd_data
> │  │          test_data.json
> │  │          train_data.json
> │  │          
> │  ├─DS2
> │  │  ├─ori_data
> │  │  │      test_data.json
> │  │  │      train_data.json
> │  │  │      
> │  │  ├─package_data
> │  │  │      dataset.pkl
> │  │  │      
> │  │  └─preprocessd_data
> │  │          test_data.json
> │  │          train_data.json
> │  │          
> │  ├─DS3
> │  │  ├─ori_data
> │  │  │      test_data.json
> │  │  │      train_data.json
> │  │  │      
> │  │  ├─package_data
> │  │  │      dataset.pkl
> │  │  │      
> │  │  └─preprocessd_data
> │  │          test_data.json
> │  │          train_data.json
> │  │          
> │  ├─DS4
> │  │  ├─ori_data
> │  │  │      test_data.json
> │  │  │      train_data.json
> │  │  │      
> │  │  ├─package_data
> │  │  │      dataset.pkl
> │  │  │      
> │  │  └─preprocessd_data
> │  │          test_data.json
> │  │          train_data.json
> │  │          
> │  └─DS5
> │      ├─ori_data
> │      │      test_data.json
> │      │      train_data.json
> │      │      
> │      ├─package_data
> │      │      dataset.pkl
> │      │      
> │      └─preprocessd_data
> │              test_data.json
> │              train_data.json
> │              
> └─RandData
>     ├─DS1
>     │  ├─ori_data
>     │  │      test_data.json
>     │  │      train_data.json
>     │  │      
>     │  ├─package_data
>     │  │      dataset.pkl
>     │  │      
>     │  └─preprocessd_data
>     │          test_data.json
>     │          train_data.json
>     │          
>     ├─DS2
>     │  ├─ori_data
>     │  │      test_data.json
>     │  │      train_data.json
>     │  │      
>     │  ├─package_data
>     │  │      dataset.pkl
>     │  │      
>     │  └─preprocessd_data
>     │          test_data.json
>     │          train_data.json
>     │          
>     └─DS3
>         ├─ori_data
>         │      test_data.json
>         │      train_data.json
>         │      
>         ├─package_data
>         │      dataset.pkl
>         │      
>         └─preprocessd_data
>                 test_data.json
>                 train_data.json



#### **Description:**

**GridData**: Store the data of the grid organization.

**RandData**: Store the data of the randomized organization.

**DS+{index}**: The location(directory)  where the index of the data set is stored.  For example, the storage location (directory) of the first data set is DS1.

**ori_data**: Store the information of a petri net, including the original petri net (![](https://latex.codecogs.com/gif.latex?{A^{&plus;}}^{T}$,&space;${A^{-}}^{T}$,&space;${M_{0}}^{T})), the reachable marking graph, the average firing rate λ, the average number of tokens and so on, where ![](https://latex.codecogs.com/gif.latex?A^&plus;,A^-)are the input matrix and output matrix of the Petri net, ![](https://latex.codecogs.com/gif.latex?M_0)is the initial marking, λ is the average firing rate of a transition. 

**preprocessd_data**: Preprocess the original ori_data into the input information of the deep learning algorithm.

**package_data** : Convert the data in **preprocessd_data** into DGL data type and pack it into pkl format. This will speed up the time to read data each time.

**test_data.json**: store test data.

**train_data.json**: store training data.



### 1.3 Data structure description

We save the data as a json file. There are two forms of data, including: the unprocessed raw SPN data (**ori_data**), and **the** input data of net learning algorithms (**preprocessd_data**) obtained after preprocessing the original SPN data. **Ori_data** and **preprocessd_data** in each data set contain two json files, including: the training set (train_data.json) and the  test set (test_data.json). The json  structure is as follows:

 

**ori_data/train_data.json**



> ├─data1
>
> │ │   petri_net:
>
> │ │   arr_vlist:
>
> │ │   arr_edge:
>
> │ │   arr_tranidx:
>
> │ │   spn_labda:
>
> │ │   spn_steadypro:
>
> │ │   spn_markdens:
>
> │ │   spn_mu:
>
> └─data2
>
> │ │   petri_net:
>
> │ │   arr_vlist:
>
> │ │   arr_edge:
>
> │ │   arr_tranidx:
>
> │ │   spn_labda:
>
> │ │   spn_steadypro:
>
> │ │   spn_markdens:
>
> │ │   spn_mu:



**preprocessd_data/train_data.json**



> ├─data1
>
> │ │   node_f:
>
> │ │   edge_index:
>
> │ │   edge_f:
>
> │ │   label:
>
> └─data2
>
> │ │   node_f:
>
> │ │   edge_index:
>
> │ │   edge_f:
>
> │ │   label:
>
> 

 

#### **Description**:  

**ori_data:**



**data+{index}**: Store the index data.

**petri_net**: Original petri net structure,(![](https://latex.codecogs.com/gif.latex?{A^{&plus;}}^{T}$,&space;${A^{-}}^{T}$,&space;${M_{0}}^{T})).

**arr_vlist**: The vertex set of the reachable marking graph.

**arr_edge**: The set of reachable marking  graph edge indexes. For example: [0,1] means that there is a directed arc from 0→1 between the 0th vertex and the 1th vertex.

**arr_tranidx**: The number of transitions in the original petri net on the reachable marking graph.

**spn_labda**: The λ corresponding to each arc of the reachable marking  graph.  Where λ is the the average firing rate of transitions.

**spn_steadypro**:  The steady state probability.

**spn_markdens**: The token probability density function.

**spn_mu**: The average number of tokens.

 

**preprocessd_data**:



**data+{index}**: Store the index data.

**node_f**: Node features.

**edge_index**: Edge index.

**edge_f**: Edge features.

**label**: The labels.



## 2. Run code

1. Download all codes on Github.
2. Download the data set and put the two files after decompression: RandData and GridData in the Data directory.
3. Install the dependent packages in the requirements.txt file.
4. Run code.

```
python main_spn_grid_data.py
```

```
python main_spn_rand_data.py
```

```
python SPNGenerate.py
```

