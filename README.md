# Reconstructing graph diffusion history from a single snapshot \(KDD 2023\)

[![Our KDD paper](https://img.shields.io/badge/2023-KDD-blue)]([https://openreview.net/forum?id=9u05zr0nhx](https://doi.org/10.1145/3580305.3599488)) [![Our paper on arXiv](https://img.shields.io/badge/2306.00488-arXiv-B31B1B.svg)](https://arxiv.org/abs/2306.00488)

```bibtex
@inproceedings{qiu2023ditto,
  title={Reconstructing graph diffusion history from a single snapshot},
  author={Ruizhong Qiu and Dingsu Wang and Lei Ying and {H. Vincent} Poor and Yifang Zhang and Hanghang Tong},
  booktitle={Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  year={2023}
}
```

![Illustration](https://github.com/q-rz/KDD23-DITTO/raw/main/dash-illus.svg)

## Dependencies

Our code was tested under the following dependencies:

- CUDA 11.4
- `torch==1.7.0`
- `class-resolver==0.3.10`
- `torch-scatter==2.0.7`
- `torch-sparse==0.6.9`
- `torch-cluster==1.5.9`
- `torch-geometric==2.0.4`
- `ndlib==5.1.1`
- `geopy==2.1.0`

## Usage

To reproduce our results:

```sh
cd scripts
./{method}-{dataset}.sh {device}
```

- `{method}`: `ditto` (ours) / `dhrec` / `cri` / `gcn` / `gin`.
  - The original code for DHREC is specially for SEIRS, so we provide our implementation of DHREC-PCDSVC for SI & SIR here.
  - The CRI paper did not publish their source code, so we implemented CRI according to their paper.
  - The implementations of GCN and GIN are from PyTorch Geometric.
- `{dataset}`: `ba-si` / `er-si` / `oregon2-si` / `prost-si` / `farmers-si` (BrFarmers) / `pol-si` / `ba-sir` / `er-sir` / `oregon2-sir` / `prost-sir` / `covid-sir` / `heb-sir` (Hebrew).
  - **Notice:** As is explained in Section 5.4, {`gcn`, `gin`} were evaluated only on {`farmers-si`, `pol-si`, `covid-sir`, `heb-sir`}.
- `{device}`: the device for PyTorch.

## Other baselines

- BRITS \(Cao et al., 2018\): [paper](https://proceedings.neurips.cc/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf), [code](https://github.com/caow13/BRITS)
- GRIN \(Cini et al., 2022\): [paper](https://openreview.net/pdf?id=kOu3-S3wJ7), [code](https://github.com/Graph-Machine-Learning-Group/grin)
- SPIN \(Marisca et al., 2022\): [paper](https://arxiv.org/pdf/2205.13479.pdf), [code](https://github.com/Graph-Machine-Learning-Group/spin)
