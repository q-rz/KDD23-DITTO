# Reconstructing graph diffusion history from a single snapshot (KDD 2023)

If you use our code, please cite our paper:

```bibtex
@inproceedings{qiu2023ditto,
  title={Reconstructing graph diffusion history from a single snapshot},
  author={Ruizhong Qiu and Dingsu Wang and Lei Ying and {H. Vincent} Poor and Yifang Zhang and Hanghang Tong},
  booktitle={Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  year={2023},
  organization={ACM},
  address={Long Beach, CA, USA},
  doi={10.1145/3580305.3599488}
}
```

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
./{method}-{dataset}.sh
```

- The `{method}` is one of `ditto` (ours) / `dhrec` / `cri` / `gcn` / `gin`.
  - The original code for DHREC is for SEIRS and is rather complicated, so we provide our simplified implementation for SI/SIR here.
  - The CRI paper did not publish their source code, so we implemented CRI according to their paper.
  - The implementations of GCN and GIN are from PyTorch Geometric.

- The `{dataset}` is one of `ba-si` / `er-si` / `oregon2-si` / `prost-si` / `farmers-si` / `pol-si` / `ba-sir` / `er-sir` / `oregon2-sir` / `prost-sir` / `covid-sir` / `heb-sir`.
  - Here `farmers-si` refers to the BrFarmers dataset in our paper.
  - Here `heb-sir` refers to the Hebrew dataset in our paper.

## Other baselines

- BRITS \(Cao et al., 2018\): [code](https://github.com/caow13/BRITS)
- GRIN \(Cini et al., 2022\): [code](https://github.com/Graph-Machine-Learning-Group/grin)
- SPIN \(Marisca et al., 2022\): [code](https://github.com/Graph-Machine-Learning-Group/spin)