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
./{method}-{dataset}.sh
```

- `{method}`: `ditto` (ours) / `gcn` / `gin` / `brits` / `grin` / `spin`
- `{dataset}`: `ba-si` / `er-si` / `prost-si` / `oregon2-si` / `farmers-si` / `pol-si` / `ba-sir` / `er-sir` / `oregon2-sir` / `prost-sir` / `covid-sir` / `heb-sir`

## Baselines

- [BRITS \(Cao et al., 2018\)](https://github.com/caow13/BRITS)
- [GRIN \(Cini et al., 2022\)](https://github.com/Graph-Machine-Learning-Group/grin)
- [SPIN \(Marisca et al., 2022\)](https://github.com/Graph-Machine-Learning-Group/spin)