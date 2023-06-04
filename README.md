# DITTO

Official code for the KDD 2023 paper [Reconstructing graph diffusion history from a single snapshot](https://doi.org/10.1145/3580305.3599488).

If you use our code, please cite our paper:

```bibtex
@inproceedings{qiu2023ditto,
  title={Reconstructing graph diffusion history from a single snapshot},
  author={Ruizhong Qiu and Dingsu Wang and Lei Ying and Poor, {H. Vicent} and Yifang Zhang and Hanghang Tong},
  booktitle={Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  year={2023},
  organization={ACM},
  address={Long Beach, CA, USA},
  doi={10.1145/3580305.3599488}
}
```

## Environment

Our code was tested under the following environment:

- Intel Xeon CPU @ 2.20GHz
- NVIDIA Tesla P100 16GB GPU
- CUDA 11.4
- `torch==1.7.0`
- `class-resolver==0.3.10`
- `torch-scatter==2.0.7`
- `torch-sparse==0.6.9`
- `torch-cluster==1.5.9`
- `torch-geometric==2.0.4`
- `ndlib==5.1.1`
- `geopy==2.1.0`