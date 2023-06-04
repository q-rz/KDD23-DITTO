cd .. && python3 ./cri.py \
    --dataset covid-sir \
    --seed 123456789 \
    --data_dir input \
    --device $1 \
    --output output/cri-covid-sir.pt
