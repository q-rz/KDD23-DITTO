cd .. && python3 ./dhrec.py \
    --dataset prost-si \
    --seed 123456789 \
    --data_dir input \
    --device cuda:0 \
    --output output/dhrec-prost-si.pt \
    --b_pI0 0.001 \
    --b_pR0 0.001 \
    --b_steps 500 \
    --b_lr 0.003
