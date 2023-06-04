cd .. && python3 ./dhrec.py \
    --dataset prost-sir \
    --seed 123456789 \
    --data_dir input \
    --device $1 \
    --output output/dhrec-prost-sir.pt \
    --b_pI0 0.001 \
    --b_pR0 0.001 \
    --b_steps 500 \
    --b_lr 0.003
