cd .. && python3 ./dhrec.py \
    --dataset oregon2-sir \
    --seed 123456789 \
    --data_dir input \
    --device cuda:0 \
    --output output/dhrec-oregon2-sir.pt \
    --b_pI0 0.001 \
    --b_pR0 0.001 \
    --b_steps 500 \
    --b_lr 0.003
