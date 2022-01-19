cd src

python run.py \
    --gpu 0 \
    --workers 1 \
    --data balance-scale \
    --model TART \
    --depth 2 \
    --layers 1 \
    --style ensemble
