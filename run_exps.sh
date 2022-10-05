for DS in 0 1 2
do
  for TRAIN_RATIO in 100 20
  do
    python main.py --ds $DS --train_ratio $TRAIN_RATIO
    python main.py --ds $DS --train_ratio $TRAIN_RATIO --dual_no_time
    python main.py --ds $DS --train_ratio $TRAIN_RATIO --time_no_agg
    python main.py --ds $DS --train_ratio $TRAIN_RATIO --sub_exps
  done
  python main.py --ds $DS --unsup
done
