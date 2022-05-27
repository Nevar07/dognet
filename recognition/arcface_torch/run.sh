
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 \
--nnodes=1 \
--node_rank=0 \
--master_addr="127.0.0.1" \
--master_port=12345 train.py $@

ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh

# python train.py \
#     --config configs/dog.py

# #
# python train_cls.py \
#   --network conv_base \
#   --save_dir models/cb_1/cls \
#   --lr 0.01
  

python inf.py \
    --network swtb \
    --weight models/swtb_1/best_model.pt \
    --cls_weight best_cls.pt \
    --out_name valid_swtb_1.csv
