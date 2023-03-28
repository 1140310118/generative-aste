# bash/train_extractor.sh -d 14res

while getopts ':d:c:b:s:l:o:' opt
do
    case $opt in
        d)
        dataset="$OPTARG" ;;
        c)
        CUDA_IDS="$OPTARG" ;;
        b)
        subname="$OPTARG" ;;
        s)
        seed="$OPTARG" ;;
        l)
        learning_rate="$OPTARG" ;;
        o)
        output_dir="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done



if [ ! "${CUDA_IDS}" ]
then
    CUDA_IDS=0
fi



if [ ! "${subname}" ]
then
    subname="test"
fi



if [ ! "${seed}" ]
then
    seed=42
fi



if [ ! "${learning_rate}" ]
then
    learning_rate=20
fi



if [ ! "${output_dir}" ]
then
    output_dir="/data/zhangyice/2023/data-augmentation/test/output/extraction"
fi



max_seq_length=-1
gradient_clip_val=1
warmup_steps=0
weight_decay=0.01

precision=16
train_batch_size=16
eval_batch_size=64
max_epochs=20

model_name_or_path="t5-base"
# model_name_or_path="/data/zhangyice/2023/pretrained_models/t5-base"
data_dir="data/t5/"



CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train.py \
  --accelerator=gpu \
  --devices=1 \
  --precision=${precision} \
  --data_dir "${data_dir}" \
  --model_name_or_path "${model_name_or_path}" \
  --output_dir "${output_dir}" \
  --learning_rate ${learning_rate}e-5 \
  --train_batch_size ${train_batch_size} \
  --eval_batch_size ${eval_batch_size} \
  --seed ${seed} \
  --warmup_steps ${warmup_steps} \
  --gradient_clip_val ${gradient_clip_val} \
  --weight_decay ${weight_decay} \
  --max_seq_length ${max_seq_length} \
  --output_sub_dir ${subname} \
  --dataset ${dataset} \
  --max_epochs ${max_epochs} \
  --do_train
