# bash/do_extraction.sh -b test.json -d data/t5/14res/test.json

while getopts ':d:c:b:m:o:v:' opt
do
    case $opt in
        d)
        data_dir="$OPTARG" ;;
        c)
        CUDA_IDS="$OPTARG" ;;
        b)
        subname="$OPTARG" ;;
        m)
        model_name_or_path="$OPTARG" ;;
        o)
        output_dir="$OPTARG" ;;
        v)
        version="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done



if [ ! "${CUDA_IDS}" ]
then
    CUDA_IDS=0
fi


if [ ! "${data_dir}" ]
then
    data_dir="data/t5/14res/test.json"
fi


if [ ! "${model_name_or_path}" ]
then
    model_name_or_path="/data/zhangyice/2023/data-augmentation/test/output/extraction/model/dataset=14res,b=test,seed=42"
fi


if [ ! "${output_dir}" ]
then
    output_dir="/data/zhangyice/2023/data-augmentation/test/output/extraction"
fi


if [ ! "${version}" ]
then
    version="v1"
fi



seed=42
max_seq_length=-1

precision=16
eval_batch_size=250




CUDA_VISIBLE_DEVICES=${CUDA_IDS} python do_extraction.py \
  --gpus=1 \
  --precision=${precision} \
  --data_dir "${data_dir}" \
  --model_name_or_path "${model_name_or_path}" \
  --output_dir "${output_dir}" \
  --eval_batch_size ${eval_batch_size} \
  --seed $seed \
  --max_seq_length ${max_seq_length} \
  --output_sub_dir ${subname} \
  --dataset_version ${version}