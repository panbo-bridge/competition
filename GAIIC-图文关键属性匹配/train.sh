mkdir ./data/tmp_data/title_dataset #创建title-img任务数据集保存文件夹
mkdir ./data/tmp_data/attr_dataset #创建attr-img任务数据集保存文件夹
python3 code/built_dataset.py   #重新构造title-img匹配数据集
python3 code/built_dataset_attr.py #重新构造attr-img匹配的数据集
#python3 code/data_process.py #进行数据集的划分，因为要划分多个不同的10fold数据集，每次需要修改代码里面的测试集的取法，所以提前生成好测试集的划分保存在code文件夹下（title_train_*fold.pth,attr_train.pth）

#训练title-img匹配模型
#参数说明：
#--model    模型训练函数
#--mode     训练title-img还是attr-img
#--save_model_name 模型训练完保存的名字
#--train_pth_path 模型训练使用的是哪一个fold的数据划分
#--dataset_path 训练数据集的路径
#--batchSize 训练的batchsize
#--lr 学习率
#--epochs 学习额epoch数
#max_seq_length 文本的最大长度


python3 code/vqa.py \
    --model visualbert \
    --mode title \
    --save_model_name ./data/model_data/best_model_visualbert_title_1fold.pth \
    --train_pth_path ./code/title_train_1.pth \
    --dataset_path ./data/tmp_data/title_dataset/ \
    --batchSize 128 \
    --lr 1e-5 \
    --epochs 100 \
    --max_seq_length 31

python3 code/vqa.py \
    --model visualbert \
    --mode title \
    --save_model_name ./data/model_data/best_model_visualbert_title_2fold.pth \
    --train_pth_path ./code/title_train_2.pth \
    --dataset_path ./data/tmp_data/title_dataset/ \
    --batchSize 128 \
    --lr 1e-5 \
    --epochs 100 \
    --max_seq_length 31
python3 code/vqa.py \
    --model visualbert \
    --mode title \
    --save_model_name ./data/model_data/best_model_visualbert_title_3fold.pth \
    --train_pth_path ./code/title_train_3.pth \
    --dataset_path ./data/tmp_data/title_dataset/ \
    --batchSize 128 \
    --lr 1e-5 \
    --epochs 100 \
    --max_seq_length 31
python3 code/vqa.py \
    --model visualbert \
    --mode title \
    --save_model_name ./data/model_data/best_model_visualbert_title_4fold.pth \
    --train_pth_path ./code/title_train_4.pth \
    --dataset_path ./data/tmp_data/title_dataset/ \
    --batchSize 128 \
    --lr 1e-5 \
    --epochs 100 \
    --max_seq_length 31

python3 code/vqa.py \
    --model visualbert \
    --mode title \
    --save_model_name ./data/model_data/best_model_visualbert_title_5fold.pth \
    --train_pth_path ./code/title_train_5.pth \
    --dataset_path ./data/tmp_data/title_dataset/ \
    --batchSize 128 \
    --lr 1e-5 \
    --epochs 100 \
    --max_seq_length 31
python3 code/vqa.py \
    --model visualbert \
    --mode title \
    --save_model_name ./data/model_data/best_model_visualbert_title_6fold.pth \
    --train_pth_path ./code/title_train_6.pth \
    --dataset_path ./data/tmp_data/title_dataset/ \
    --batchSize 128 \
    --lr 1e-5 \
    --epochs 100 \
    --max_seq_length 31
python3 code/vqa.py \
    --model visualbert \
    --mode title \
    --save_model_name ./data/model_data/best_model_visualbert_title_7fold.pth \
    --train_pth_path ./code/title_train_7.pth \
    --dataset_path ./data/tmp_data/title_dataset/ \
    --batchSize 128 \
    --lr 1e-5 \
    --epochs 100 \
    --max_seq_length 31

python3 code/vqa.py \
    --model uniter \
    --mode title \
    --save_model_name ./data/model_data/best_model_uniter_title.pth \
    --train_pth_path ./code/title_train_1.pth \
    --dataset_path ./data/tmp_data/title_dataset/ \
    --batchSize 128 \
    --lr 1e-5 \
    --epochs 100 \
    --max_seq_length 31

python3 code/vqa.py \
    --model lxmert \
    --mode title \
    --save_model_name ./data/model_data/best_model_lxmert_title.pth \
    --train_pth_path ./code/title_train_1.pth \
    --dataset_path ./data/tmp_data/title_dataset/ \
    --batchSize 128 \
    --lr 1e-5 \
    --epochs 100 \
    --max_seq_length 31


#属性模型训练
#--pretrain_model 训练完成的title模型作为attr模型的预训练模型
python3 code/vqa.py \
    --model visualbert \
    --mode attr \
    --save_model_name ./data/model_data/best_model_visualbert_attr.pth \
    --train_pth_path ./code/attr_train.pth \
    --dataset_path ./data/tmp_data/attr_dataset/ \
    --pretrain_model ./data/best_model_visualbert_title_c5.pth \
    --batchSize 512 \
    --lr 1e-5 \
    --epochs 100 \
    --max_seq_length 4

python3 code/vqa.py \
    --model uniter \
    --mode attr \
    --save_model_name ./data/model_data/best_model_uniter_attr.pth \
    --train_pth_path ./code/attr_train.pth \
    --dataset_path ./data/tmp_data/attr_dataset/ \
    --pretrain_model ./data/model_data/best_model_uniter_title.pth \
    --batchSize 512 \
    --lr 1e-5 \
    --epochs 100 \
    --max_seq_length 4

python3 code/vqa.py \
    --model lxmert \
    --mode attr \
    --save_model_name ./data/model_data/best_model_lxmert_attr.pth \
    --train_pth_path ./code/attr_train.pth \
    --dataset_path ./data/tmp_data/attr_dataset/ \
    --pretrain_model ./data/model_data/best_model_lxmert_title.pth \
    --batchSize 512 \
    --lr 1e-5 \
    --epochs 100 \
    --max_seq_length 4

