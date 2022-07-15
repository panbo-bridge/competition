path=$1
echo "模型预测开始，一共有9个tilte匹配模型，3个attr匹配模型"
echo "第一个title模型预测"

#参数解释
#--model 选择模型进行预测
#--mode 预测attr还是title
#--load_model_name 选择加载的预测模型路径
#--output 输出该模型的预测结果路径
#--testdata_path 测试集路径
#--maz_seq_length 文本长度
python3 code/pred.py \
    --model visualbert \
    --mode title \
   --load_model_name ../input/project/data/best_model_visualbert_title_1fold.pth \
    --output ./data/tmp_data/testB_title_visualbert_1fold.txt \
    --testdata_path $path \
    --max_seq_length 31

#python3 code/pred.py \
#    --model visualbert \
#    --mode title \
#    --load_model_name ./data/best_visualbert_title_B20.pth \
#    --output ./data/tmp_data/testB_title_visualbert_1fold.txt \
#    --testdata_path $path \
#    --max_seq_length 31

echo "第2个title模型预测"
python3 code/pred.py \
    --model visualbert \
    --mode title \
    --load_model_name ./data/best_model_visualbert_title_2fold.pth \
    --output ./data/tmp_data/testB_title_visualbert_2fold.txt \
    --testdata_path $path \
    --max_seq_length 31 

echo "第3个title模型预测"
python3 code/pred.py \
    --model visualbert \
    --mode title \
    --load_model_name ./data/best_model_visualbert_title_3fold.pth \
    --output ./data/tmp_data/testB_title_visualbert_3fold.txt \
    --testdata_path $path \
    --max_seq_length 31 

echo "第4个title模型预测"
python3 code/pred.py \
    --model visualbert \
    --mode title \
    --load_model_name ./data/best_model_visualbert_title_4fold.pth \
    --output ./data/tmp_data/testB_title_visualbert_4fold.txt \
    --testdata_path $path \
    --max_seq_length 31 

echo "第5个title模型预测"
python3 code/pred.py \
    --model visualbert \
    --mode title \
    --load_model_name ./data/best_model_visualbert_title_5fold.pth \
    --output ./data/tmp_data/testB_title_visualbert_5fold.txt \
    --testdata_path $path \
    --max_seq_length 31 

echo "第6个title模型预测"
python3 code/pred.py \
    --model visualbert \
    --mode title \
    --load_model_name ./data/best_model_visualbert_title_6fold.pth \
    --output ./data/tmp_data/testB_title_visualbert_6fold.txt \
    --testdata_path $path \
    --max_seq_length 31 

echo "第7个title模型预测"
python3 code/pred.py \
    --model visualbert \
    --mode title \
    --load_model_name ./data/best_model_visualbert_title_7fold.pth \
    --output ./data/tmp_data/testB_title_visualbert_7fold.txt \
    --testdata_path $path \
    --max_seq_length 31 

echo "第8个title模型预测"
python3 code/pred.py \
    --model uniter \
    --mode title \
    --load_model_name ./data/best_model_uniter_title.pth \
    --output ./data/tmp_data/testB_title_uniter.txt \
    --testdata_path $path \
    --max_seq_length 31 

echo "第9个title模型预测"
python3 code/pred.py \
    --model lxmert \
    --mode title \
    --load_model_name ./data/best_model_lxmert_title.pth \
    --output ./data/tmp_data/testB_title_lxmert.txt \
    --testdata_path $path \
    --max_seq_length 31 

echo "第1个attr模型预测"
python3 code/pred.py \
    --model visualbert \
    --mode attr \
    --load_model_name ./data/best_model_visualbert_attr.pth \
    --output ./data/tmp_data/testB_attr_visualbert.txt \
    --testdata_path $path \
    --max_seq_length 4 

echo "第2个attr模型预测"
python3 code/pred.py \
    --model uniter \
    --mode attr \
    --load_model_name ./data/best_model_uniter_attr.pth \
    --output ./data/tmp_data/testB_attr_uniter.txt \
    --testdata_path $path \
    --max_seq_length 4 

echo "第3个attr模型预测"
python3 code/pred.py \
    --model lxmert \
    --mode attr \
    --load_model_name ./data/best_model_lxmert_attr.pth \
    --output ./data/tmp_data/testB_attr_lxmert.txt \
    --testdata_path $path \
    --max_seq_length 4 


python3 ./code/attr_title_merge.py  #合并属性匹配和标题匹配的预测结果
echo "结果预测已完成"













