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
    --load_model_name /home/mw/input/just5571/best_visualbert_B3.18_19.pth \
    --output ./data/tmp_data/testB_title_visualbert_B3.18_19.txt \
    --testdata_path $path \
    --max_seq_length 31

#echo "第1个attr模型预测"
python3 code/pred.py \
    --model visualbert \
    --mode attr \
    --load_model_name /home/mw/input/just5571/best_visualbert_B3.18_19.pth \
   --output ./data/tmp_data/testB_attr_visualbert_B3.18_19.txt \
    --testdata_path $path \
    --max_seq_length 31 

#python3 ./code/attr_title_merge_one.py  #合并属性匹配和标题匹配的预测结果
#echo "结果预测已完成"













