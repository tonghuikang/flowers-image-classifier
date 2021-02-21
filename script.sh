python train.py "./flowers" --arch densenet169 --gpu --epochs=100 > ./logs/finetune_only_top_layer.txt
python train.py "./flowers" --arch densenet169 --gpu --train_all_layers --epochs=100 > ./logs/finetune_whole_model.txt
python train.py "./flowers" --arch densenet169 --gpu --not_use_pretrained --train_all_layers --epochs=100 > ./logs/train_from_scratch.txt