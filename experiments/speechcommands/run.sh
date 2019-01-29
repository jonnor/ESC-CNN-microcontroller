
python3 train.py --data_dir=../../data/speech_dataset/ --train_dir=./train 
--how_many_training_steps=10000,3000 --learning_rate=0.01,0.001 --model_architecture=low_latency_conv 
--preprocess=average
