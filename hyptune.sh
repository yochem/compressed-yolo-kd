#!/usr/bin/env bash

# . ../venv/bin/activate

teacher=params/yolov5m.yaml
teacher_weights=runs/train/v5m/weights/best.pt
student=params/yolov5n.yaml
quantized_student=params/Qyolov5n.yaml

train_and_val() {
	name=$1
	shift
	echo python run.py train --name $name $@
	echo python run.py val --weights runs/train/$name/weights/best.pt
	echo
}

for hyp in 0 0.005 0.01 0.05 0.1 0.5 1 2 3 5 9; do
	train_and_val hyp-$hyp \
		--cfg $quantized_student \
		--comphyp $hyp \
		--device 1
done


#echo "Sleeping now... almost shutting down"
#sleep 10
#Echo "done at $(date)" > done.txt
#sudo shutdown now
