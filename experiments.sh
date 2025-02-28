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

# normal student
train_and_val v5n \
	--cfg $student

# normal teacher
train_and_val v5m \
	--cfg $teacher

# offline kd
train_and_val offline \
	--cfg $student \
	--teacher-weight $teacher_weights

# offline kd comp
train_and_val offline-comp \
	--cfg $quantized_student \
	--teacher-weight $teacher_weights \
	--comphyp 1.0

# online kd
train_and_val kdcl \
	--cfg $student \
	--teacher-weight $teacher \
	--colab

# online kd comp
train_and_val kdcl-comp \
	--cfg params/Qyolov5n.yaml \
	--teacher-weight $teacher \
	--colab \
	--comphyp 1.0

#echo "Sleeping now... almost shutting down"
#sleep 10
#Eecho "done at $(date)" > done.txt
#sudo shutdown now
