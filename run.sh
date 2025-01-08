#!/usr/bin/env bash

DATA=data/bears.yaml
EPOCHS=30

. ../venv/bin/activate

usage() {
	[ -n $1 ] && { echo "$1"; echo }
	echo "Usage: $0 <command> [args]"
	echo "Commands:"
	echo "	export <model>"
	echo "	kd <name>"
	echo "	single <name>"
	echo "	val <model>"
	exit 1
}

[ $# -eq 0 ] && usage

case "$1" in
	export)
		[ $# -ne 2 ] && usage "Error: '$1' command requires a model argument."
		model=$2
		python export.py \
			--data "$DATA" \
			--weights ../../results/$model/weights/best.pt \
			--batch-size 1 \
			--device 0 \
			--img-size 320 \
			--int8 \
			--include tflite
		;;

	kd)
		[ $# -ne 2 ] && usage "Error: '$1' command requires a name argument."
		python train.py \
			--data "$DATA" \
			--cfg Qyolov5n.yaml \
			--weights '' \
			--batch-size 128 \
			--teacher_weight pts/v5m.pt \
			--device 0 \
			--img-size 320 \
			--epochs "$EPOCHS" \
			--exist-ok \
			--name "$2"
		;;

	single)
		[ $# -lt 2 ] && usage "Error: '$1' command requires a name argument."
		[ $# -lt 3 ] && usage "Error: '$1' command requires a model argument."
		python train.py \
			--data "$DATA" \
			--cfg $3 \
			--weights '' \
			--batch-size 128 \
			--device 1 \
			--img-size 320 \
			--epochs "$EPOCHS" \
			--exist-ok \
			--name "$2"
		;;

	val)
		[ $# -ne 2 ] && usage "Error: '$1' command requires a model argument."
		model=$2
		python val.py \
			--data "$DATA" \
			--weights ../../results/$model/weights/best.pt \
			--device 0 \
			--img-size 320 \
			--task test \
			--name "$model" \
			--batch-size 128
		;;

	*)
		usage "Error: Unknown command '$1'"
		;;
esac
