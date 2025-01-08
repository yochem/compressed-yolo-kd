#!/usr/bin/env bash

DATA=data/bears.yaml
EPOCHS=30
MODELDIR=pts

. ../venv/bin/activate

usage() {
	[ -n "$1" ] && { echo "$1"; echo ""; }
	echo "Usage: $0 <command> [args]"
	echo "Commands:"
	echo "  export <model>"
	echo "  kd <model> <name>"
	echo "  single <model> <name>"
	echo "  val <model>"
	exit 1
}

[ $# -eq 0 ] && usage
mkdir -p "$MODELDIR"

case "$1" in
	export)
		[ $# -ne 2 ] && usage "Error: '$1' command requires a model argument."
		model=$2
		python export.py \
			--device 0 \
			--img-size 320 \
			--data "$DATA" \
			--weights "$MODELDIR/$model/weights/best.pt" \
			--batch-size 1 \
			--int8 \
			--include tflite
		;;

	kd)
		[ $# -qt 2 ] && usage "Error: '$1' command requires a models/ path"
		[ $# -qt 3 ] && usage "Error: '$1' command requires a name"
		python train.py \
			--device 0 \
			--img-size 320 \
			--data "$DATA" \
			--cfg "$2" \
			--weights '' \
			--batch-size 128 \
			--teacher_weight "$MODELDIR/v5m.pt" \
			--epochs "$EPOCHS" \
			--exist-ok \
			--name "$3" && \
		cp -f "runs/train/$3/weights/best.pt" "$MODELDIR/$3.pt"
		;;

	single)
		[ $# -qt 2 ] && usage "Error: '$1' command requires a models/ path"
		[ $# -qt 3 ] && usage "Error: '$1' command requires a name"
		python train.py \
			--device 0 \
			--img-size 320 \
			--data "$DATA" \
			--cfg "$2" \
			--weights '' \
			--batch-size 128 \
			--epochs "$EPOCHS" \
			--exist-ok \
			--name "$3" && \
		cp -f "runs/train/$3/weights/best.pt" "$MODELDIR/$3.pt"
		;;

	val)
		[ $# -ne 2 ] && usage "Error: '$1' command requires a model argument."
		model=$2
		python val.py \
			--device 0 \
			--img-size 320 \
			--data "$DATA" \
			--weights "$MODELDIR/$model.pt" \
			--task test \
			--name "$model" \
			--batch-size 128
		;;

	*)
		usage "Error: Unknown command '$1'"
		;;
esac
