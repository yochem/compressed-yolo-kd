import os
import sys
import argparse
from pathlib import Path

if not os.environ.get('VIRTUAL_ENV'):
    print('activate virtual env first', file=sys.stderr)
    exit(1)


import train
import export
import val


def require(args, opt):
    if not getattr(args, opt):
        print(f'{args.operation} requires {opt}', file=sys.stderr)
        exit(2)


parser = argparse.ArgumentParser(description="YOLOv5 operations")
parser.add_argument(
    "operation",
    choices=["export", "kd", "single", "val"],
    help="Operation to perform",
)

parser.add_argument("--data", default="params/bear.yaml")
parser.add_argument("--cfg", help="model config")
parser.add_argument("--hyp", default="params/hyp.yaml")
parser.add_argument("--weights", default="")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
parser.add_argument("--device", type=int, default=0, help="Device to use")
parser.add_argument("--img-size", type=int, default=320, help="Image size")
parser.add_argument("--int8", action="store_true", help="Use int8")
parser.add_argument("--teacher_weight", default="pts/v5m.pt", help="Teacher weight file")
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument("--name", help="Name")

args = parser.parse_args()
args.exist_ok = True

if args.operation == "export":
    require(args, 'weights')
    args.batch_size = 1
    args.include = 'tflite'
    export.run(**vars(args))
elif args.operation == "kd":
    require(args, 'name')
    train.run(**vars(args))
elif args.operation == "single":
    require(args, 'name')
    require(args, 'cfg')
    args.teacher_weight = None
    train.run(**vars(args))
elif args.operation == "val":
    require(args, 'weights')
    args.task = "test"
    args.name = Path(args.weights).parent.parent.name
    val.run(**vars(args))
else:
    require(args, 'operation')
