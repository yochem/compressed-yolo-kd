import os
import sys
import argparse
from pathlib import Path

if not os.environ.get("VIRTUAL_ENV"):
    print("activate virtual env first", file=sys.stderr)
    exit(1)


def export_command(args):
    import export

    args.batch_size = 1
    args.include = "tflite"
    export.run(**vars(args))


def train_command(args):
    if opt.colab and opt.teacher_weight is None:
        print('colab requires teacher weight/cfg', file=sys.stderr)
        exit(1)
    import train

    train.run(**vars(args))


def val_command(args):
    import val

    args.save_conf = True
    args.save_json = True
    args.task = "test"
    args.name = Path(args.weights).parent.parent.name
    val.run(**vars(args))


parser = argparse.ArgumentParser(description="YOLOv5 operations")
subparsers = parser.add_subparsers(dest="command", required=True)

# Common arguments
common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument("--data", default="params/data-bears.yaml")
common_parser.add_argument("--device", default="0", help="Device to use")
common_parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
common_parser.add_argument("--imgsz", type=int, default=320, help="Image size")

# Export subcommand
export_parser = subparsers.add_parser("export", parents=[common_parser])
export_parser.add_argument("--weights", required=True)
export_parser.add_argument("--int8", action="store_true", help="Use int8")
export_parser.set_defaults(func=export_command)

# Train subcommand
train_parser = subparsers.add_parser("train", parents=[common_parser])
train_parser.add_argument("--cfg", required=True, help="model config")
train_parser.add_argument("--hyp", default="params/hyp.yaml")
train_parser.add_argument("--weights", default="")
train_parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
train_parser.add_argument("--name", required=True, help="Name")
train_parser.add_argument("--teacher-weight", help="Teacher weight file")
train_parser.add_argument("--colab", action="store_true", help="KDCL mode")
train_parser.set_defaults(func=train_command)

# Val subcommand
val_parser = subparsers.add_parser("val", parents=[common_parser])
val_parser.add_argument("--weights", required=True)
val_parser.set_defaults(func=val_command)

args = parser.parse_args()
args.exist_ok = True

func = args.func
del args.func
del args.command
func(args)
