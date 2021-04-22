import warnings
import argparse

warnings.filterwarnings("ignore")


p = argparse.ArgumentParser()
p.add_argument("-folderpath", "-fp", type=str, help="path to folder with images", required=True)
p.add_argument("-mode", "-m", choices=["train", "predict"], default="predict")
p.add_argument("-modelpath", "-mp", type=str, help="path to model", default="", metavar="")
p.add_argument("-outputpath", "-op", type=str, help="path to output for predict", default="", metavar="")
p.add_argument("-maxlr", "-mlr", type=float, help="max learning rate", default=0.01, metavar=0.01)
p.add_argument("-gradclip", "-gc", type=float, help="grad clip", default=0.1, metavar=0.1)
p.add_argument("-weightdecay", "-wd", type=float, help="weight decay", default=1e-4, metavar=1e-4)
p.add_argument("-epochs", "-e", type=int, help="number epochs", default=20, metavar=20)
p.add_argument("-batchsize", "-bs", type=int, help="batchsize", default=48, metavar=48)
print(p.parse_args().mode)
