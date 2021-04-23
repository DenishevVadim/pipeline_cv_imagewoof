import warnings
import argparse
from NN_tools.dataset import *
from NN_tools.nn_usage import *
from NN_tools.neural_nation import *
warnings.filterwarnings("ignore")


p = argparse.ArgumentParser()
#p.add_argument("-folderpath", "-fp", type=str, help="path to folder with images", required=True)
p.add_argument("-folderpath", "-fp", type=str, help="path to folder with images", required=False, default='/home/vadim/Downloads/imagewoof2-160/train/n02086240')
p.add_argument("-mode", "-m", choices=["train", "predict"], default="predict")
p.add_argument("-modelpath", "-mp", type=str, help="path to model", default="", metavar="")
p.add_argument("-outputpath", "-op", type=str, help="path to output for predict", default="", metavar="")
p.add_argument("-maxlr", "-mlr", type=float, help="max learning rate", default=0.01, metavar=0.01)
p.add_argument("-gradclip", "-gc", type=float, help="grad clip", default=0.1, metavar=0.1)
p.add_argument("-weightdecay", "-wd", type=float, help="weight decay", default=1e-4, metavar=1e-4)
p.add_argument("-epochs", "-e", type=int, help="number epochs", default=20, metavar=20)
p.add_argument("-batchsize", "-bs", type=int, help="batchsize", default=48, metavar=48)
args = p.parse_args()

classes = ['Australian terrier', 'Border terrier', 'Samoyed', 'Beagle', 'Shih-Tzu', 'English foxhound',
               'Rhodesian ridgeback', 'Dingo', 'Golden retriever', 'Old English sheepdog']

if args.mode == 'train':
    train_dl = get_dataset(args.folderpath + '/train', transform=get_train_transforms(),
                           batch_size=args.batchsize, shuffle=True)
    val_dl = get_dataset(args.folderpath + '/val', transform=get_val_transforms(),
                         batch_size=args.batchsize, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = Net(3, 10)
    model.to(device);
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_dl))

    model, loss = fit_model(model, optimizer, scheduler, device, train_dl=train_dl, val_dl=val_dl, epochs=args.epochs,
                            grad_clip=args.grad_clip)
    print(loss)
    torch.save(model.state_dict(), 'model_dict')
    torch.save(model, 'model_zip')
else:
    custom_dataset = CustomDataset(args.folderpath, transform=get_val_transforms(), is_labeled = False)
    test_dl = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=15, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = Net(3, 10)
    model.to(device);
    test = get_predict(model, device, test_dl)
