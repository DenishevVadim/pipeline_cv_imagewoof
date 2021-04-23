import warnings
import argparse
from NN_tools.dataset import *
from NN_tools.nn_usage import *
from NN_tools.neural_nation import *
warnings.filterwarnings("ignore")


p = argparse.ArgumentParser()
p.add_argument("-folderpath", "-fp", type=str, help="path to folder with images", required=True)
p.add_argument("-mode", "-m", choices=["train", "predict"], default="predict")
p.add_argument("-modelname", "-mn", type=str, help="name to model, default basemodel", default="basemodel", metavar="basemodel")
p.add_argument("-outputname", "-on", type=str, help="name to output for predict /data/name.json", default="default", metavar="default")
p.add_argument("-maxlr", "-mlr", type=float, help="max learning rate", default=0.01, metavar=0.01)
p.add_argument("-gradclip", "-gc", type=float, help="grad clip", default=0.1, metavar=0.1)
p.add_argument("-weightdecay", "-wd", type=float, help="weight decay", default=1e-4, metavar=1e-4)
p.add_argument("-epochs", "-e", type=int, help="number epochs", default=20, metavar=20)
p.add_argument("-batchsize", "-bs", type=int, help="batchsize", default=16, metavar=16)
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
    optimizer = optim.SGD(model.parameters(), lr=args.maxlr, weight_decay=args.weightdecay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_dl))

    model, loss = fit_model(model, optimizer, scheduler, device, train_dl=train_dl, val_dl=val_dl, epochs=args.epochs,
                            grad_clip=args.gradclip)
    print(loss)
    torch.save(model.state_dict(), 'model_dict.pt')
else:
    custom_dataset = CustomDataset(args.folderpath, transform=get_val_transforms(), is_labeled = False)
    test_dl = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=args.batchsize, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = Net(3, 10)
    model.load_state_dict(torch.load(f"data/{args.modelname}.pt", map_location=device))
    model.to(device);
    output = get_predict(model, device, test_dl)
    output.to_json(f"data/{args.outputname}.json")
