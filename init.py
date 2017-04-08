import model as mdl
import argparse

parser = argparse.ArgumentParser(description='Initial training')
parser.add_argument('-model_version',
					default = 'default',
                    help='version of model')
parser.add_argument('-output_model',
					default = 'model.h5',
                    help='file name for resulting model')
parser.add_argument('-log_path',
					default = './data/driving_log.csv',
                    help='path to datalog')
parser.add_argument('-img_path',
					default = './data/IMG/',
                    help='path to images folder')
parser.add_argument('-mrg_log_path',
					default = None,
                    help='path to datalog')
parser.add_argument('-mrg_img_path',
					default = None,
                    help='path to images folder')
parser.add_argument('-mrg_rate',
					type=float,
					default = 0.3,
                    help='path to images folder')
parser.add_argument('-epochs',
					type=int,
					default = 5,
                    help='number of learning epochs')

args = parser.parse_args()

mdl.train_model(model_file_name=args.output_model,
				log_path = args.log_path, 
				img_path = args.img_path,
				mrg_log_path = args.mrg_log_path, 
				mrg_img_path = args.mrg_img_path,
				mrg_rate = args.mrg_rate,
				epochs = args.epochs,
				version = args.model_version)