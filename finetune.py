import model as mdl
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Initial training')
parser.add_argument('-model_version',
					default = 'default',
                    help='version of model')
parser.add_argument('-src_model',
					default = 'model.h5',
                    help='file name for initial model')
parser.add_argument('-tgt_model',
					default = 'model_tuned.h5',
                    help='file name for resulting model')
parser.add_argument('-log_path',
					default = './data2/driving_log.csv',
                    help='path to datalog')
parser.add_argument('-img_path',
					default = './data2/IMG/',
                    help='path to images folder')
parser.add_argument('-mrg_log_path',
					default = None,
                    help='path to mix datalog')
parser.add_argument('-mrg_img_path',
					default = None,
                    help='path to mix images folder')
parser.add_argument('-mrg_rate',
					type=float,
					default = 0.3,
                    help='mix rate - relative to main datacet')
parser.add_argument('-epochs',
					type=int,
					default = 5,
                    help='number of learning epochs')

args = parser.parse_args()

mdl.fine_tune_model(src_file_name=args.src_model,
					tgt_file_name=args.tgt_model,
					log_path = args.log_path, 
					img_path = args.img_path,
					mrg_log_path = args.mrg_log_path, 
					mrg_img_path = args.mrg_img_path,
					mrg_rate = args.mrg_rate,
					epochs = args.epochs,
					version = args.model_version)