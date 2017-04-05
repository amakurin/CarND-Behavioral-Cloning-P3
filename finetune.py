import model as mdl

mdl.fine_tune_model(src_file_name='model.h5',
					tgt_file_name='model_tuned.h5',
					log_path = './data2/driving_log.csv', 
					img_path = './data2/IMG/')