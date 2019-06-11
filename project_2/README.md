run eval_rnn_VAR_bert.py 

The code expects as input a csv file (with header). The expected format is: first four 
columns of each row are input sentences and the last two columns are possible endings.

'''
python3 eval_rnn_VAR_bert.py --datapath '/path/to/csvfile' --path_to_modelfile '/path/to/learned_model.pth' 
  --savepath '/path/to/output.csv'
'''

Please note: code requires pytorch. It does not use gpu (no cuda).
