gen_path='RESULT/epoch600'
real_path='data/CUB-200/images'

python3 fid_score_sub.py  $gen_path  $real_path --device cuda:0
              
			