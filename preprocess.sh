# python preprocess.py --filter_fn livingroom
# python preprocess.py --filter_fn library
# python preprocess.py --filter_fn diningroom

python rerender_roomshape.py --filter_fn bedroom --data_filter val
python rerender_roomshape.py --filter_fn livingroom --data_filter val
python rerender_roomshape.py --filter_fn diningroom --data_filter val