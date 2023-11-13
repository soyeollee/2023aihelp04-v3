#export OMP_NUM_THREADS=4
#export MKL_NUM_THREADS=4

python trainval.py # --run 00_aihelp --train-config 00_aihelp --val-config 00_aihelp --train-tf-config 00_aihelp --val-tf-config 00_aihelp
#python trainval.py --run 01_aihelp --train-config 00_aihelp --val-config 00_aihelp --train-tf-config 01_aihelp --val-tf-config 00_aihelp
#python trainval.py --run 00-1_aihelp --train-config 00-1_aihelp --val-config 00_aihelp --train-tf-config 00_aihelp --val-tf-config 00_aihelp
#python trainval.py --run 00-2_aihelp --train-config 00-2_aihelp --val-config 00_aihelp --train-tf-config 00_aihelp --val-tf-config 00_aihelp