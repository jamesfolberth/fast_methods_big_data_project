#! /usr/bin/env bash
rm -f nohup.out & nohup matlab -nojvm -nodisplay -nodesktop -nosplash -r "fixed_mat_driver(); quit;" -logfile logfile.out < /dev/null &
