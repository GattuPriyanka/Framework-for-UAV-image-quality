@ECHO OFF
echo y|conda create --name quality python=3
call activate quality
echo y|pip install -r requirements.txt
echo y|conda install -c anaconda mkl-service
echo y|conda install -c conda-forge mkl_random
echo y|conda install -c conda-forge mkl_fft
call deactivate