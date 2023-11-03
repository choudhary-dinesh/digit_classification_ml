#command to run under project dir to build docker images 
sudo docker build -t docker_assigment:v1 -f docker/Dockerfile .

#docker run command that mount volume, run it in interactive mode and take python script args as well to trai svm model
sudo docker run -v /home/dineshkumar/Codes/digit-classification/models:/digits/models -it docker_assigment:v1 --total_run 1 --dev_size 0.2 --test_size 0.2 --model_type 'svm' 

#same docker run command to train tree model
sudo docker run -v /home/dineshkumar/Codes/digit-classification/models:/digits/models -it docker_assigment:v1 --total_run 1 --dev_size 0.2 --test_size 0.2 --model_type 'tree'

