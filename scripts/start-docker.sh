export workpath=$PWD
mkdir cms_open_data_ML
chmod -R 777 cms_open_data_ML
docker run -it --name my_ML -P -p 8888:8888 -v ${workpath}/cms_open_data_ML:/code gitlab-registry.cern.ch/cms-cloud/python-vnc:python3.10.12