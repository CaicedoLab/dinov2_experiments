##### injectRank0Name.sh #####

!/bin/sh

# A prescript for the workers to inject 
# the address and port of the rdzv server 
# into their submit file.

# We assume this is in a file in the
# cwd named "rank0_contact"

read name port < rank0_contact

sed -e "s/SERVER_IP_ADDRESS_GOES_HERE/${name}/g" -e "s/SERVER_PORT_GOES_HERE/${port}/g" < gpuworker.sub.template > gpuworker.sub
#################
