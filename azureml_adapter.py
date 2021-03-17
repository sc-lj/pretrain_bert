import os


def set_environment_variables_for_nccl_backend(addr, master_port, rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = str(addr)

    # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(master_port)

    # TODO make this parameterizable
    os.environ['NCCL_SOCKET_IFNAME'] = '^docker0,lo'

    print('RANK = {}'.format(os.environ['RANK']))
    print('WORLD_SIZE = {}'.format(os.environ['WORLD_SIZE']))
    print('MASTER_ADDR = {}'.format(os.environ['MASTER_ADDR']))
    print('MASTER_PORT = {}'.format(os.environ['MASTER_PORT']))


def get_local_rank():
    return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

def get_global_size():
    return int(os.environ['OMPI_COMM_WORLD_SIZE'])

def get_local_size():
    return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])	

def get_world_size():
    return int(os.environ['WORLD_SIZE'])
	 
