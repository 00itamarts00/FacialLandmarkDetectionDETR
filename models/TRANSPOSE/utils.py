
def get_transpose_params(params):
    cpy = params.copy()
    backbone = params['transpose_args']['backbone']
    cpy['transpose_args']['backbone_params'] = params['transpose_args']['backbone_params'][backbone]
    return cpy
