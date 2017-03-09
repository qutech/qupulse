class dummy_pytabor:
    pass

class dummy_pyvisa:
    pass

class dummy_teawg:
    model_properties_dict = dict()
    class TEWXAwg:
        def __init__(self, *args, **kwargs):
            pass
        send_cmd = __init__
        send_query = send_cmd
        select_channel = send_cmd
        send_binary_data = send_cmd
        download_sequencer_table = send_cmd
