def get_model_create(model_name):
    module_name = 'rl.models.' + model_name
    mod = __import__(module_name, fromlist=[''])
    return getattr(mod, 'create')
