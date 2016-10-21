# NOTE(maciekk): We should abandon this idea.

# from TheanoLib.modules import vcopy
#
# __author__ = 'maciek'
#
#
# class Module(object):
#     def __init__(self, name=''):
#         self.name = name
#
#     # NOTE:
#     # We should not declare any shared vars here, because when we call apply twice, we will get two
#     # separate vars, which is not what we often want, shared vars should be created in __init__
#     def apply(self, v, **kwargs):
#         """
#         This should be redefined for every module.
#         :param input:
#         :param kwargs:
#         :return:
#         """
#         raise NotImplementedError
#
#
# class Sequential(Module):
#     def __init__(self, name='', modules=[]):
#         super(Sequential, self).__init__(name)
#         self.modules = []
#         for module in modules:
#             self.add(module)
#
#     def add(self, module):
#         self.modules.append(module)
#         return module
#
#     def __getitem__(self, item):
#         return self.modules[item]
#
#     def apply(self, v, **kwargs):
#         for module in self.modules:
#             v = module.apply(v, **kwargs)
#         output_v = vcopy(v)
#         return output_v
#
#
#
# class DenseMul(Module):
#     def __init__(self, n_input, n_output, W_init=None, W_lr=None, b_lr=None, b_init=Constant(0.0), name='',
#                  numpy_rng=None):
#
#         if W_init is None:
#             W_bound = np.sqrt(1. / n_input)
#             W_init = Uniform(range=W_bound)
#
#         super(DenseMul, self).__init__(name)
#         self.n_input = n_input
#         self.n_output = n_output
#         self.numpy_rng = numpy_rng
#         self.W_init = W_init
#         self.b_init = b_init
#         self.W_lr = W_lr
#         self.b_lr = b_lr
#
#     def _allocate_params(self):
#         self.W = self.create_shared('W', self.W_init, (self.n_input, self.n_output))
#         self.b = self.create_shared('b', self.b_init, (self.n_output,))
#
#         self.params = [self.W, self.b]
#
#         self.opt_params = [Bunch(param=self.W, lr=self.W_lr),
#                            Bunch(param=self.b, lr=self.b_lr)]
#
#         self.reg_params = [self.W]
#
#
#     def apply(self, v, **kwargs):
#         input = v.output
#         output = T.dot(input, self.W) + self.b
#         nv = vcopy(v)
#         nv.update(output=output)
#         return self.post_apply(nv, **kwargs)
#
#     def __str__(self):
#         d = [('name', self.name),
#              ('n_input', self.n_input),
#              ('n_output', self.n_output),
#              ('W_init', str(self.W_init)),
#              ('b_init', str(self.b_init))
#         ]
#
#         return 'DenseMul ' + utils.list_desc(d)
#
#     def ftf_cost(self, p=2):
#         g = gram_matrix_flat(self.W)
#         return ftf_cost(g, p=p)
#

