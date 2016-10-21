def set_matplotlib_backend():
    import matplotlib
    import os
#    v = os.environ.get('MATPLOTLIB_BACKEND')
    matplotlib_backend = 'Agg'

    print 'running matplotlib hack!!, backend=', matplotlib_backend
    if matplotlib_backend:
        matplotlib.use(matplotlib_backend)

set_matplotlib_backend()

