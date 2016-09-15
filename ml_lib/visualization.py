
import matplotlib.patches
import numpy as np

from ml_utils import mkdir_p


class Visualizable(object):
    def apply(self, plt):
        raise NotImplementedError()

    def distance(self, (x, y)):
        raise NotImplementedError()

    def get_text_on_click(self):
        return None

# WARNING: mess with x vs. y !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 Polygon vs Box
class Box(Visualizable):
    def __init__(self, xmin, xmax, ymin, ymax, edgecolor='yellow', linewidth=2, label=''):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.edgecolor = edgecolor
        self.linewidth = linewidth

    def apply(self, plt):
        print 'drawing', self.xmin, self.xmax, self.ymin, self.ymax
        plt.gca().add_patch(
            plt.Rectangle(xy=(self.ymin, self.xmin), width=self.ymax - self.ymin, height=self.xmax - self.xmin,
                          fill=False,
                          edgecolor=self.edgecolor, linewidth=self.linewidth)
            )

class BoxWithText(Visualizable):
    def __init__(self, xmin, xmax, ymin, ymax, text, edgecolor='yellow', linewidth=2, fontsize=10, boxstyle=None, pad=None):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.edgecolor = edgecolor
        self.linewidth = linewidth
        self.text = text
        self.fontsize = fontsize
        self.boxstyle = boxstyle
        self.pad = pad

    def apply(self, plt):
        print 'drawing', self.xmin, self.xmax, self.ymin, self.ymax, self.text
        ax = plt.gca()

        ax.add_patch(
            plt.Rectangle(xy=(self.ymin, self.xmin), width=self.ymax - self.ymin, height=self.xmax - self.xmin,
                          fill=False,
                          edgecolor=self.edgecolor, linewidth=self.linewidth)
            )

        ax.text(self.ymin, self.xmin, self.text, bbox=dict(facecolor=self.edgecolor, alpha=0.5, boxstyle=self.boxstyle, pad=self.pad),
                fontsize=self.fontsize, color='white')


class PolygonVisualizable(Visualizable):
    def __init__(self, xy_list, edgecolor='yellow', linewidth=2, fill=False, text_on_click=None):
        self.xy_list = xy_list
        self.edgecolor = edgecolor
        self.linewidth = linewidth
        self.fill = fill
        self.text_on_click = text_on_click

    def get_text_on_click(self):
        return self.text_on_click

    def distance(self, (x, y)):
        print self.xy_list
        xs = map(lambda a: a[0], self.xy_list)
        ys = map(lambda a: a[1], self.xy_list)
        mid_x = sum(xs) / len(xs)
        mid_y = sum(ys) / len(ys)
        return float((x - mid_x) ** 2 + (y - mid_y) ** 2)

    def apply(self, plt):
        print 'Polygon:apply'
        print self.xy_list
        ax = plt.gca()
        xy = np.asarray(self.xy_list)
        patch = matplotlib.patches.Polygon(xy, edgecolor=self.edgecolor, linewidth=self.linewidth,
                                           fill=self.fill)
        ax.add_patch(patch)



def visualize(im, visualizables, filepath=None, figure_name='', dpi=900):
    print ''
    import matplotlib.pyplot as plt

    plt.cla()
    # if hd:
    #     print 'hd', hd
    #     fig = matplotlib.pyplot.figure(figsize=(24.0, 15.0))

    fig = plt.gcf()
    fig.canvas.set_window_title(figure_name)
    # if resolution is not None:
    #     print 'resolution!!!', resolution
    #     plt.imshow(im, shape=resolution)
    # else:
    plt.imshow(im)
    for v in visualizables:
        v.apply(plt)


    def onclick(event):
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            closest_vis = None
            best_distance = np.inf
            for vis in visualizables:
                if vis.get_text_on_click() is not None:
                    dist = vis.distance((x, y))
                    if dist < best_distance:
                        best_distance = dist
                        closest_vis = vis
            if closest_vis is not None:
                print closest_vis.get_text_on_click()
            else:
                print 'No closest_vis'


    print 'show!!!!'
    import os

    if filepath is not None or (os.environ.get('PLOT_DIR', None) and figure_name):
        print 'KU!!!!!!!!! 0'
        if filepath is None:
            dir = os.environ.get('PLOT_DIR')
            import os
            mkdir_p(dir)
            filepath = os.path.join(dir, figure_name + '.png')

        print 'visualize to filepath:'
        print filepath
        plt.savefig(filepath, dpi=dpi)
    elif not os.environ.get('PLOT_DIR', None):
        print 'KU!!!!!!!!! 1'
        plt.ion()
        plt.show()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.pause(0.001)
        raw_input("<Hit Enter To Close>")
    else:
        pass


