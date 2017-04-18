from itertools import cycle
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from numpy import size
from scipy.io import loadmat

# fig_contents = loadmat('1-roc.fig', squeeze_me=True, struct_as_record=False)
# matfig = fig_contents['hgS_070000']
# childs = matfig.children

def plot_mat_roc_curve(file_name):
    # Setup plot details
    color_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    color_names = [name for name, color in color_dict.items()]
    colors = cycle(color_names)
    lw = 2

    mat_file = loadmat(file_name, squeeze_me=True, struct_as_record=False)
    plot_data = mat_file['hgS_070000'].children
    # if size(plot_data) > 1:
    #     legs = plot_data[1]
    #     leg_entries = tuple(legs.properties.String)
    #     plot_data = plot_data[0]
    #     print leg_entries
    # else:
    #     legs=0
    

    plt.clf()
    plt.hold(True)
    counter = 0
    for line in plot_data.children:
        # line = plot_lines[index]
        x = line.properties.XData
        y = line.properties.YData
        
        plt.plot(x, y, lw=lw, color=color)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    # plt.legend(loc="lower left")
    plt.grid()
    plt.hold(False)
    plt.savefig('./plot_mat.png')


def main():
    file_name='2-roc.fig'
    plot_mat_roc_curve(file_name),

if __name__ == "__main__":
    main()