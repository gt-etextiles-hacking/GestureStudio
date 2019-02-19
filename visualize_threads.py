import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

fig = plt.figure()
csv_data = genfromtxt(sys.argv[1], delimiter=',')
data = csv_data[1:, 2:17]
rows, cols = data.shape

vis_rows = cols * 3
vis_data = np.random.rand(vis_rows, cols)

im = plt.imshow(vis_data, animated=True)
i = 0

ani = None
pause = False

# allows you to pause/play by clicking on figure
def onClick(event):
    global pause
    pause ^= True

    if pause:
        ani.event_source.stop()
    else:
        ani.event_source.start()

# animation update function
def updatefig(*args):
    global vis_data, data, i, vis_rows, im, rows, ani, pause
    try:
        vis_data[:,:] = data[i:i + vis_rows, :]
    except:
        plt.close(fig)
        sys.exit()

    im.set_array(vis_data)
    i += 1
    return im,

fig.canvas.mpl_connect('button_press_event', onClick)
ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)
plt.show()