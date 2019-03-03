import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys




fig = plt.figure()

print("Would you like to visualize the annotated version of this data ? (y/n)")
annotate = (input().lower() == 'y')
if (annotate):
    try:
        csv_data = np.genfromtxt('./data/annotated/{0}'.format(sys.argv[1]), delimiter=',')
        if(csv_data.any()):
            columns = csv_data.shape[1]
            if(columns == 16): 
                print("Visualizing annotated data")
                annotate = True; 
                csv_data = genfromtxt('./data/annotated/{0}'.format(sys.argv[1]), delimiter=',')
    except IOError:
        annotate = False 
        print("IOError")
        print("This data has not been annotated yet.")
        csv_data = genfromtxt('./data/converted/{0}'.format(sys.argv[1]), delimiter=',')
else: 
    csv_data = genfromtxt('./data/converted/{0}'.format(sys.argv[1]), delimiter=',')


# data = csv_data[1:, 2:17]
data = csv_data

# data is a n x 15 array
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
    global vis_data, data, i, vis_rows, im, rows, ani, pause, annotate
    try:
        vis_data[:,:] = data[i:i + vis_rows, :]
    except:
        plt.close(fig)
        sys.exit()

    if(annotate and data[i,-1] == 1): 
        plt.title("Positive Annotation")
    else: 
        plt.title("Negative Annotation")


    im.set_array(vis_data)
    i += 1
    return im,

fig.canvas.mpl_connect('button_press_event', onClick)
ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)
plt.show()