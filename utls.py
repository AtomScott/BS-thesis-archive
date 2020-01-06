import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers
import numpy as np

def make_video(pose, out, fps=100):
    """
    Make a gif file animating the pose data.

    Args:
        pose: Pose data to animate
        out: path to save output gif

    Returns:
        None

    Raises:
        Exception: description
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('Time')

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, 2)

    graph, = ax.plot([], [], [],'ro')

    def update(i):
        print('{0}/{1}'.format(i, len(pose)), end='\r')
        graph.set_data(pose[i,:,0],pose[i,:,2])
        graph.set_3d_properties(pose[i,:,1])
        title.set_text('Time={0}'.format(i/fps))
        return graph, title

    anim = FuncAnimation(fig, update, frames=range(len(pose)),
                    blit=True)
    anim.save(out, writer='ffmpeg', fps=fps)
    plt.close()
