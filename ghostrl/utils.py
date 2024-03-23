import matplotlib.animation as animation
import matplotlib.pyplot as plt


def combine_frames(frames, gif_dest=None):
    """
    Takes a list of frames - results of env.render() from environments with
    render_mode='rgb_array', and combines them into a recording.
    Displays the recording or saves it as a gif if git_dest is provided.
    """
    fig = plt.figure()
    plt.axis('off')
    ims = []
    for frame in frames:
        im = plt.imshow(frame, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(
        fig, ims, interval=400, blit=True, repeat_delay=100)

    # save animation
    if gif_dest is not None:
        ani.save(gif_dest, writer='pillow', fps=2)
    else:
        plt.show()

