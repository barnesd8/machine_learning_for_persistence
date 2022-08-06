from PIL import Image, ImageDraw
from sklearn import datasets
import matplotlib.pyplot as plt
import io

def create_data():
    n_samples = 100
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    X,y = noisy_circles
    OC = X[y==0]
    return OC

def balls(r, OC):
    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    ax.scatter(OC[:, 0], OC[:, 1], s=10, c='black', marker=".")
    n = OC.shape[0]
    for i in range(0, n):
        circle = plt.Circle(OC[i], radius=r/2, color='purple', alpha=.25)
        ax.add_patch(circle)

    plt.axis('off') 
    return fig

def make_gif():
    frames = []
    OC = create_data()
    radii = [.05, .075, .1, .125, .15,.175, .2,.225, .25,.275, .3,.325, .35,.375, .4,.425, .45,.475, .5, .475, .45, .425, .4, .375, .35, .325, .3, .275, .25, .225, .2, .175, .15, .125, .1, .075, .05]
    for r in radii:
        img_buf = io.BytesIO()
        balls(r, OC).savefig(img_buf, format='png')
        im = Image.open(img_buf)
        frames.append(im)
        
    frame_one = frames[0]
    frame_one.save("balls.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)
    
if __name__ == "__main__":
    make_gif()