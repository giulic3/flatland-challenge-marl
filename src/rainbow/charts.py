#https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
# Function to smooth a curve
import sys
import torch
import numpy as np 
import matplotlib.pyplot as plt

def smooth(y, box_pts):

	box = np.ones(box_pts)/box_pts
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth

def main(args):
	
	metrics = torch.load(args[1])
	rewards = metrics['rewards'] # list of lists
	Qs = metrics['Qs'] # list of lists
	episodes = metrics['episodes'] # list
	
	rewards_mean = torch.tensor(rewards, dtype=torch.float32).mean(1).squeeze()
	Qs_mean = torch.tensor(Qs, dtype=torch.float32).mean(1).squeeze()
	
	fig, (ax1, ax2) = plt.subplots(nrows=2)
	ax1.plot(episodes, rewards_mean.numpy(), 'b-', linewidth=1.5)
	#ax1.plot(episodes, smooth(rewards_mean.numpy(), 5), 'b-', linewidth=1.5)
	ax1.set(ylabel='Reward')
	ax1.set_title('Learning curve')
	ax2.plot(episodes, Qs_mean.numpy(), 'r-', linewidth=1.5)
	#ax2.plot(episodes, smooth(Qs_mean.numpy(), 5), 'r-', linewidth=1.5)
	ax2.set(xlabel='Episode', ylabel='Q')
	
	# Path where to save chart
	path = args[1].split('/')[:-1]
	print('/'.join(path))
	fig.savefig('/'.join(path)+ "/learning-curve")
	plt.close(fig)

if __name__ == '__main__':
	main(sys.argv)