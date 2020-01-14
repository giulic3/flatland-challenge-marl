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
	
	limit = int(args[2])
	metrics = torch.load(args[1])
	rewards = metrics['rewards'][:limit] # list of lists
	#Qs = metrics['Qs'][:limit] # list of lists
	episodes = metrics['episodes'][:limit] # list
	
	rewards_mean = torch.tensor(rewards, dtype=torch.float32).mean(1).squeeze()
	#Qs_mean = torch.tensor(Qs, dtype=torch.float32).mean(1).squeeze()
	
	#plt.plot(episodes, rewards_mean.numpy(), 'r-', linewidth=1.5)
	plt.plot(episodes, smooth(rewards_mean.numpy(), 15), 'b-', linewidth=1.5)
	plt.xlabel("Episode")
	plt.ylabel("Reward")
	#plt.plot(episodes, Qs_mean.numpy(), 'r-', linewidth=1.5)
	#plt.plot(episodes, smooth(Qs_mean.numpy(), 30), 'g-', linewidth=1.5)
	plt.savefig("reward-smoothed.png")
	plt.show()

if __name__ == '__main__':
	main(sys.argv)