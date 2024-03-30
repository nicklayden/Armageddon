import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



def initial_density(r):
	# if p=8, similar to BTS collapse example.
	# horizon appears across the domain at similar times
	# if p=16, horizon forms at center and expands out
	p = 16
	return (1/p*np.pi)*np.exp(-4*r**2)



def Mprime(r,m):
	# Field equation for the mass function
	return [4*np.pi*initial_density(r)*r**2]

def t_b(r,m):
	# time of the big bang or big crunch for a 
	# marginally bound model
	return 1/(np.sqrt(2*m))*(2/3) * r**(3/2)


@np.vectorize
def R(t,r,m):
	# Exact solution for a marginally bound model
	return np.cbrt(9*m/2 * (t-t_b(r,m))**2)


def rdot(t,z,m):
	# Friedmann equation
	return -np.sqrt(2*m/z)

@np.vectorize
def M(r):
	# DD's mass function
	return r**3 

# Event functions that detect shell collapse and horizon formation
# the shell collapse is a terminal condition.
def full_collapse(t, y,m): return y[0]+1e-14
full_collapse.terminal = True

def horizon_event(t,y,m): return y[0]-2*m
horizon_event.terminal = False


# Simulation parameters
r_start = 1e-8
r_end = np.sqrt(1.6)
t_start = 0
t_end = 50
N_t = 100000
N_r = 2000

r_grid = np.linspace(r_start,r_end,N_r)
t_grid = np.linspace(0,t_end,N_t)

# Mass function from solving density field equation
mass = solve_ivp(Mprime,[0,r_end],[0],method="RK45",t_eval=r_grid)


# Schwarzschild exterior radius
sch_mass = mass.y[0][-1]
sch_rad  = 2*sch_mass

# Plot Schwarzschild radius for the exterior solution
plt.plot([sch_rad,sch_rad],[0,50],ls="--",c='k')

# capture all times and radii for which R-2M=0
horizon_events = []
horizon_radii = []
horizon_times = []

# For each worldline, solve the Friedmann equation
for i in range(0,len(r_grid),50):
	
	# Data for the current worldline
	shell_mass = mass.y[0][i]
	shell_radius = r_grid[i]

	# Solution to the Friedmann equation
	Radius = solve_ivp(rdot, [0,t_end], [shell_radius],
		args=[shell_mass],dense_output=True,
		t_eval=t_grid,events=[horizon_event,full_collapse])
	# Plot the solution curve for this worldline
	plt.plot(Radius.y[0],Radius.t,lw=0.5,c="blue")

	# Identify special events: Shell collapse (R=0), Horizon appearance (R=2M)
	if len(Radius.t_events)>1:
		for j in range(len(Radius.t_events)):
			horizon_times.append(Radius.t_events[j])
			plt.scatter(Radius.y_events[j],Radius.t_events[j],c="r")
	elif len(Radius.t_events == 1):
		horizon_times.append(Radius.t_events)

for i in range(len(horizon_times)):
	print(horizon_times[i])

# get time of last shell collapse
t_last_collapse = Radius.t[-1]

# Resize plot window and show
plt.ylim(0,t_last_collapse)
plt.xlim(0,r_end)
plt.show()


















