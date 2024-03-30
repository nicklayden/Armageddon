import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

@np.vectorize
def Ueqn(Mi,Ri,Api,Api1,Rpi,Rpi1,Rdi,dt,x):
  a = 2* Mi/Ri
  b = - 1 + np.exp(dt/2 * (Api/Rpi*Rdi + Api1/Rpi1* x))
  return a+b - x**2



density = np.genfromtxt("Results/density.txt")
rgrid = np.genfromtxt("Results/r_grid.txt")
mass = np.genfromtxt("Results/Mass.txt")
mass_prime = np.genfromtxt("Results/Mass_prime.txt")
Ainit = np.genfromtxt("Results/A_metric.txt")
Rdot=np.genfromtxt("Results/R_dot.txt")
R_metric = np.genfromtxt("Results/R_metric.txt")
A_prime = np.genfromtxt("Results/A_metric_prime.txt")
R_prime = np.genfromtxt("Results/R_prime.txt")
rho = np.genfromtxt("Results/rho.txt")
mass_dot = np.genfromtxt("Results/mass_dot.txt")
B_metric= np.genfromtxt

kappa = np.pi*4

r_s = 3.0
N_space = 200

dt = 0.0025
Nt = 86
t_end = dt*Nt

t_grid = np.arange(0,Nt,dt)
t_grid = np.linspace(0,t_end,Nt)
r_grid = np.linspace(0,r_s,N_space)

x_data = r_grid
data_plotted = R_metric


# plt.plot(R_metric[0],data_plotted[1,:],lw=1.5,c='b', label="Initial Data")
# for i in range(1,Nt-1,5):
# 	plt.plot(R_metric[i],data_plotted[i,:],lw=0.5,c='b',alpha=0.2)
# plt.plot(R_metric[-1],data_plotted[-1,:],lw=1.5,c='r', label=r"$R(t_{44},r)$")
# plt.title(r"  $R(t,r), \omega=0$,dt=0.0025,dr=3/200.")
# plt.xlabel("R [geometric units]")
# plt.ylabel(r"$\tilde{\rho}$ [geometric units]")
# plt.legend(loc="upper right")


plt.plot(data_plotted[:,0],t_grid,lw=1.5,c='b', label="Initial Data")
for i in range(1,N_space-1,4):
  plt.plot(data_plotted[:,i],t_grid,lw=0.5,c='b',alpha=0.8)
plt.plot(data_plotted[:,-1],t_grid,lw=1.5,c='r', label=r"$R(t_{44},r)$")

# plt.yscale('log')
# plt.xscale('log')

# plt.show()




# Very minimal code to create the animation. Honestly it never really gets any 'cleaner' than this in Python in my experience. 
# Some things are just not as nice in python as they are in Maple. And really that's just because all the ugly parts are hidden away in Maple haha
# fig, ax = plt.subplots();
# line, = ax.plot([],[],c='b',lw=2);

# def animate(i):
#     y = data_plotted[i];
#     line.set_data(rgrid,y);
#     if i == 0:
#         line.set_label("test")
#     return line,

# def init():
#     line.set_data([],[]);
#     return line,

# # ax.plot(rgrid,u_inf(x,c,d),c='r',lw=2);
# ax.set_xlim(0,3);
# ax.set_ylim(-2,0);
# anim = FuncAnimation(fig,animate,np.arange(1,N-1));
# ax.legend();

# print(mass[0,j],Rinit[0,j],A_prime[0,j],A_prime[1,j],R_prime[0,j],R_prime[1,j],Rdot[j])
# print(Ueqn(mass[0,j],Rinit[0,j],A_prime[0,j],A_prime[1,j],R_prime[0,j],R_prime[1,j],Rdot[0,j],dt,0))
# print(Ueqn(mass[0,j],Rinit[0,j],A_prime[0,j],A_prime[1,j],R_prime[0,j],R_prime[1,j],Rdot[0,j],dt,-1))
# plt.plot(u_grid,Ueqn(mass[0,j],Rinit[0,j],A_prime[0,j],A_prime[1,j],R_prime[0,j],R_prime[1,j],Rdot[0,j],dt,u_grid))


# plt.plot(t_grid,density[:,0],lw=1.5)
# plt.title("Central Density Evolution")
# plt.xlabel("t [geometric units]")
# plt.ylabel(r"$\rho(t,0)$ [geometric units]")

# plt.show()



def initial_density(r):
  # if p=8, similar to BTS collapse example.
  # horizon appears across the domain at similar times
  # if p=16, horizon forms at center and expands out
  p = 16.0
  return (1/(p*np.pi))*np.exp(-4.0*(r**2))



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
r_start = r_grid[0]
r_end = r_grid[-1]
t_start = 0
# t_end = 50
N_t = 100000
N_r = 200

# r_grid = np.linspace(r_start,r_end,N_r)
t_grid = np.linspace(0,t_end,N_t)

# Mass function from solving density field equation
# mass = solve_ivp(Mprime,[0,r_end],[0],method="RK45",t_eval=r_grid)


# Schwarzschild exterior radius
# sch_mass = mass.y[0][-1]
# sch_rad  = 2*sch_mass

# Plot Schwarzschild radius for the exterior solution
# plt.plot([sch_rad,sch_rad],[0,50],ls="--",c='k')

# capture all times and radii for which R-2M=0
horizon_events = []
horizon_radii = []
horizon_times = []

# # For each worldline, solve the Friedmann equation
# for i in range(1,len(r_grid),10):
  
#   # Data for the current worldline
#   shell_mass = mass[0,i]
#   shell_radius = r_grid[i]

#   # Solution to the Friedmann equation
#   Radius = solve_ivp(rdot, [0,t_end], [r_grid[i]],
#     args=[shell_mass],dense_output=True,
#     t_eval=t_grid,events=[horizon_event,full_collapse],
#     method="RK45")
#   # Plot the solution curve for this worldline
#   plt.plot(Radius.y[0],Radius.t,lw=0.5,c="k",ls='--')

#   # Identify special events: Shell collapse (R=0), Horizon appearance (R=2M)
#   if len(Radius.t_events)>1:
#     for j in range(len(Radius.t_events)):
#       horizon_times.append(Radius.t_events[j])
#       # plt.scatter(Radius.y_events[j],Radius.t_events[j],c="r")
#   elif len(Radius.t_events == 1):
#     horizon_times.append(Radius.t_events)


# get time of last shell collapse
# t_last_collapse = Radius.t[-1]

# Resize plot window and show
# plt.ylim(0,t_last_collapse)
# plt.xlim(0,r_end)
# plt.show()



# plt.yscale('log')
# plt.xscale('log')

# plt.ylim(0,0.67)
# plt.xlim(0,r_start)
plt.show()

























