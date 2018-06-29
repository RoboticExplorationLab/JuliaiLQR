using iLQR
using PendulumDynamics
using DoublePendulumDynamics
using Plots

## Simple Pendulum iLQR

#initial and goal conditions
x0 = [0.; 0.]
xf = [pi; 0.] # (ie, swing up)

#costs
Q = 1e-5*eye(PendulumDynamics.state_dim)
Qf = 25.*eye(PendulumDynamics.state_dim)
R = 1e-5*eye(PendulumDynamics.control_dim)

#simulation
dt = 0.1
tf = 1.

X, U = iLQR.solve(x0,PendulumDynamics.control_dim,PendulumDynamics.f,PendulumDynamics.Df,Q,R,Qf,xf,dt,tf)

P = plot(linspace(0,tf,size(X,2)),X[1,:])
P = plot!(linspace(0,tf,size(X,2)),X[2,:],title="Simple Pendulum (iLQR)",xlabel="Time")
display(P)

## Double Pendulum iLQR

#initial and goal conditions
x0 = [0.;0.;0.;0.]
xf = [pi;0.;0.;0.]

#costs
Q = 1e-5*eye(DoublePendulumDynamics.state_dim)
Qf = 25.*eye(DoublePendulumDynamics.state_dim)
R = 1e-5*eye(DoublePendulumDynamics.control_dim)

#simulation
dt = 0.1
tf = 1.

X, U = iLQR.solve(x0,DoublePendulumDynamics.control_dim,DoublePendulumDynamics.f,DoublePendulumDynamics.Df,Q,R,Qf,xf,dt,tf)

P = plot(linspace(0,tf,size(X,2)),X[1,:])
P = plot!(linspace(0,tf,size(X,2)),X[2,:],title="Double Pendulum (iLQR)",xlabel="Time")
display(P)
