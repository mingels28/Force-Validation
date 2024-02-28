using WaterLily, StaticArrays, CUDA
using Statistics: mean
using ParametricBodies 
@assert CUDA.functional();

#define NACA airfoil
function NACAfoil(p=6;θ=0,Re=1e3,U=1,T=Float32,mem=Array)
    n = 2^p
    L = T(n/4)

    # Map from simulation coordinate x to surface coordinate ξ
    nose = SA[T(n),T(n)]     #translation relative to nose [x,y]
    θ    = T(θ)              #angle of attack [rad]

    R = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]   #rotation matrix

    function map(x,t)
        ξ = R*(x-nose)              #translate and rotate
        return SA[ξ[1],abs(ξ[2])]   #return new location 
    end

    #Define foil using NACA0012 profile equation: https://tinyurl.com/NACA00xx
    # NACA(s)   = 0.6f0*(0.2969f0s-0.126f0s^2-0.3516f0s^4+0.2843f0s^6-0.1036f0s^8)
    NACA(s)   = 0.6f0*(0.2969f0s-0.126f0s^2-0.3516f0s^4+0.2843f0s^6-0.1036f0s^8)
    foil(s,t) = L*SA[(1-s)^2,NACA(1-s)]
    
    body = ParametricBody(foil,(0,1);map,T=T,mem=mem)

    Simulation((2n,2n),(U,0),L;ν=U*L/Re,body,T,mem)
end

#function to increment time step and calculate forces
function get_force(sim, t)
	sim_step!(sim, t, remeasure=true, verbose=true);
	sz = size(sim.flow.p);
	df = ones(Float32, tuple(sz..., length(sz))) |> CuArray;
	return -WaterLily.∮nds(sim.flow.p, df ,sim.body, t*sim.L/sim.U)./(0.5*sim.L*sim.U^2)
end

""" simulation parameters """
duration = 200    #length of time series
step     = 0.05   #time step increment
Re       = 10000  #Reynolds number
p        = 8      #effective spatial resolution
α        = 10     #angle of attack [deg]

""" initialize the simulation """
sim  = NACAfoil(p;Re=Re,θ=α*π/180,mem=CuArray); 
t = range(sim_time(sim),sim_time(sim)+duration,step=step);

""" get forces """
forces = [get_force(sim,tᵢ) for tᵢ ∈ t]

#extract the forces components
Fx = [forces[x][1] for x ∈ range(1,length(forces))];
Fy = [forces[x][2] for x ∈ range(1,length(forces))];
