using StaticArrays, WaterLily, CUDA, Plots
@assert CUDA.functional();
gr();

""" geometry function """
function cylinder(p=8;Re=1000,U=1,mem=Array,ϵ=0.1)
    n = 2^p                     #number of grid points
    L = n/20                    #characteristic geometry size
    center = SA[n/2,n/2,n/4]    #center of 3D domain

    #define norm2 function for convenience
    norm2(x) = √sum(abs2,x)

    #cylinder SDF
    function sdf(xyz,t)
        x,y,z = xyz - center    
        norm2(SA[x,y])-L/2      
    end

    #3D simulation
    Simulation((2n,n,Int(n/2)),(U,0,0),L;U,ν=U*L/Re,
        body=AutoBody(sdf),mem=mem,T=Float32,ϵ=ϵ) 
end

""" function to increment time step and calculate forces """
function get_force(sim,t)
	sim_step!(sim, t, remeasure=false, verbose=true)
	sz = size(sim.flow.p)
	df = ones(Float32, tuple(sz..., length(sz))) |> CuArray
	return -WaterLily.∮nds(sim.flow.p,df,sim.body,t*sim.L/sim.U)./(0.5*sim.L*sz[3]*sim.U^2)
end

""" define the parameters for your simulation """
p        = 8     #effective grid resolution
Re       = 10000 #Reynolds number
step     = 0.05  #time step for recording data
duration = 10   #length of simulation

""" initialize the simulation """
sim  = cylinder(p;Re=Re,mem=CUDA.CuArray); 
t = range(sim_time(sim),sim_time(sim)+duration,step=step);

""" get forces """
forces = [get_force(sim,tᵢ) for tᵢ ∈ t]

#extract the forces components
Fx = [forces[x][1] for x ∈ range(1,length(forces))];
Fy = [forces[x][2] for x ∈ range(1,length(forces))];
