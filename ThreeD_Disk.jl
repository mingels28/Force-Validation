using StaticArrays, WaterLily, CUDA, Plots
@assert CUDA.functional();
gr();

""" geometry function """
function disk(p=7;Re=1000,U=1,mem=Array,ϵ=0.1)
    n = 2^p                     
    L = n/20

    theta =0*π/180
    rot_mat = SA[1 0 0; 0 cos(theta) -sin(theta); 0 sin(theta) cos(theta)] #rotate about x-axis (pitch)
    # rot_mat = SA[cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)] #rotate about y-axis (roll)
    # rot_mat = SA[cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1]  #rotate about z-axis (yaw)

    #define 2norm function for convenience
    norm2(x) = √sum(abs2,x)

    #define segment sdf 
    function sdf(xyz,t)
        x,y,z = xyz
        r = norm2(SA[y,z]);
        norm2(SA[x,r-min(r,L)]) - 1.5
    end

    map(xyz,t) = rot_mat*(xyz - SA[n/2,n/2,n/2])
   
    #3D simulation
    Simulation((2n,n,n),(U,0,0),L;U,ν=U*L/Re,body=AutoBody(sdf,map),mem=mem,T=Float32,ϵ=ϵ) 
end

""" function to increment time step and calculate forces """
function get_force(sim,t)
	sim_step!(sim, t, remeasure=false, verbose=true)
	sz = size(sim.flow.p)
	df = ones(Float32, tuple(sz..., length(sz))) |> CuArray
	return -WaterLily.∮nds(sim.flow.p,df,sim.body,t*sim.L/sim.U)./(0.5*pi*sim/L^2/4*sim.U^2)
end

""" define the parameters for your simulation """
p        = 8     #effective grid resolution
Re       = 10000 #Reynolds number
step     = 0.05  #time step for recording data
duration = 20   #length of simulation

""" initialize the simulation """
sim  = disk(p;Re=Re,mem=CUDA.CuArray); 
t = range(sim_time(sim),sim_time(sim)+duration,step=step);

""" get forces """
forces = [get_force(sim,tᵢ) for tᵢ ∈ t]

#extract the forces components
Fx = [forces[x][1] for x ∈ range(1,length(forces))];
Fy = [forces[x][2] for x ∈ range(1,length(forces))];
