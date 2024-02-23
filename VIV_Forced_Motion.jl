using WaterLily, StaticArrays, ForwardDiff, FindPeaks1D, LinearAlgebra, CUDA
using Statistics: mean
@assert CUDA.functional();
gr();


""" set up a function for your 3D model """
function VIV_3D(p=8;Re=4000,U=1,Vr=5,Ay_D=0.4,Ax_D=0.0,ϕ=0,mem=Array)
    #define simulation size, geometry dimensions, viscosity
    n = 2^p
    center,D = SA[n/2,n/2,n/4], n/20

    #sdf for a cylinder
    norm2(x) = √sum(abs2,x)
    function sdf(xyz,t)
        x,y,z = xyz - center
        norm2(SA[x,y]) - D/2
    end

    #motion
    function map(xyz,t)
        ω = 2π*U/(Vr*D)
        xyz .- SA[Ax_D*D*sin(2*ω*t - ϕ), Ay_D*D*sin(ω*t), 0]
    end

    #initialize simulation
    Simulation((2n,n,Int(n/2)),(U,0,0),D;ν=U*D/Re,body=AutoBody(sdf,map),mem,T=Float32)
end

""" set up a function for your 2D model """
function VIV_2D(p=8;Re=4000,U=1,Vr=5,Ay_D=0.4,Ax_D=0.0,ϕ=0,mem=Array)
    #define simulation size, geometry dimensions, viscosity
    n = 2^p
    center,D = SA[n/2,n/2], n/20

    #sdf for a cylinder
    norm2(x) = √sum(abs2,x)
    function sdf(x,t)
        norm2(x.-center) - D/2
    end

    #motion
    function map(x,t)
        ω = 2π*U/(Vr*D)
        x .- SA[Ax_D*D*sin(2*ω*t - ϕ), Ay_D*D*sin(ω*t)]
    end

    #initialize simulation
    Simulation((2n,n),(U,0),D;ν=U*D/Re,body=AutoBody(sdf,map),mem,T=Float32)
end

 
""" define the parameters for your simulation """
p        = 8;        #grid resolution
n        = 2^p;      #grid size
Ay_D     = 1.175;    #CF amplitude scaled by diameter
H        = n/2;      #cylinder span length
Vr       = 8.0;      #reduced velocity
Re       = 4000;     #Reynolds number
U        = 1;        #free stream velocity
ω        = 2π/Vr;    #frequency of motion
step     = 0.05;     #time step for simulation
duration = 20Vr;     #end time for simulation

#initialize simulation and run for duration before recording data
sim = VIV_2D(p;Vr=Vr,Ay_D=Ay_D,Re=Re,U=1,mem=CUDA.CuArray);
sim_step!(sim,duration-step;remeasure=true,verbose=true);
t = sim_time(sim) .+ range(0,duration;step=step);

#preallocate some memory for forces and flow viz if desired
Fx = zeros(size(t)) |> CuArray;     #drag
Fy = zeros(size(t)) |> CuArray;     #lift

#main loop for getting force, vorticity, tke, etc.
for i ∈ range(1,length(t))
    #incriment simulation in time
    sim_step!(sim,t[i];remeasure=true,verbose=true);

    #get drag and lift
    df = ones(Float32,tuple(size(sim.flow.p)...,length(size(sim.flow.p)))) |> CuArray;
    Fx[i],Fy[i] = -WaterLily.∮nds(sim.flow.p,df,sim.body,t[i]*sim.L/sim.U)./(0.5*sim.L*H*sim.U^2);
end

#different scaling for 2D or 3D and make save file name
if length(size(sim.flow.p)) == 2
    Fx = Fx.*H;
    Fy = Fy.*H;

    save_name = string("./data/JLD2_files/VIV2D_Vr",Vr,"_AyD",Ay_D,"_Re",Re,".jld2");
else
    save_name = string("./data/JLD2_files/VIV3D_Vr",Vr,"_AyD",Ay_D,"_Re",Re,".jld2");
end

#convert back to regular arrays
Fx = Fx |> Array;
Fy = Fy |> Array;    

#define the motion
y = [Ay_D*sim.L*sin(ω*x) for x ∈ t];

#motion velocity
Vy = diff(y)./diff(t);
Vy = append!(Vy,Vy[end]);

#motion acceleration
Ay = diff(Vy)./diff(t);
Ay = append!(Ay,Ay[end]);

#de-mean the lift
Fy = Fy .- mean(Fy);

#use motion to define cycles
pks = findpeaks1d(y);

#linearized phased forces
CLv = mean([sqrt(2/(length(Fy[pks[1][i]:pks[1][i+1]])))*(Fy[pks[1][i]:pks[1][i+1]]⋅Vy[pks[1][i]:pks[1][i+1]]) / 
                                                    sqrt(Vy[pks[1][i]:pks[1][i+1]]⋅Vy[pks[1][i]:pks[1][i+1]])   
                                                    for i ∈ range(1,length(pks[1])-1)]);

CLa = mean([sqrt(2/(length(Fy[pks[1][i]:pks[1][i+1]])))*(Fy[pks[1][i]:pks[1][i+1]]⋅Ay[pks[1][i]:pks[1][i+1]]) / 
                                                    sqrt(Ay[pks[1][i]:pks[1][i+1]]⋅Ay[pks[1][i]:pks[1][i+1]])   
                                                    for i ∈ range(1,length(pks[1])-1)]);
CEA = (Vr^2/(2π^3))*(CLa/Ay_D);